#!/usr/bin/env python3
"""Fetch the real HumanEval and MBPP problem sets.

"The HumanEval/MBPP runner must load the intended HumanEval/MBPP problems."

Downloads the canonical, published task sets from their upstream sources,
normalises each record into this repository's ``EvalTask`` on-disk schema
(the same shape ``evaluation/{humaneval,mbpp}/sample_tasks.jsonl`` already
use), and writes them alongside a provenance manifest recording exactly
where the bytes came from and their SHA-256, so a reviewer can verify this is
the real benchmark and not a hand-edited stand-in.

Sources
-------
HumanEval (164 problems, Chen et al. 2021)
    ``openai/human-eval``, ``data/HumanEval.jsonl.gz``. MIT licence. Used
    unfiltered — every field maps 1:1 onto ``EvalTask``.

MBPP (Austin et al. 2021)
    ``google-research/google-research``, ``mbpp/mbpp.jsonl``. CC-BY-4.0.
    The paper's convention reserves task_id 1-10 as few-shot prompt examples
    and 11-510 as the evaluation set; this script keeps only that range (500
    problems) by default. MBPP's ``code`` field is a complete function
    (``def name(...): ...``), but ``EvalTask.canonical_solution`` must be
    only the *body* (the adapter supplies the ``def`` line itself) — records
    whose solution does not open with a single top-level ``def`` line, or
    whose entry point cannot be read off ``test_list[0]``, are skipped and
    counted rather than mis-parsed.

Offline-first is preserved
---------------------------
Nothing else in this repository requires this script to have been run:
``evaluation.base.BenchmarkAdapter._resolve_tasks_path`` already falls back
to the bundled ``sample_tasks.jsonl`` (with a loud log warning) whenever the
configured ``tasks_path`` is missing. A fresh clone with no network access
runs every test and every other script exactly as it did in Week 1/2.

Usage::

    python scripts/fetch_benchmark_data.py
    python scripts/fetch_benchmark_data.py --benchmark humaneval
    python scripts/fetch_benchmark_data.py --force
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, run_cli, setup_logging
from utils.errors import ClaspP5Error
from utils.io_utils import write_json, write_jsonl
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.timing import utc_timestamp

_LOG = get_logger(__name__)

_HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
_HUMANEVAL_LICENSE = "MIT (openai/human-eval)"

_MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
_MBPP_LICENSE = "CC-BY-4.0 (google-research/google-research, mbpp)"
#: Austin et al. 2021 convention: 1-10 are few-shot prompt examples, the
#: evaluation set is 11-510.
_MBPP_EVAL_RANGE = range(11, 511)

_TIMEOUT_SECONDS = 60
_DEF_LINE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(.*\)\s*:\s*$")
_FIRST_ASSERT_FN = re.compile(r"^assert\s+([A-Za-z_]\w*)\s*\(")


class FetchError(ClaspP5Error):
    """Downloading or parsing a benchmark source failed."""


@dataclass(frozen=True)
class FetchResult:
    benchmark: str
    source_url: str
    license: str
    raw_sha256: str
    tasks_written: int
    tasks_skipped: int
    skip_reasons: dict[str, int]
    output_path: Path
    manifest_path: Path


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download(url: str) -> bytes:
    _LOG.info("Fetching %s", url)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(
            f"Could not download {url}: {exc}. Benchmark data fetching requires network access; "
            f"the harness falls back to the bundled sample fixture (5 tasks) if this file is never "
            f"written — see evaluation/base.py's BenchmarkAdapter._resolve_tasks_path."
        ) from exc


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------
def fetch_humaneval(output_dir: Path) -> FetchResult:
    raw = _download(_HUMANEVAL_URL)
    raw_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        text = gzip.decompress(raw).decode("utf-8")
    except OSError as exc:
        raise FetchError(f"HumanEval download at {_HUMANEVAL_URL} is not valid gzip: {exc}") from exc

    tasks: list[dict[str, Any]] = []
    skipped = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            tasks.append(
                {
                    "task_id": record["task_id"],
                    "benchmark": "HumanEval",
                    "prompt": record["prompt"],
                    "entry_point": record["entry_point"],
                    "canonical_solution": record["canonical_solution"],
                    "test_code": record["test"],
                    "metadata": {
                        "origin": "openai/human-eval",
                        "source_url": _HUMANEVAL_URL,
                        "license": _HUMANEVAL_LICENSE,
                    },
                }
            )
        except KeyError as exc:
            _LOG.warning("HumanEval line %d missing field %s; skipped", lineno, exc)
            skipped += 1

    output_path = output_dir / "tasks.jsonl"
    write_jsonl(output_path, tasks)
    manifest_path = _write_manifest(
        output_dir / "tasks_manifest.json",
        benchmark="HumanEval",
        source_url=_HUMANEVAL_URL,
        license_=_HUMANEVAL_LICENSE,
        raw_sha256=raw_sha256,
        written=len(tasks),
        skipped=skipped,
        skip_reasons={"missing_field": skipped} if skipped else {},
        split="full (164 problems, no filtering)",
    )
    return FetchResult(
        benchmark="HumanEval",
        source_url=_HUMANEVAL_URL,
        license=_HUMANEVAL_LICENSE,
        raw_sha256=raw_sha256,
        tasks_written=len(tasks),
        tasks_skipped=skipped,
        skip_reasons={"missing_field": skipped} if skipped else {},
        output_path=output_path,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------
def _mbpp_signature_and_body(code: str, expected_entry_point: str) -> tuple[str, str] | None:
    """Split MBPP's ``code`` (a full function) into (signature_line, body).

    Returns ``None`` when ``code`` does not open with a single top-level
    ``def <expected_entry_point>(...):`` line — the shape every adapter in
    this repository assumes canonical_solution/prompt to have. A handful of
    MBPP references (helper classes, multi-function solutions, decorators)
    do not fit this shape; those tasks are skipped rather than mis-parsed.
    """
    lines = code.splitlines()
    if not lines:
        return None
    match = _DEF_LINE.match(lines[0].strip())
    if not match or match.group(1) != expected_entry_point:
        return None
    body_lines = lines[1:]
    if not body_lines or not all(not ln or ln.startswith((" ", "\t")) for ln in body_lines):
        return None
    return lines[0].strip(), "\n".join(body_lines) + "\n"


def fetch_mbpp(output_dir: Path, *, eval_range: range = _MBPP_EVAL_RANGE) -> FetchResult:
    raw = _download(_MBPP_URL)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    tasks: list[dict[str, Any]] = []
    skip_reasons: dict[str, int] = {"out_of_split": 0, "no_entry_point": 0, "unparseable_solution": 0}

    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = int(record["task_id"])
        if task_id not in eval_range:
            skip_reasons["out_of_split"] += 1
            continue

        test_list: list[str] = record.get("test_list", [])
        entry_match = _FIRST_ASSERT_FN.match(test_list[0].strip()) if test_list else None
        if not entry_match:
            skip_reasons["no_entry_point"] += 1
            continue
        entry_point = entry_match.group(1)

        parsed = _mbpp_signature_and_body(record["code"], entry_point)
        if parsed is None:
            skip_reasons["unparseable_solution"] += 1
            continue
        signature, body = parsed

        test_code_lines = list(test_list)
        setup = record.get("test_setup_code") or ""
        test_code = (setup + "\n" if setup else "") + "\n".join(test_code_lines) + "\n"

        tasks.append(
            {
                "task_id": f"MBPP/{task_id}",
                "benchmark": "MBPP",
                "prompt": record["text"],
                "entry_point": entry_point,
                "canonical_solution": body,
                "test_code": test_code,
                "metadata": {
                    "origin": "google-research/google-research:mbpp",
                    "source_url": _MBPP_URL,
                    "license": _MBPP_LICENSE,
                    "signature": signature,
                    "test_list": test_list,
                },
            }
        )

    output_path = output_dir / "tasks.jsonl"
    write_jsonl(output_path, tasks)
    total_skipped = sum(skip_reasons.values())
    manifest_path = _write_manifest(
        output_dir / "tasks_manifest.json",
        benchmark="MBPP",
        source_url=_MBPP_URL,
        license_=_MBPP_LICENSE,
        raw_sha256=raw_sha256,
        written=len(tasks),
        skipped=total_skipped,
        skip_reasons={k: v for k, v in skip_reasons.items() if v},
        split=f"test split, task_id {eval_range.start}-{eval_range.stop - 1} "
        f"(Austin et al. 2021 few-shot/eval convention)",
    )
    return FetchResult(
        benchmark="MBPP",
        source_url=_MBPP_URL,
        license=_MBPP_LICENSE,
        raw_sha256=raw_sha256,
        tasks_written=len(tasks),
        tasks_skipped=total_skipped,
        skip_reasons=skip_reasons,
        output_path=output_path,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
def _write_manifest(
    path: Path,
    *,
    benchmark: str,
    source_url: str,
    license_: str,
    raw_sha256: str,
    written: int,
    skipped: int,
    skip_reasons: dict[str, int],
    split: str,
) -> Path:
    write_json(
        path,
        {
            "benchmark": benchmark,
            "source_url": source_url,
            "license": license_,
            "split": split,
            "raw_download_sha256": raw_sha256,
            "fetched_at": utc_timestamp(),
            "tasks_written": written,
            "tasks_skipped": skipped,
            "skip_reasons": skip_reasons,
            "provenance": "REAL — fetched from the published benchmark's canonical source.",
        },
    )
    return path


_FETCHERS: dict[str, Callable[[Path], FetchResult]] = {
    "humaneval": fetch_humaneval,
    "mbpp": fetch_mbpp,
}


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "mbpp", "all"],
        default="all",
        help="Which benchmark's real task set to fetch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if tasks.jsonl already exists.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    targets = ["humaneval", "mbpp"] if args.benchmark == "all" else [args.benchmark]

    lines = ["", "CLASP-P5 · Fetch Real Benchmark Data", "=" * 68, ""]
    results: list[FetchResult] = []

    for name in targets:
        output_dir = paths.evaluation / name
        existing = output_dir / "tasks.jsonl"
        if existing.is_file() and not args.force:
            lines.append(f"  {name:<10} SKIPPED — {paths.relative(existing)} already exists (--force to refetch)")
            continue
        result = _FETCHERS[name](output_dir)
        results.append(result)
        lines.append(
            f"  {result.benchmark:<10} {result.tasks_written} task(s) written, "
            f"{result.tasks_skipped} skipped  ->  {paths.relative(result.output_path)}"
        )
        if result.skip_reasons:
            lines.append(f"               skip reasons: {result.skip_reasons}")
        lines.append(f"               licence: {result.license}")
        lines.append(f"               manifest: {paths.relative(result.manifest_path)}")

    lines.append("")
    lines.append("  Next: point configs/evaluation.yaml benchmarks.<name>.tasks_path at the file(s)")
    lines.append("  above (already done in this repository's committed config).")
    lines.append("")
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
