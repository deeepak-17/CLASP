#!/usr/bin/env python3
"""Rebuild every P5 result, report, figure and the archive from committed inputs — one command.

Steps, in dependency order (each is its own script and can be run alone):

1. score the base-model HumanEval samples into the guard anchor (and its REAL
   results.json record);
2. personalization results from the edge round manifests;
3. adapter lineage;
4. evaluation noise and the locked guard thresholds;
5. report figures and results tables;
6. slides headline numbers;
7. the checksummed archive manifest.

Nothing here needs a GPU, a network or the other services; inputs are the
files under ``results/edge_rounds/`` and ``results/humaneval_guard/``.

Usage::

    python scripts/rebuild_results.py
    python scripts/rebuild_results.py --label "phase-2 final"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, run_cli, setup_logging
from evaluation.utils.paths import project_paths

SCRIPTS = Path(__file__).resolve().parent


def steps(label: str) -> list[tuple[str, list[str]]]:
    return [
        ("guard anchor", ["score_humaneval_samples.py", "--samples", "results/humaneval_guard/base_samples.jsonl",
                          "--out", "results/humaneval_guard/base_anchor.json", "--no-report", "--record-result"]),
        ("personalization", ["build_personalization_report.py"]),
        ("lineage", ["export_lineage.py"]),
        ("noise", ["build_noise_report.py"]),
        ("figures", ["build_report_figures.py"]),
        ("slides data", ["build_slides_data.py"]),
        ("archive", ["archive_results.py", "--label", label]),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--label", default="rebuild", help="Label stored in the archive manifest.")
    return parser


def main(args: argparse.Namespace) -> int:
    root = project_paths().root
    lines = ["", "CLASP-P5 · rebuild results", "=" * 68]
    # Run each step as its own process, in order; stop at the first failure.
    for name, argv in steps(args.label):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / argv[0]), *argv[1:]],
            cwd=root, capture_output=True, text=True, check=False,
        )
        lines.append(f"  {'ok' if proc.returncode == 0 else 'FAILED':<7} {name}")
        if proc.returncode != 0:
            lines += ["", proc.stdout[-2000:], proc.stderr[-2000:]]
            emit(lines)
            return EXIT_FAILURE
    lines.append("")
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
