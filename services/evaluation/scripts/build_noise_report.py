#!/usr/bin/env python3
"""Measure evaluation noise and derive the guard thresholds from it.

Reads ``configs/guard_thresholds.yaml`` and the scored HumanEval baseline
anchor it names, then writes:

* ``results/noise_report.json`` — the noise floor of the HumanEval guard
  (bootstrap interval, paired minimum detectable drop at the scored size and
  at the full benchmark, items needed to resolve the D5 tolerance) and of
  in-project exact match; with ``--candidate-anchor`` also the measured
  discordance and the noise-aware verdict on the candidate's drop;
* ``reports/eval_noise_report.md`` — the same, written up.

Usage::

    python scripts/build_noise_report.py
    python scripts/build_noise_report.py --candidate-anchor eval_out/candidate/anchor.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.completion import noise_band as d5_noise_band
from evaluation.eval_harness.noise import (
    binomial_standard_error,
    classify_guard_drop,
    guard_noise_report,
    paired_min_detectable_drop,
)
from evaluation.utils.config import load_yaml, resolve_path
from evaluation.utils.errors import EvaluationError
from evaluation.utils.io_utils import read_json, write_json
from evaluation.utils.paths import project_paths
from evaluation.utils.reporting import MarkdownReport
from evaluation.utils.timing import utc_timestamp

#: In-project exact match measured in the integration round (60 examples, greedy),
#: as recorded in results/in_project_metric.json.
IN_PROJECT_SOURCE = "results/in_project_metric.json"


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="Defaults to configs/guard_thresholds.yaml.")
    parser.add_argument("--candidate-anchor", type=Path, default=None, help="Scored candidate anchor.json.")
    parser.add_argument("--out", type=Path, default=None, help="Defaults to results/noise_report.json.")
    return parser


def _per_task(anchor: dict[str, Any]) -> dict[str, bool]:
    rows = anchor.get("per_task")
    if not rows:
        raise EvaluationError("anchor has no per_task outcomes; re-score it with score_humaneval_samples.py")
    return {row["task_id"]: bool(row["passed"]) for row in rows}


def _candidate_block(base: dict[str, bool], cand: dict[str, bool], guard_cfg: dict[str, Any]) -> dict[str, Any]:
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise EvaluationError("baseline and candidate anchors share no tasks")
    n = len(shared)
    discordant = sum(1 for t in shared if base[t] != cand[t])
    discordance = discordant / n
    drop = (sum(base[t] for t in shared) - sum(cand[t] for t in shared)) / n
    floor = paired_min_detectable_drop(discordance, n)
    tolerance = float(guard_cfg["tolerance_pass_at_1"])
    return {
        "n_shared_tasks": n,
        "discordant_tasks": discordant,
        "measured_discordance": round(discordance, 4),
        "pass_at_1_drop": round(drop, 4),
        "noise_floor": round(floor, 4),
        "verdict": classify_guard_drop(drop, tolerance=tolerance, noise_floor=floor),
    }


def _in_project_block(paths, cfg: dict[str, Any]) -> dict[str, Any]:
    data = read_json(paths.root / IN_PROJECT_SOURCE)
    n = int(cfg["n_examples"])
    d = float(cfg["exact_match_assumed_discordance"])
    rows = []
    for row in data["in_project_completion"]["rows"]:
        em = float(row["exact_match"])
        rows.append(
            {
                "client": row["client"],
                "version": row["version"],
                "edit_similarity": row["edit_similarity"],
                "exact_match": em,
                "exact_match_standard_error": round(binomial_standard_error(em, n), 4),
            }
        )
    bands = {}
    for client, values in cfg["d5_baseline_repeats"].items():
        band, note = d5_noise_band([{"edit_similarity": float(v)} for v in values])
        bands[client] = {"repeats": [float(v) for v in values], "band": round(band, 6), "note": note}
    return {
        "source": IN_PROJECT_SOURCE,
        "n_examples": n,
        "d5_noise_band": {
            "definition": "spread of 3 repeated baseline evaluations (D5)",
            "value": max(b["band"] for b in bands.values()),
            "per_client": bands,
            "degenerate": all(b["band"] == 0.0 for b in bands.values()),
        },
        "exact_match_paired_noise_floor": round(paired_min_detectable_drop(d, n), 4),
        "exact_match_assumed_discordance": d,
        "edit_similarity_band": None,
        "edit_similarity_band_status": (
            "supplementary bootstrap band not measurable from the recorded round: it needs "
            "per-example rows (evaluation.completion.per_example_rows) for both versions; "
            "evaluation.eval_harness.noise.paired_bootstrap_diff_ci computes it from them"
        ),
        "rows": rows,
    }


def _report(result: dict[str, Any], path: Path) -> Path:
    g = result["humaneval_guard"]
    report = MarkdownReport(
        "Evaluation noise and guard thresholds",
        subtitle="How large a change must be before D5 can tell it from noise",
    )
    report.heading("HumanEval guard")
    report.key_values(
        {
            "baseline anchor": result["sources"]["baseline_anchor"],
            "tasks scored": g["n_tasks"],
            "baseline pass@1": g["pass_at_1"],
            "standard error": g["standard_error"],
            "95% bootstrap interval": f"{g['bootstrap_ci_95'][0]} – {g['bootstrap_ci_95'][1]}",
            "D5 tolerance (max allowed drop)": g["tolerance"],
            "unpaired minimum detectable drop at this size": g["unpaired_min_detectable_drop_at_n"],
        }
    )
    report.paragraph(
        "Candidate and baseline are scored on the same tasks, so the relevant noise is paired: "
        "only tasks whose outcome flips between the two models contribute. The fraction that "
        "flips (discordance) is unknown until a candidate is scored, so the floor is shown for "
        "a range of plausible values."
    )
    report.table(
        ["discordance", f"min detectable drop @ {g['n_tasks']} tasks", f"@ {g['full_benchmark_size']} tasks",
         f"tasks needed to resolve {g['tolerance']}"],
        [
            [r["discordance"], r["min_detectable_drop_at_n"], r["min_detectable_drop_full_benchmark"],
             r["items_needed_for_tolerance"]]
            for r in g["paired"]
        ],
    )
    report.heading("Locked thresholds", level=3)
    report.bullets(result["locked_thresholds"]["statements"])
    if result.get("candidate"):
        c = result["candidate"]
        report.heading("Candidate", level=3)
        report.key_values(c)
    ip = result["in_project"]
    report.heading("In-project completion")
    band = ip["d5_noise_band"]
    report.paragraph(
        f"**D5 noise band** ({band['definition']}): **{band['value']}**. "
        + (
            "Greedy decoding is deterministic, so three repeats of the same baseline are identical and "
            "the band is zero: under it, any positive gain counts as an improvement. It measures decode "
            "noise only, not the variation between independently trained adapters."
            if band["degenerate"]
            else "Measured from the repeats listed in configs/guard_thresholds.yaml."
        )
    )
    report.table(
        ["client", "version", "edit similarity", "exact match", "exact-match SE"],
        [[r["client"], r["version"], r["edit_similarity"], r["exact_match"], r["exact_match_standard_error"]]
         for r in ip["rows"]],
    )
    report.paragraph(
        f"At {ip['n_examples']} paired examples and an assumed {ip['exact_match_assumed_discordance']} "
        f"discordance, an exact-match change smaller than {ip['exact_match_paired_noise_floor']} is "
        f"inside noise. Edit-similarity band: {ip['edit_similarity_band_status']}."
    )
    return report.write(path)


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    cfg = load_yaml(args.config or paths.configs / "guard_thresholds.yaml")
    guard_cfg = cfg["humaneval_guard"]
    boot = cfg["bootstrap"]
    anchor_path = resolve_path(guard_cfg["baseline_anchor"])
    base = _per_task(read_json(anchor_path))

    guard = guard_noise_report(
        list(base.values()),
        tolerance=float(guard_cfg["tolerance_pass_at_1"]),
        full_benchmark_size=int(guard_cfg["full_benchmark_size"]),
        seed=int(boot["seed"]),
    )
    assumed = float(guard_cfg["assumed_discordance"])
    floor_now = paired_min_detectable_drop(assumed, guard["n_tasks"])
    floor_full = paired_min_detectable_drop(assumed, guard["full_benchmark_size"])
    statements = [
        "D5 in-project noise band = spread of 3 repeated baseline evaluations (computed with "
        "evaluation.completion.noise_band); see the in-project section for its value.",
        f"D5 tolerance stays {guard['tolerance']} (registry-owned); P5 does not loosen it.",
        f"Guard noise floor at the scored {guard['n_tasks']} tasks (assumed discordance {assumed}): "
        f"{floor_now:.4f}. A drop above the tolerance but below this is reported `within_noise`, not a pass.",
        f"On the full {guard['full_benchmark_size']}-task benchmark the floor falls to {floor_full:.4f} — "
        "still above the tolerance, so a 2-point regression is not resolvable from one greedy sample per "
        "task on HumanEval alone.",
        f"Resolving the tolerance needs about {guard['paired'][1]['items_needed_for_tolerance']} paired tasks "
        f"at discordance {guard['paired'][1]['discordance']} — e.g. HumanEval + MBPP together, or several "
        "samples per task.",
    ]
    result: dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "sources": {"baseline_anchor": paths.relative(anchor_path), "config": "configs/guard_thresholds.yaml"},
        "humaneval_guard": guard,
        "locked_thresholds": {
            "tolerance_pass_at_1": guard["tolerance"],
            "assumed_discordance": assumed,
            "noise_floor_at_scored_n": round(floor_now, 4),
            "noise_floor_full_benchmark": round(floor_full, 4),
            "statements": statements,
        },
        "in_project": _in_project_block(paths, cfg["in_project"]),
    }
    if args.candidate_anchor:
        result["candidate"] = _candidate_block(base, _per_task(read_json(args.candidate_anchor)), guard_cfg)
        result["sources"]["candidate_anchor"] = paths.relative(args.candidate_anchor)

    out = args.out or paths.eval_results / "noise_report.json"
    write_json(out, result)
    lines = [
        "",
        "CLASP-P5 · evaluation noise",
        "=" * 68,
        f"  guard     pass@1 {guard['pass_at_1']} over {guard['n_tasks']} tasks, "
        f"95% CI {guard['bootstrap_ci_95'][0]}–{guard['bootstrap_ci_95'][1]}",
        f"  floor     {floor_now:.4f} at {guard['n_tasks']} tasks, {floor_full:.4f} at "
        f"{guard['full_benchmark_size']} (tolerance {guard['tolerance']})",
    ]
    if result.get("candidate"):
        lines.append(f"  candidate {result['candidate']['verdict']}  drop {result['candidate']['pass_at_1_drop']}")
    lines.append(f"  result    {paths.relative(out)}")
    if not args.no_report:
        lines.append(report_line(_report(result, paths.reports / "eval_noise_report.md")))
    lines.append("")
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
