#!/usr/bin/env python3
"""A REAL sanity check of the scoring pipeline.

"Run Pass@k on the Week-1 baseline model; sanity-check the numbers."

``scripts/run_baseline_eval.py`` runs the harness against
``MockEdgeInferenceClient``, which never produces a passing completion by
design — its Pass@k is always 0.0, which checks the plumbing but not the
scorer's correctness (a scorer with a sign error would also print 0.0 there).

This script is the real check: it loads the actual published HumanEval/MBPP
problem sets (fetched by ``scripts/fetch_benchmark_data.py``; falls back to
the bundled fixture with a warning if that has not been run), feeds each
task's own ``canonical_solution`` back through the exact same
assemble -> execute -> score path the harness uses, and asserts Pass@1 == 1.0
— every task's own reference answer should trivially pass its own test. No
model, mock or otherwise, is involved: this isolates the scorer and executor
from any question about generation quality.

A task whose canonical solution does *not* pass is a real bug (in the
benchmark parsing, the adapter's program assembly, or the executor) and is
reported individually rather than folded into a pass rate.

Usage::

    python scripts/sanity_check_scoring.py
    python scripts/sanity_check_scoring.py --benchmark HumanEval --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.execution import execute_program
from evaluation.harness import load_evaluation_config
from evaluation.models import EvalTask, TaskOutcome, parse_benchmark_name
from evaluation.registry import build_adapter
from evaluation.scoring import aggregate_pass_at_k
from interfaces.contracts import BenchmarkName
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.timing import Stopwatch

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="Evaluation config. Defaults to configs/evaluation.yaml.")
    parser.add_argument(
        "--benchmark", default=None, help="Restrict to one benchmark (HumanEval or MBPP). Default: both."
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap tasks per benchmark (default: all).")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Per-program execution timeout, seconds."
    )
    return parser


def _check_benchmark(name: BenchmarkName, config, *, limit: int | None, timeout: float):
    adapter = build_adapter(name, config)
    tasks: list[EvalTask] = adapter.load_tasks(limit=limit)

    outcomes: list[TaskOutcome] = []
    failures: list[tuple[str, str]] = []

    with Stopwatch(name.value) as watch:
        for task in tasks:
            if task.canonical_solution is None:
                failures.append((task.task_id, "no canonical_solution recorded for this task"))
                outcomes.append(TaskOutcome(task_id=task.task_id, benchmark=name, completions=[], passed=[False]))
                continue
            program = adapter.assemble_program(task, task.canonical_solution)
            outcome = execute_program(program, timeout_seconds=timeout)
            if not outcome.passed:
                failures.append((task.task_id, outcome.stderr_tail[-300:]))
            outcomes.append(
                TaskOutcome(
                    task_id=task.task_id,
                    benchmark=name,
                    completions=[task.canonical_solution],
                    passed=[outcome.passed],
                )
            )

    aggregate = aggregate_pass_at_k(outcomes, [1])
    return {
        "benchmark": name,
        "num_tasks": len(tasks),
        "pass_at_1": aggregate.pass_at_k.get(1, 0.0),
        "failures": failures,
        "elapsed_seconds": watch.elapsed_seconds,
    }


def main(args: argparse.Namespace) -> int:
    config = load_evaluation_config(args.config)
    paths = project_paths()

    benchmarks = [parse_benchmark_name(args.benchmark)] if args.benchmark else list(BenchmarkName)
    results = [_check_benchmark(b, config, limit=args.limit, timeout=args.timeout) for b in benchmarks]

    lines = [
        "",
        "CLASP-P5 · Real Scoring Sanity Check",
        "=" * 68,
        "  (canonical reference solutions, real benchmark data, real execution — no model)",
        "",
    ]
    all_ok = True
    for result in results:
        ok = result["pass_at_1"] == 1.0 and not result["failures"]
        all_ok = all_ok and ok
        lines.append(
            f"    {result['benchmark'].value:<10} {'PASS' if ok else 'FAIL'}  "
            f"pass@1={result['pass_at_1']:.4f}  tasks={result['num_tasks']}  "
            f"{result['elapsed_seconds']:.2f}s"
        )
        for task_id, detail in result["failures"][:5]:
            lines.append(f"        FAIL  {task_id}: {detail.splitlines()[-1] if detail else detail}")
        if len(result["failures"]) > 5:
            lines.append(f"        ... and {len(result['failures']) - 5} more")
    lines.append("")
    lines.append(
        "  Verdict: "
        + (
            "scorer/executor/adapters are self-consistent (every reference solution passes its own test)."
            if all_ok
            else "one or more canonical solutions did not pass — investigate before trusting Pass@k numbers."
        )
    )
    lines.append("")

    if not args.no_report:
        report = MarkdownReport(
            title="CLASP-P5 · Scoring Sanity Check",
            subtitle="real HumanEval/MBPP canonical solutions through the real execution path",
        )
        report.heading("1. Verdict")
        report.status_line(all_ok, "every canonical solution passed its own test" if all_ok else "failures found")
        report.heading("2. Results")
        report.table(
            ["Benchmark", "Tasks", "Pass@1", "Failures", "Elapsed (s)"],
            [
                [r["benchmark"].value, r["num_tasks"], f"{r['pass_at_1']:.4f}", len(r["failures"]), f"{r['elapsed_seconds']:.2f}"]
                for r in results
            ],
        )
        if any(r["failures"] for r in results):
            report.heading("3. Failures")
            for r in results:
                if r["failures"]:
                    report.bullets([f"`{tid}`: {detail.splitlines()[-1] if detail else detail}" for tid, detail in r["failures"]])
        report.rule()
        report.paragraph("Generated by `scripts/sanity_check_scoring.py`. This is REAL data and REAL execution — no mock model involved.")
        path = report.write(paths.reports / "scoring_sanity_check_report.md")
        lines.append(report_line(path))
        lines.append("")

    emit(lines)
    return EXIT_OK if all_ok else EXIT_FAILURE


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
