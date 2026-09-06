#!/usr/bin/env python3
"""Week 1 · Friday — dry-run the evaluation harness on a tiny sample.

Exercises the whole harness path end to end against the mock Edge client:
config resolution, task loading, prompt construction, generation, truncation,
program assembly and artefact writing. Pass@k scoring is Week 3 and is not
attempted here.

Usage::

    python scripts/dry_run_harness.py
    python scripts/dry_run_harness.py --limit 2
    python scripts/dry_run_harness.py --benchmark HumanEval
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.harness import (
    EvaluationHarness,
    default_dry_run_report_path,
    load_evaluation_config,
    render_dry_run_report,
)
from evaluation.models import parse_benchmark_name
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="Evaluation config. Defaults to configs/evaluation.yaml."
    )
    parser.add_argument("--limit", type=int, default=None, help="Tasks per benchmark. Overrides run.limit.")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Restrict the run to one benchmark (HumanEval or MBPP).",
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="Skip writing the run JSON to evaluation/results/.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Report path. Defaults to reports/dry_run_report.md."
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_evaluation_config(args.config)

    if args.benchmark:
        selected = parse_benchmark_name(args.benchmark)
        benchmarks = {
            key: replace(value, enabled=(key == selected.value.lower()))
            for key, value in config.benchmarks.items()
        }
        config = replace(config, benchmarks=benchmarks)
        _LOG.info("Restricted run to %s", selected.value)

    harness = EvaluationHarness(config)
    run = harness.run(limit=args.limit, write_artifact=not args.no_artifact)
    paths = project_paths()

    lines = [
        "",
        "CLASP-P5 · Week 1 Friday — Harness Dry Run",
        "=" * 60,
        f"  run id:   {run.run_id}",
        f"  mode:     {run.mode}",
        f"  backend:  {config.backend.kind}  ({run.model_id})",
        f"  elapsed:  {run.elapsed_seconds:.3f}s",
        "",
    ]
    for summary in run.summaries:
        lines.append(
            f"    {summary.benchmark.value:<12} "
            f"{summary.tasks_succeeded}/{summary.tasks_attempted} task(s) generated, "
            f"{summary.completions_generated} completion(s), "
            f"{summary.tasks_failed} failed"
        )
    lines.append("")
    lines.append("  NOTE: Pass@k scoring is a Week-3 deliverable — this run is unscored.")
    lines.append("")

    if run.artifact_path:
        lines.append(f"  artefact: {paths.relative(run.artifact_path)}")

    if not args.no_report:
        output = args.output or default_dry_run_report_path()
        path = render_dry_run_report(run, config).write(output)
        lines.append(report_line(path))
    lines.append("")

    emit(lines)
    return EXIT_OK if run.ok else EXIT_FAILURE


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
