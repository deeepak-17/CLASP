#!/usr/bin/env python3
"""Week 1 · Monday — survey candidate datasets and record the selection.

Scores every candidate in ``configs/dataset_survey.yaml`` against the weighted
criteria and writes ``reports/dataset_survey.md``.

Usage::

    python scripts/survey_datasets.py
    python scripts/survey_datasets.py --config configs/dataset_survey.yaml
    python scripts/survey_datasets.py --strict     # fail if selection != top scorer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from corpus.survey import load_survey_config, render_survey_report, score_survey
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Survey config. Defaults to configs/dataset_survey.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. Defaults to reports/dataset_survey.md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the recorded selection is not the top-scoring candidate.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_survey_config(args.config)
    outcome = score_survey(config)

    lines = [
        "",
        "CLASP-P5 · Week 1 Monday — Dataset Survey",
        "=" * 60,
        f"  criteria:  {len(config.criteria)} weighted",
        f"  candidates: {len(config.candidates)} scored",
        "",
        "  Ranking:",
    ]
    for item in outcome.ranking:
        marker = "<- SELECTED" if item.is_selected else ""
        lines.append(
            f"    {item.rank}. {item.candidate.name:<42} "
            f"{item.weighted_score:>6.3f}  ({item.normalised_score:>5.1f}%) {marker}"
        )
    lines.append("")

    for warning in outcome.warnings:
        lines.append(f"  WARNING: {warning}")
    if outcome.warnings:
        lines.append("")

    if not args.no_report:
        output = args.output or (project_paths().reports / "dataset_survey.md")
        path = render_survey_report(outcome).write(output)
        lines.append(report_line(path))
        lines.append("")

    emit(lines)

    if args.strict and not outcome.selection_is_top_scorer:
        _LOG.error("--strict: selected candidate is not the top scorer")
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
