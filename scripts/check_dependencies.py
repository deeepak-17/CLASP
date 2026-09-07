#!/usr/bin/env python3
"""Verify HumanEval/MBPP harness dependencies.

Reports which declared dependencies are present, which are missing and
blocking now, and which are deferred to a later week.

Usage::

    python scripts/check_dependencies.py
    python scripts/check_dependencies.py --strict   # fail on deferred gaps too
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.dependencies import check_dependencies, render_dependency_report
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. Defaults to reports/dependency_report.md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail when optional dependencies are absent.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    result = check_dependencies()

    lines = [
        "",
        "CLASP-P5 · Harness Dependencies",
        "=" * 60,
        f"  python:   {result.python_version}  ({'ok' if result.python_ok else 'TOO OLD'})",
        f"  platform: {result.platform}",
        "",
    ]
    for status in result.statuses:
        flag = "ok " if status.ok else "!! "
        needed = "now" if status.spec.required_now else "later"
        lines.append(
            f"  {flag}{status.spec.package:<14} {(status.version or '-'):<12} "
            f"[{needed:<3}] {status.state}"
        )
    lines.append("")

    if result.blocking:
        lines.append(f"  {len(result.blocking)} blocking issue(s):")
        for status in result.blocking:
            lines.append(f"    - {status.spec.package}: {status.spec.hint()}")
        lines.append("")
    else:
        lines.append("  All required dependencies are satisfied.")
        lines.append("")

    if result.deferred:
        lines.append(f"  {len(result.deferred)} optional dependency/ies not required yet:")
        for status in result.deferred:
            lines.append(
                f"    - {status.spec.package} (optional — not required yet)"
            )
        lines.append("")

    if not args.no_report:
        output = args.output or (project_paths().reports / "dependency_report.md")
        path = render_dependency_report(result).write(output)
        lines.append(report_line(path))
        lines.append("")

    emit(lines)

    if not result.ok:
        return EXIT_FAILURE
    if args.strict and result.deferred:
        _LOG.error("--strict: %d deferred dependency/ies are absent", len(result.deferred))
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
