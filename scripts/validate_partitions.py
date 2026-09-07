#!/usr/bin/env python3
"""Validate partitions are non-overlapping and balanced.

Reads the manifest and every shard back from disk, then checks disjointness,
content disjointness, coverage, project purity, minimum shard size and the
balance metrics.

Exit code is non-zero when any hard constraint fails, so this is CI-ready.

Also the Week-4 Tuesday leakage-validation path: strategy-agnostic like
``build_partitions.py``, so ``--config configs/partition_per_developer.yaml``
runs the identical checks against the per-developer manifest — disjointness
and content-disjointness are manifest-wide, so this is exactly how
"per-developer partitions don't leak across projects" gets verified.

Usage::

    python scripts/validate_partitions.py
    python scripts/validate_partitions.py --fail-on-warning
    python scripts/validate_partitions.py --config configs/partition_per_developer.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from corpus.collector import load_corpus_records
from partitions.partitioner import load_partition_config, load_partition_manifest, load_shard_records
from partitions.validation import PartitionValidator, render_validation_report
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="Partition config. Defaults to configs/partition.yaml."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path. Defaults to the one named in the config.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Promote balance warnings to errors. Overrides validation.fail_on_warning.",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip the corpus coverage check (avoids re-reading the full corpus).",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Report path. Defaults to reports/partition_validation_report.md."
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_partition_config(args.config)
    validation_config = config.validation
    if args.fail_on_warning:
        validation_config = replace(validation_config, fail_on_warning=True)

    manifest_path = args.manifest or config.output.manifest_path
    manifest = load_partition_manifest(manifest_path)
    paths = project_paths()
    _LOG.info("Validating %s", paths.relative(Path(manifest_path)))

    shard_records = {
        shard.client_id: load_shard_records(manifest, shard.client_id) for shard in manifest.shards
    }

    corpus_records = None
    if not args.skip_coverage:
        corpus_records = load_corpus_records(config.input.corpus_path)

    result = PartitionValidator(validation_config).validate(
        manifest, shard_records, corpus_records=corpus_records
    )

    lines = [
        "",
        "CLASP-P5 · Partition Validation",
        "=" * 68,
        f"  manifest: {paths.relative(Path(manifest_path))}",
        f"  verdict:  {'PASS' if result.ok else 'FAIL'}",
        "",
    ]
    for check in result.checks_passed:
        lines.append(f"    ok    {check}")
    for warning in result.warnings:
        lines.append(f"    warn  {warning}")
    for error in result.errors:
        lines.append(f"    FAIL  {error}")
    lines.append("")

    if result.balance:
        balance = result.balance
        lines += [
            "  Balance:",
            f"    shards ............... {balance.num_shards}",
            f"    files min/mean/max ... {balance.min_files} / {balance.mean_files:.1f} / {balance.max_files}",
            f"    imbalance ratio ...... {balance.imbalance_ratio:.3f}  (max {validation_config.max_imbalance_ratio:.2f})",
            f"    gini ................. {balance.gini:.4f}  (max {validation_config.max_gini:.2f})",
            f"    coeff. of variation .. {balance.coefficient_of_variation:.4f}  (max {validation_config.max_coefficient_of_variation:.2f})",
            "",
        ]

    if not args.no_report:
        report_name = f"partition_validation_report{config.output.report_suffix()}.md"
        output = args.output or (paths.reports / report_name)
        path = render_validation_report(manifest, result, validation_config).write(output)
        lines.append(report_line(path))
        # Machine-readable sibling (Week-4 point 9: "produce machine-readable
        # output suitable for later reports") for BOTH project-level and
        # per-developer runs — this script is shared between them.
        from utils.io_utils import write_json

        json_path = write_json(output.with_suffix(".json"), result.to_dict())
        lines.append(f"  report (json): {paths.relative(json_path)}")
        lines.append("")

    emit(lines)
    return EXIT_OK if result.ok else EXIT_FAILURE


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
