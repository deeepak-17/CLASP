#!/usr/bin/env python3
"""Build federated partitions and their metadata.

Reads the collected corpus, applies the configured partition strategy, writes
one JSONL shard per federated client, the contract manifest, and the
descriptive metadata document.

Also the Week-4 Monday build path: this script is strategy-agnostic (it just
runs whatever ``partition.strategy`` the given config names), so
``--config configs/partition_per_developer.yaml`` builds the per-developer
partition with no code changes — reusing this exact script, per the Week-4
"reuse existing... configuration" requirement.

Usage::

    python scripts/build_partitions.py
    python scripts/build_partitions.py --dry-run
    python scripts/build_partitions.py --config configs/partition.yaml
    python scripts/build_partitions.py --config configs/partition_per_developer.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from partitions.metadata import (
    build_partition_metadata,
    render_metadata_report,
    write_partition_metadata,
)
from partitions.partitioner import Partitioner, load_partition_config
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="Partition config. Defaults to configs/partition.yaml."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute shards and the manifest without writing them."
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Write the contract manifest only, skipping the descriptive metadata document.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_partition_config(args.config)
    partitioner = Partitioner(config)
    result = partitioner.run(dry_run=args.dry_run)
    manifest = result.manifest
    paths = project_paths()

    lines = [
        "",
        "CLASP-P5 · Partition Build",
        "=" * 68,
        f"  strategy: {manifest.strategy.value}",
        f"  seed:     {manifest.seed}",
        f"  clients:  {manifest.num_clients}",
        f"  clusters: {manifest.num_clusters}",
        f"  files:    {manifest.total_files}",
        f"  elapsed:  {result.elapsed_seconds:.3f}s",
        "",
        f"  {'client':<22}{'cluster':<12}{'files':>7}{'code lines':>13}",
    ]
    for shard in manifest.shards:
        lines.append(
            f"  {shard.client_id:<22}{shard.cluster_id:<12}"
            f"{shard.num_files:>7}{shard.num_code_lines:>13}"
        )
    lines.append("")

    if args.dry_run:
        lines += ["  DRY RUN — nothing written.", ""]
        emit(lines)
        return EXIT_OK

    lines.append(f"  manifest: {paths.relative(result.manifest_path)}")

    if not args.skip_metadata:
        # Read shards back from disk so the metadata describes what was
        # actually written, not what was held in memory.
        from partitions.partitioner import load_shard_records

        shard_records = {
            shard.client_id: load_shard_records(manifest, shard.client_id)
            for shard in manifest.shards
        }
        metadata = build_partition_metadata(
            manifest, shard_records, manifest_path=result.manifest_path
        )
        # Namespaced by output dir so this shared script is safe to reuse for
        # configs/partition_per_developer.yaml without clobbering Week-2's
        # datasets/partitions/partition_metadata.json (see
        # PartitionOutputConfig.report_suffix's docstring).
        metadata_path = write_partition_metadata(
            metadata, config.output.partitions_dir / "partition_metadata.json"
        )
        lines.append(f"  metadata: {paths.relative(metadata_path)}")

        if not args.no_report:
            report_name = f"partition_metadata{config.output.report_suffix()}.md"
            path = render_metadata_report(metadata).write(paths.reports / report_name)
            lines.append(report_line(path))

        # Week-4: PerDeveloperStrategy records *how* each shard's developer
        # boundary was derived. Not part of any frozen contract, so it is a
        # sidecar rather than a manifest field; skipped entirely for
        # project_level runs, where no assignment carries a developer_basis.
        basis_entries = {
            a.client_id: {"basis": a.developer_basis, "label": a.developer_label}
            for a in result.assignments
            if a.developer_basis is not None
        }
        if basis_entries:
            from utils.io_utils import write_json

            basis_path = write_json(config.output.partitions_dir / "developer_basis.json", basis_entries)
            lines.append(f"  dev basis: {paths.relative(basis_path)}")

    lines.append("")
    lines.append("  Next: python scripts/validate_partitions.py")
    lines.append("")
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
