#!/usr/bin/env python3
"""E3.1 support — materialize one client's partition shard as a real file tree.

Turns one client's JSONL shard (``datasets/partitions/<client_id>.jsonl``)
into an actual directory of ``.py`` files at their original relative paths —
what P1's task E3.1 ("Train client LoRA on a real partition") needs as
input — with a deterministic ``held_out_fraction`` of files set aside into a
separate directory so P1 never trains on the files used to score the result.

Read-only against ``datasets/partitions/``: only reads the manifest and shard
already built by ``scripts/build_partitions.py``; never writes there, and
never touches the manifest.

Usage::

    python scripts/materialize_client_repo.py --client-id client-flask
    python scripts/materialize_client_repo.py --client-id client-flask --dry-run
    python scripts/materialize_client_repo.py --client-id client-flask --held-out-fraction 0.2 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from partitions.materialize import (
    load_materialize_config,
    materialize_client_repo,
    render_materialize_report,
)
from partitions.partitioner import load_partition_manifest
from utils.config import resolve_path
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--client-id", required=True, help="Client to materialize, e.g. client-flask."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Materialize config. Defaults to configs/materialize_client.yaml.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Partition manifest. Defaults to the config's manifest_path.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output_root from the config."
    )
    parser.add_argument(
        "--held-out-fraction",
        type=float,
        default=None,
        help="Override held_out_fraction from the config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed from the config.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute the split without writing any files."
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_materialize_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_root=resolve_path(args.output_dir))
    if args.held_out_fraction is not None:
        config = replace(config, held_out_fraction=args.held_out_fraction)
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    manifest_path = args.manifest or config.manifest_path
    manifest = load_partition_manifest(manifest_path)
    paths = project_paths()

    result = materialize_client_repo(manifest, args.client_id, config, dry_run=args.dry_run)

    lines = [
        "",
        "CLASP-P5 · E3.1 support — Materialize Client Repo",
        "=" * 68,
        f"  client:             {result.client_id}",
        f"  cluster:            {result.cluster_id}",
        f"  seed:               {result.seed}",
        f"  held_out_fraction:  {result.held_out_fraction:.2%}",
        f"  kept files:         {result.num_kept}",
        f"  held-out files:     {result.num_held_out}",
        "",
    ]

    if args.dry_run:
        lines += ["  DRY RUN — nothing written.", ""]
        emit(lines)
        return EXIT_OK

    lines.append(f"  repo dir:      {paths.relative(result.repo_dir)}")
    lines.append(f"  held-out dir:  {paths.relative(result.held_out_dir)}")
    lines.append(f"  manifest:      {paths.relative(result.manifest_path)}")

    if not args.no_report:
        report_name = f"materialize_{result.client_id}.md"
        path = render_materialize_report(result).write(paths.reports / report_name)
        lines.append(report_line(path))

    lines.append("")
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
