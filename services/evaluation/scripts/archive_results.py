#!/usr/bin/env python3
"""Write (or check) the checksummed archive manifest of P5's reported results.

Usage::

    python scripts/archive_results.py --label "phase-2 final"   # write results/ARCHIVE_MANIFEST.json
    python scripts/archive_results.py --check                    # exit 1 if anything changed
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, run_cli, setup_logging
from evaluation.eval_harness.archive import MANIFEST_NAME, build_manifest, verify_manifest
from evaluation.utils.io_utils import read_json, write_json
from evaluation.utils.paths import project_paths


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--label", default="", help="Free-text label stored in the manifest.")
    parser.add_argument("--check", action="store_true", help="Verify instead of writing.")
    return parser


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    path = paths.eval_results / MANIFEST_NAME
    if args.check:
        diff = verify_manifest(read_json(path), paths.root)
        lines = ["", "CLASP-P5 · archive check", "=" * 68]
        for kind, items in diff.items():
            lines.append(f"  {kind:<8} {len(items)}")
            lines += [f"    {item}" for item in items]
        clean = not any(diff.values())
        lines += [f"  result   {'archive intact' if clean else 'archive DIFFERS from manifest'}", ""]
        emit(lines)
        return EXIT_OK if clean else EXIT_FAILURE
    manifest = build_manifest(paths.root, label=args.label)
    write_json(path, manifest)
    emit(["", "CLASP-P5 · archive", "=" * 68, f"  files    {manifest['n_files']}",
          f"  bytes    {manifest['total_bytes']:,}", f"  manifest {paths.relative(path)}", ""])
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
