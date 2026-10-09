#!/usr/bin/env python3
"""Export adapter lineage (client -> cluster -> composite, across rounds) for the dashboard.

Reads edge round manifests — by default the real rounds under
``results/edge_rounds/`` — and writes ``results/lineage.json``, which the
dashboard's Lineage page renders.

Usage::

    python scripts/export_lineage.py
    python scripts/export_lineage.py path/to/round_manifest.json ...
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, run_cli, setup_logging
from evaluation.eval_harness.lineage import build_lineage
from evaluation.utils.io_utils import read_json, write_json
from evaluation.utils.paths import project_paths


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("manifests", type=Path, nargs="*", help="Edge round_manifest.json file(s).")
    parser.add_argument("--out", type=Path, default=None, help="Defaults to results/lineage.json.")
    return parser


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    sources = args.manifests or sorted((paths.eval_results / "edge_rounds").glob("round*_manifest.json"))
    lineage = build_lineage(read_json(p) for p in sources)
    lineage["sources"] = [paths.relative(p) for p in sources]
    out = args.out or paths.eval_results / "lineage.json"
    write_json(out, lineage)

    kinds = Counter(n["kind"] for n in lineage["nodes"])
    edges = Counter(e["kind"] for e in lineage["edges"])
    emit(
        [
            "",
            "CLASP-P5 · adapter lineage",
            "=" * 68,
            f"  rounds   {', '.join(str(r) for r in lineage['rounds'])}",
            "  nodes    " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())),
            "  edges    " + ", ".join(f"{k} {v}" for k, v in sorted(edges.items())),
            f"  lineage  {paths.relative(out)}",
            "",
        ]
    )
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
