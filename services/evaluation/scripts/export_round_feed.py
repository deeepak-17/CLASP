#!/usr/bin/env python3
"""Export federated-round manifests into the dashboard's rounds.json.

Reads one or more ``round{N}_manifest.json`` files written by
``scripts/demo_round.py`` and writes ``eval_harness/results/rounds.json``,
which ``dashboard`` copies in (``npm run sync-data``) for its Rounds page.

Usage::

    python scripts/export_round_feed.py experiments/w12-integration/results/round1_manifest.json
    python scripts/export_round_feed.py experiments/w12-integration/results/round*_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, run_cli, setup_logging
from evaluation.eval_harness.round_feed import build_feed
from evaluation.utils.io_utils import read_json
from evaluation.utils.paths import project_paths


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("manifests", type=Path, nargs="+", help="round{N}_manifest.json file(s).")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output path. Defaults to eval_harness/results/rounds.json."
    )
    return parser


def main(args: argparse.Namespace) -> int:
    feed = build_feed(read_json(path) for path in args.manifests)
    out = args.out or project_paths().eval_results / "rounds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(feed, indent=2) + "\n", encoding="utf-8")

    lines = ["", "CLASP-P5 · round feed", "=" * 68]
    for entry in feed["rounds"]:
        guard = "guard scored" if entry["guard"]["available"] else "guard UNAVAILABLE"
        lines.append(f"  round {entry['round']}  ({guard})")
        for cluster in entry["clusters"]:
            delta = cluster["edit_similarity_delta"]
            lines.append(
                f"    {cluster['registry_name']:<22} v{cluster['baseline']['version']} -> "
                f"v{cluster['candidate']['version']}  delta "
                f"{'n/a' if delta is None else f'{delta:+.4f}'}  "
                f"{cluster['decision']['action'].upper()}"
                f"{'' if cluster['authoritative'] else ' (provisional)'}"
            )
    lines.append(f"  alerts         {len(feed['alerts'])}")
    for alert in feed["alerts"]:
        lines.append(f"    r{alert['round']} {alert['cluster_id']:<12} {alert['kind']}: {alert['message']}")
    lines += [f"  feed           {out}", ""]
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
