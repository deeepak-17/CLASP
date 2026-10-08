#!/usr/bin/env python3
"""Build the personalization results (JSON + report) from edge round manifests.

Reads the edge lane's ``round_manifest.json`` files — by default the two real
rounds vendored under ``results/edge_rounds/`` — and writes:

* ``results/personalization.json`` — per-client base / client-only / composite
  held-out perplexity per round and the round-over-round comparison, which
  the dashboard's Personalization page renders;
* ``reports/personalization_report.md`` — the same numbers as a write-up.

Usage::

    python scripts/build_personalization_report.py
    python scripts/build_personalization_report.py path/to/round_manifest.json ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.eval_harness.personalization import build_feed
from evaluation.utils.io_utils import read_json, write_json
from evaluation.utils.paths import project_paths
from evaluation.utils.reporting import MarkdownReport

#: The vendored real rounds, in order, with the label each is reported under.
DEFAULT_ROUNDS = (
    ("results/edge_rounds/round1_manifest.json", "round 1 — clients trained on the bare base"),
    ("results/edge_rounds/round2_d3_manifest.json", "round 2 — D3 order (client on frozen base + 0.5·cluster)"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("manifests", type=Path, nargs="*", help="Edge round_manifest.json file(s).")
    parser.add_argument("--out", type=Path, default=None, help="Defaults to results/personalization.json.")
    return parser


def _report(feed: dict, path: Path) -> Path:
    report = MarkdownReport(
        "Personalization results",
        subtitle="Held-out perplexity on each client's own never-trained-on files (lower is better)",
    )
    report.paragraph(
        "Each row compares three models on one client's held-out files: the frozen base, "
        "the client-only adapter, and the three-layer composite base + α·cluster + β·client "
        "(D6). The personalization delta is composite − base; the cluster contribution is "
        "composite − client-only at the reference α (negative means the cluster layer helps). "
        "Every number is read from the edge round manifest named under Sources."
    )
    for entry in feed["rounds"]:
        s = entry["summary"]
        report.heading(entry["label"])
        report.key_values(
            {
                "model": entry["model_id"],
                "seed": entry["seed"],
                "manifest written": entry["utc"],
                "α grid": ", ".join(str(a) for a in entry["alpha_grid"]),
                "clients improved over base": f"{s['n_improved']} / {s['n_clients']}",
                "mean personalization delta (ppl)": s["mean_delta_ppl"],
                "cluster layer helps": f"{s['n_cluster_helps']} / {s['n_clients']} clients",
                "sweep chose α = 0 (cluster off)": f"{s['n_alpha_zero']} / {s['n_clients']} clients",
            }
        )
        report.table(
            ["client", "held-out tokens", "base", "client only", "composite", "Δ vs base", "cluster @ α_ref", "best α"],
            [
                [
                    c["client_id"],
                    c["n_tokens"],
                    c["base_ppl"],
                    c["client_only_ppl"],
                    c["composite_ppl"],
                    f"{c['personalization_delta_ppl']:+.3f}",
                    f"{c['cluster_contribution_at_alpha_ref']:+.3f}",
                    c["best_alpha"],
                ]
                for c in entry["clients"]
            ],
        )
    for comp in feed["comparisons"]:
        report.heading(f"Round {comp['from_round']} → round {comp['to_round']}")
        report.paragraph(
            f"Composite perplexity improved on {comp['n_composite_better']} of {comp['n_clients']} "
            "clients. The cluster-contribution columns show whether the cluster layer moved from "
            "harmful (positive) to helpful (negative) once clients were trained in the D3 order."
        )
        report.table(
            ["client", "composite Δ (r→r)", "cluster contribution before", "after", "best α before", "after"],
            [
                [
                    r["client_id"],
                    f"{r['composite_ppl_change']:+.3f}",
                    f"{r['cluster_contribution_before']:+.3f}",
                    f"{r['cluster_contribution_after']:+.3f}",
                    r["best_alpha_before"],
                    r["best_alpha_after"],
                ]
                for r in comp["clients"]
            ],
        )
    report.heading("Caveats")
    report.bullets(
        [
            "1.3B `dev` profile on a 4 GB RTX 2050 — not comparable to a 6.7B figure.",
            "client-requests' held-out split is one 513-token block; its row is the noisiest.",
            "Each cluster adapter aggregates all members, including the client being evaluated "
            "(inherent to federated averaging); held-out files were never trained on.",
            "Perplexity is the edge's held-out training eval. D5's primary signal is in-project "
            "edit similarity — see the In-Project page and the noise report.",
        ]
    )
    report.heading("Sources")
    report.bullets([f"`{src}`" for src in feed["sources"]])
    return report.write(path)


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    if args.manifests:
        sources = [Path(p) for p in args.manifests]
        labels = [None] * len(sources)
    else:
        sources = [paths.root / rel for rel, _ in DEFAULT_ROUNDS]
        labels = [label for _, label in DEFAULT_ROUNDS]
    feed = build_feed(
        (read_json(p) for p in sources),
        labels=labels,
        sources=[paths.relative(p) for p in sources],
    )
    out = args.out or paths.eval_results / "personalization.json"
    write_json(out, feed)

    lines = ["", "CLASP-P5 · personalization", "=" * 68]
    for entry in feed["rounds"]:
        s = entry["summary"]
        lines.append(
            f"  round {entry['round']}  improved {s['n_improved']}/{s['n_clients']}  "
            f"mean Δppl {s['mean_delta_ppl']:+.4f}  cluster helps {s['n_cluster_helps']}/{s['n_clients']}"
        )
    for comp in feed["comparisons"]:
        lines.append(
            f"  r{comp['from_round']}→r{comp['to_round']}  composite better on "
            f"{comp['n_composite_better']}/{comp['n_clients']} clients"
        )
    lines.append(f"  feed     {paths.relative(out)}")
    if not args.no_report:
        lines.append(report_line(_report(feed, paths.reports / "personalization_report.md")))
    lines.append("")
    emit(lines)
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
