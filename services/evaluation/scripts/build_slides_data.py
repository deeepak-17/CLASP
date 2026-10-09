#!/usr/bin/env python3
"""Pick the headline numbers for the demo and slides from the committed results.

Reads ``results/personalization.json``, ``results/noise_report.json`` and
``results/in_project_metric.json`` and writes:

* ``results/slides_data.json`` — each headline with its value, a one-line
  claim, and the file + field it was read from;
* ``docs/report/slides_data.md`` — the same as a table for the deck.

A headline is never typed in by hand: change a result, re-run, and the slide
number follows.

Usage::

    python scripts/build_slides_data.py
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, run_cli, setup_logging
from evaluation.utils.io_utils import read_json, write_json
from evaluation.utils.paths import project_paths
from evaluation.utils.reporting import MarkdownReport
from evaluation.utils.timing import utc_timestamp


def build_parser() -> argparse.ArgumentParser:
    return base_parser(__doc__.splitlines()[0])


def headlines(pers: dict[str, Any], noise: dict[str, Any], inproj: dict[str, Any]) -> list[dict[str, Any]]:
    first, last = pers["rounds"][0], pers["rounds"][-1]
    s1, s2 = first["summary"], last["summary"]
    comp = pers["comparisons"][-1]
    changes = [abs(r["composite_ppl_change"]) for r in comp["clients"]]
    lo, hi = min(changes), max(changes)
    g = noise["humaneval_guard"]
    t = noise["locked_thresholds"]
    abl = {r["cluster"]: r for r in inproj["aggregation_ablation"]["rows"]}
    rows = {r["version"]: r for r in inproj["in_project_completion"]["rows"]}
    naive = next(r for v, r in rows.items() if "naive" in v)
    svd = next(r for v, r in rows.items() if "SVD" in v)
    return [
        {
            "id": "personalization",
            "value": f"{s2['n_improved']}/{s2['n_clients']} clients, mean {s2['mean_delta_ppl']:+.3f} ppl",
            "claim": "Every client's composite has lower perplexity than the base on its own never-trained-on code.",
            "source": "results/personalization.json · rounds[-1].summary",
        },
        {
            "id": "d3_cluster_layer",
            "value": f"helps {s1['n_cluster_helps']}/{s1['n_clients']} → {s2['n_cluster_helps']}/{s2['n_clients']} clients (vs α = 0)",
            "claim": "Supports D3, not conclusive: in round 2, α = 0 removes a layer the client was trained on.",
            "source": "results/personalization.json · rounds[*].summary.n_cluster_helps",
        },
        {
            "id": "round_over_round",
            "value": f"lower on {comp['n_composite_better']}/{comp['n_clients']} clients, by {lo:.3f}–{hi:.3f} ppl",
            "claim": "Within unmeasured seed noise: one seed, no interval.",
            "source": "results/personalization.json · comparisons[-1]",
        },
        {
            "id": "aggregation",
            "value": " · ".join(f"{c} {r['ratio']}" for c, r in abl.items()),
            "claim": "SVD aggregation is ~2× closer to the exact average than naive factor averaging.",
            "source": "results/in_project_metric.json · aggregation_ablation",
        },
        {
            "id": "in_project",
            "value": f"edit sim {naive['edit_similarity']:.3f} → {svd['edit_similarity']:.3f}",
            "claim": "The SVD aggregate also completes held-out code better (within noise at 60 examples).",
            "source": "results/in_project_metric.json · in_project_completion",
        },
        {
            "id": "guard_baseline",
            "value": f"pass@1 {g['pass_at_1']:.2f} (95% CI {g['bootstrap_ci_95'][0]:.2f}–{g['bootstrap_ci_95'][1]:.2f})",
            "claim": f"Base model on the {g['n_tasks']}-task HumanEval guard subset.",
            "source": "results/noise_report.json · humaneval_guard",
        },
        {
            "id": "guard_noise",
            "value": f"{t['noise_floor_at_scored_n'] * 100:.1f} pts now, {t['noise_floor_full_benchmark'] * 100:.1f} pts on full HumanEval",
            "claim": f"D5's {g['tolerance'] * 100:g}-point tolerance is below the guard's noise floor.",
            "source": "results/noise_report.json · locked_thresholds",
        },
    ]


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    res = paths.eval_results
    items = headlines(
        read_json(res / "personalization.json"),
        read_json(res / "noise_report.json"),
        read_json(res / "in_project_metric.json"),
    )
    out = res / "slides_data.json"
    write_json(out, {"generated_at": utc_timestamp(), "headlines": items})
    md = MarkdownReport("Slides data", subtitle="Headline numbers for the demo and deck, read from committed results")
    md.table(["headline", "value", "claim", "source"], [[h["id"], h["value"], h["claim"], h["source"]] for h in items])
    md_path = md.write(paths.docs / "report" / "slides_data.md")
    emit(["", "CLASP-P5 · slides data", "=" * 68, *[f"  {h['id']:<18} {h['value']}" for h in items],
          f"  data     {paths.relative(out)}", f"  table    {paths.relative(md_path)}", ""])
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
