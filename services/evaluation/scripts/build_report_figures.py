#!/usr/bin/env python3
"""Regenerate the evaluation report's figures and results tables from committed results.

Inputs (all under ``results/``): ``personalization.json``, ``noise_report.json``
and ``in_project_metric.json``. Outputs:

* ``docs/report/figures/*.svg`` — the evaluation-section figures;
* ``docs/report/results_tables.md`` — every number the evaluation section
  quotes, as tables, with the file each came from.

Run the exporters first (``build_personalization_report.py``,
``build_noise_report.py``); this script only reads their output, so the
report can never show a number the pipeline did not produce.

Usage::

    python scripts/build_report_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, run_cli, setup_logging
from evaluation.eval_harness.noise import paired_min_detectable_drop
from evaluation.eval_harness.personalization import FEED_VERSION
from evaluation.utils.errors import EvaluationError
from evaluation.utils.io_utils import atomic_write_text, read_json
from evaluation.utils.paths import project_paths
from evaluation.utils.reporting import MarkdownReport
from evaluation.utils.svg_charts import Series, grouped_bars, lines

TASK_COUNTS = (20, 50, 100, 164, 250, 500, 1000)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to docs/report/.")
    return parser


def _short(client_id: str) -> str:
    return client_id.split("/")[-1].removeprefix("client-")


def _per_client(feed: dict[str, Any], key) -> tuple[list[str], list[Series]]:
    clients = [c["client_id"] for c in feed["rounds"][0]["clients"]]
    series = []
    for entry in feed["rounds"]:
        by_id = {c["client_id"]: c for c in entry["clients"]}
        series.append(Series(f"Round {entry['round']}", [key(by_id[c]) if c in by_id else None for c in clients]))
    return [_short(c) for c in clients], series


def figures(pers: dict[str, Any], noise: dict[str, Any], inproj: dict[str, Any]) -> dict[str, str]:
    cats, reduction = _per_client(pers, lambda c: round(c["base_ppl"] - c["composite_ppl"], 4))
    _, contribution = _per_client(pers, lambda c: c["cluster_contribution_at_alpha_ref"])
    out = {
        "fig_personalization_gain.svg": grouped_bars(
            cats, reduction,
            title="Perplexity reduction vs the frozen base, per client",
            subtitle="Held-out files never trained on; composite = base + α·cluster + β·client. Higher is better.",
            y_label="base − composite perplexity", fmt=lambda v: f"{v:.2f}",
        ),
        "fig_cluster_contribution.svg": grouped_bars(
            cats, contribution,
            title="Cluster layer's effect at α = 0.5",
            subtitle="Composite − client-only perplexity. Below zero, the cluster layer helps.",
            y_label="Δ perplexity", fmt=lambda v: f"{v:+.3f}" if v else "0",
        ),
    }
    abl = inproj["aggregation_ablation"]["rows"]
    out["fig_aggregation_error.svg"] = grouped_bars(
        [r["cluster"] for r in abl],
        [Series("Naive per-factor average", [r["naive_avg"] for r in abl]),
         Series("Exact SVD (rank 16)", [r["svd"] for r in abl])],
        title="Aggregation error vs the exact weighted average",
        subtitle="Relative Frobenius error over 96 modules on the trained adapters. Lower is better.",
        y_label="relative error", fmt=lambda v: f"{v:.2f}",
    )
    comp = inproj["in_project_completion"]["rows"]
    out["fig_in_project_completion.svg"] = grouped_bars(
        ["edit similarity", "exact match"],
        [Series(f"{r['client']} · {r['version']}", [r["edit_similarity"], r["exact_match"]]) for r in comp],
        title="In-project next-line completion by aggregate",
        subtitle=f"{inproj['held_out_examples']} held-out examples, greedy decoding. Higher is better.",
        fmt=lambda v: f"{v:.2f}",
    )
    g = noise["humaneval_guard"]
    grid = [row["discordance"] for row in g["paired"]]
    out["fig_guard_noise_floor.svg"] = lines(
        TASK_COUNTS,
        [Series(f"discordance {d}", [paired_min_detectable_drop(d, n) * 100 for n in TASK_COUNTS]) for d in grid],
        title="Smallest HumanEval pass@1 drop distinguishable from noise",
        subtitle="Paired comparison on the same tasks, 95% two-sided. Dashed: the D5 tolerance.",
        x_label="tasks scored (log scale)", y_label="pass@1 points", log_x=True,
        reference=(g["tolerance"] * 100, f"D5 tolerance {g['tolerance'] * 100:g} pts"),
        fmt=lambda v: f"{v:.0f}",
    )
    return out


def tables(pers: dict[str, Any], noise: dict[str, Any], inproj: dict[str, Any], path: Path) -> Path:
    report = MarkdownReport("Evaluation results tables", subtitle="Every number quoted in the evaluation section")
    for entry in pers["rounds"]:
        s = entry["summary"]
        report.heading(f"Personalization — {entry['label']}")
        report.table(
            ["client", "tokens", "base", "client only", "composite", "Δ vs base", "cluster @ α=0.5", "best α"],
            [[c["client_id"], c["n_tokens"], c["base_ppl"], c["client_only_ppl"], c["composite_ppl"],
              f"{c['personalization_delta_ppl']:+.3f}", f"{c['cluster_contribution_at_alpha_ref']:+.3f}",
              c["best_alpha"]] for c in entry["clients"]],
        )
        report.paragraph(
            f"Improved over base: {s['n_improved']}/{s['n_clients']}; mean Δ {s['mean_delta_ppl']:+.4f} "
            f"(range {s['min_delta_ppl']:+.3f} to {s['max_delta_ppl']:+.3f}); cluster layer helps "
            f"{s['n_cluster_helps']}/{s['n_clients']}; sweep chose α = 0 for {s['n_alpha_zero']}/{s['n_clients']}."
        )
    report.heading("Aggregation error (relative Frobenius vs exact average)")
    report.table(["cluster", "naive average", "exact SVD", "ratio"],
                 [[r["cluster"], r["naive_avg"], r["svd"], r["ratio"]] for r in inproj["aggregation_ablation"]["rows"]])
    report.heading("In-project completion")
    report.table(["client", "cluster", "aggregate", "edit similarity", "exact match"],
                 [[r["client"], r["cluster"], r["version"], r["edit_similarity"], r["exact_match"]]
                  for r in inproj["in_project_completion"]["rows"]])
    g = noise["humaneval_guard"]
    report.heading("HumanEval guard noise")
    report.key_values({
        "tasks scored": g["n_tasks"], "base pass@1": g["pass_at_1"], "standard error": g["standard_error"],
        "95% bootstrap interval": f"{g['bootstrap_ci_95'][0]} – {g['bootstrap_ci_95'][1]}",
        "D5 tolerance": g["tolerance"],
    })
    report.table(["discordance", f"floor @ {g['n_tasks']}", f"floor @ {g['full_benchmark_size']}",
                  f"tasks for {g['tolerance']}"],
                 [[r["discordance"], r["min_detectable_drop_at_n"], r["min_detectable_drop_full_benchmark"],
                   r["items_needed_for_tolerance"]] for r in g["paired"]])
    report.heading("Sources")
    report.bullets([f"`{s}`" for s in pers["sources"]] + [f"`{noise['sources']['baseline_anchor']}`",
                                                           "`results/in_project_metric.json`"])
    return report.write(path)


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    res = paths.eval_results
    try:
        pers = read_json(res / "personalization.json")
        noise = read_json(res / "noise_report.json")
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error with the fix
        raise EvaluationError(
            f"{exc} — run build_personalization_report.py and build_noise_report.py first"
        ) from exc
    if not str(pers.get("feed_version", "")).startswith(FEED_VERSION.split(".")[0] + "."):
        raise EvaluationError("personalization.json was written by an incompatible exporter version")
    inproj = read_json(res / "in_project_metric.json")

    out_dir = args.out_dir or paths.docs / "report"
    fig_dir = out_dir / "figures"
    written = []
    for name, svg in figures(pers, noise, inproj).items():
        written.append(atomic_write_text(fig_dir / name, svg))
    table_path = tables(pers, noise, inproj, out_dir / "results_tables.md")

    emit(["", "CLASP-P5 · report figures", "=" * 68,
          *[f"  figure   {paths.relative(p)}" for p in written],
          f"  tables   {paths.relative(table_path)}", ""])
    return EXIT_OK


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
