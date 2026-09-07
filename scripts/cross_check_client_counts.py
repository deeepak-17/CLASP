#!/usr/bin/env python3
"""Cross-check both partition strategies against P2's client count.

"Cross-check both partition types against Prasanth's client count."

Reuses exactly the same admission logic Week 2's
``scripts/check_contract_compliance.py`` already applies to the
project-level manifest (``interfaces.cluster_client.MockClusterClient`` —
"at least 2 clients", "no empty shards", and a *warning* — not an error —
when the declared client count differs from what P2 currently simulates),
and runs it a second time against the per-developer manifest. No new
admission rules are invented for Week 4.

This script does not change P5's partitioning strategy to make the counts
agree — per the Week-4 instructions, a mismatch is an integration/
coordination issue to raise, not something P5 resolves unilaterally by
picking numbers that happen to fit P2's current simulator width.

Usage::

    python scripts/cross_check_client_counts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from interfaces.cluster_client import MockClusterClient, PlanAcceptance
from partitions.partitioner import load_partition_config, load_partition_manifest
from utils.errors import ClaspP5Error
from utils.io_utils import write_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport

_LOG = get_logger(__name__)

_CONFIGS = {
    "project_level": "configs/partition.yaml",
    "per_developer": "configs/partition_per_developer.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=None, help="Report path. Defaults to reports/client_count_cross_check.md."
    )
    return parser


def _check_one(label: str, config_path: str) -> dict:
    paths = project_paths()
    config = load_partition_config(paths.root / config_path)
    manifest_path = Path(config.output.manifest_path)
    if not manifest_path.is_file():
        raise ClaspP5Error(
            f"{label}: manifest not found at {manifest_path}. Run "
            f"`python scripts/build_partitions.py --config {config_path}` first."
        )
    manifest = load_partition_manifest(manifest_path)

    client = MockClusterClient(
        expected_client_count=config.integration.expected_client_count,
        max_client_count=config.integration.max_client_count,
    )
    acceptance: PlanAcceptance = client.register_partition_plan(manifest)
    return {
        "label": label,
        "config_path": config_path,
        "strategy": manifest.strategy.value,
        "actual_client_count": manifest.num_clients,
        "num_clusters": manifest.num_clusters,
        "p2_expected_client_count": config.integration.expected_client_count,
        "p2_max_client_count": config.integration.max_client_count,
        "accepted": acceptance.accepted,
        "errors": acceptance.errors,
        "warnings": acceptance.warnings,
        "matches_p2_expectation": manifest.num_clients == config.integration.expected_client_count,
    }


def main(args: argparse.Namespace) -> int:
    paths = project_paths()
    results = [_check_one(label, path) for label, path in _CONFIGS.items()]

    lines = [
        "",
        "CLASP-P5 · P2 Client-Count Cross-Check",
        "=" * 68,
        "",
    ]
    any_hard_error = False
    for r in results:
        any_hard_error = any_hard_error or not r["accepted"]
        lines.append(
            f"  {r['label']:<14} actual={r['actual_client_count']:<4} "
            f"P2 expects={r['p2_expected_client_count']:<4} "
            f"{'MATCH' if r['matches_p2_expectation'] else 'MISMATCH (warning, not a failure)'}"
        )
        for w in r["warnings"]:
            lines.append(f"      warn  {w}")
        for e in r["errors"]:
            lines.append(f"      FAIL  {e}")
    lines.append("")
    lines.append(
        "  Verdict: this is a coordination finding for integration week, not a defect in "
        "either module. See the report for what needs to be decided."
    )
    lines.append("")

    if not args.no_report:
        report = _render_report(results)
        output = args.output or (paths.reports / "client_count_cross_check.md")
        path = report.write(output)
        lines.append(report_line(path))
        json_path = write_json(output.with_suffix(".json"), {"checks": results})
        lines.append(f"  report (json): {paths.relative(json_path)}")
        lines.append("")

    emit(lines)
    # A count MISMATCH is a warning (matches Week-2 precedent), not a failure;
    # only a hard MockClusterClient rejection (e.g. <2 clients) fails the run.
    return EXIT_OK if not any_hard_error else EXIT_FAILURE


def _render_report(results: list[dict]) -> MarkdownReport:
    report = MarkdownReport(
        title="CLASP-P5 · P2 Client-Count Cross-Check",
        subtitle="project-level and per-developer counts vs P2's current contract",
    )
    report.heading("1. Verdict")
    mismatches = [r for r in results if not r["matches_p2_expectation"]]
    report.status_line(
        all(r["accepted"] for r in results),
        f"{len(mismatches)}/{len(results)} strategy(ies) differ from P2's current expected_client_count "
        f"(reported as warnings, not failures — see the finding below)",
    )

    report.heading("2. Actual vs expected counts")
    report.table(
        ["Strategy", "Actual P5 clients", "Clusters (projects)", "P2 expected", "P2 max", "Match?"],
        [
            [
                r["strategy"],
                r["actual_client_count"],
                r["num_clusters"],
                r["p2_expected_client_count"],
                r["p2_max_client_count"],
                r["matches_p2_expectation"],
            ]
            for r in results
        ],
    )

    report.heading("3. Why the counts differ")
    report.paragraph(
        "P2's Cluster Layer currently simulates a fixed, small number of Flower clients "
        "(`expected_client_count` in `configs/partition*.yaml`'s `integration` section — 3 as of "
        "Week 2/3, per `interfaces/cluster_client.py`'s `DEFAULT_EXPECTED_CLIENTS`, rising to 5 "
        "per cluster from Week 8). Both P5 partition strategies derive their client count from "
        "the corpus and the strategy, not from P2's simulator width:"
    )
    report.bullets(
        [
            "**project_level** — one client per source project (6 in D1). Already exceeded P2's "
            "Week-2/3 simulated count of 3; this is the same finding Week 2's "
            "`check_contract_compliance.py` already surfaced and the README already documents.",
            "**per_developer** — one client per developer slice *within* each project (module-path "
            "or hash-chunk groups; see `partitions/strategies.py`), which is structurally always "
            ">= the project-level count and typically several times larger, because it is meant "
            "to model individual-scale federated clients, not project-scale ones.",
        ]
    )

    report.heading("4. What needs to be decided during integration")
    report.bullets(
        [
            "Whether P2's simulator width should scale to match P5's per-developer client count "
            "for any experiment that specifically exercises per-developer federation, or whether "
            "per-developer partitioning is intended for a *future* phase where P2's simulator has "
            "already scaled (a later planned client-count increase is a step in "
            "that direction, but even 5 clients/cluster is below the per-developer count here).",
            "Whether project-level remains the client boundary for the Phase-II demo (matching "
            "P2's current 3-5 simulated clients) while per-developer is evaluated/reported "
            "separately as a data-pipeline capability, without being wired into a live FedProx "
            "round this phase.",
            "Neither option requires a P5 code change — both configs and both manifests already "
            "exist and validate cleanly; this is purely a scope/scheduling decision across P2 and P5.",
        ]
    )

    report.rule()
    report.paragraph("Generated by `scripts/cross_check_client_counts.py`.")
    return report


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
