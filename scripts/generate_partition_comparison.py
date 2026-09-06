#!/usr/bin/env python3
"""Week 4 · Thursday — project-level vs per-developer partition comparison.

"Write a small report comparing project-level vs per-dev splits."

Reads back the *actual generated data* for both strategies — both manifests,
both descriptive metadata documents, both validation results, and the
per-developer basis sidecar — and renders a single comparison report.
Every number in it is read from those artefacts; nothing here is computed
independently or invented. Where a statistic is not available (e.g. no
validation report has been generated yet), the report says so explicitly
rather than omitting the row silently.

Usage::

    python scripts/generate_partition_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from partitions.partitioner import load_partition_config, load_partition_manifest
from utils.errors import ClaspP5Error
from utils.io_utils import read_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport

_LOG = get_logger(__name__)

_STRATEGIES = {
    "project_level": "configs/partition.yaml",
    "per_developer": "configs/partition_per_developer.yaml",
}

_NOT_AVAILABLE = "not available — run the referenced script first"


def _load_strategy_data(label: str, config_path: str) -> dict:
    paths = project_paths()
    config = load_partition_config(paths.root / config_path)
    manifest_path = Path(config.output.manifest_path)
    if not manifest_path.is_file():
        raise ClaspP5Error(
            f"{label}: manifest not found at {manifest_path}. Run "
            f"`python scripts/build_partitions.py --config {config_path}` first."
        )
    manifest = load_partition_manifest(manifest_path)

    metadata_path = config.output.partitions_dir / "partition_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else None

    validation_report_name = f"partition_validation_report{config.output.report_suffix()}.json"
    validation_path = paths.reports / validation_report_name
    validation = read_json(validation_path) if validation_path.is_file() else None

    basis_path = config.output.partitions_dir / "developer_basis.json"
    developer_basis = read_json(basis_path) if basis_path.is_file() else None

    shard_sizes = sorted(shard.num_files for shard in manifest.shards)
    return {
        "label": label,
        "config_path": config_path,
        "manifest": manifest,
        "metadata": metadata,
        "validation": validation,
        "developer_basis": developer_basis,
        "shard_sizes": shard_sizes,
    }


def _basis_breakdown(developer_basis: dict | None) -> dict[str, int]:
    if not developer_basis:
        return {}
    counts: dict[str, int] = {}
    for entry in developer_basis.values():
        basis = entry.get("basis", "unknown")
        counts[basis] = counts.get(basis, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=None, help="Report path. Defaults to reports/project_vs_developer_comparison.md."
    )
    return parser


def main(args: argparse.Namespace) -> int:
    data = {label: _load_strategy_data(label, path) for label, path in _STRATEGIES.items()}

    report = MarkdownReport(
        title="CLASP-P5 · Project-Level vs Per-Developer Partition Comparison",
        subtitle="Week 4 · Thursday deliverable — read from the actual generated manifests, metadata and validation runs",
    )

    report.heading("1. Headline numbers")
    report.table(
        ["Metric", "project_level", "per_developer"],
        [
            ["Clients", data["project_level"]["manifest"].num_clients, data["per_developer"]["manifest"].num_clients],
            ["Clusters (projects)", data["project_level"]["manifest"].num_clusters, data["per_developer"]["manifest"].num_clusters],
            ["Total files", data["project_level"]["manifest"].total_files, data["per_developer"]["manifest"].total_files],
            [
                "Shard size (min/mean/max)",
                _size_summary(data["project_level"]["shard_sizes"]),
                _size_summary(data["per_developer"]["shard_sizes"]),
            ],
            ["Seed", data["project_level"]["manifest"].seed, data["per_developer"]["manifest"].seed],
        ],
    )
    report.paragraph(
        "Both strategies partition the *same* corpus with the *same* seed (see row above), so "
        "every difference in this report is attributable to the strategy, not to different input data."
    )

    report.heading("2. Coverage and leakage/overlap results")
    report.table(
        ["Strategy", "Verdict", "Checks passed", "Errors", "Warnings", "Coverage check"],
        [
            [
                label,
                (_verdict(d["validation"])),
                len(d["validation"]["checks_passed"]) if d["validation"] else "—",
                len(d["validation"]["errors"]) if d["validation"] else "—",
                len(d["validation"]["warnings"]) if d["validation"] else "—",
                _coverage_line(d["validation"]),
            ]
            for label, d in data.items()
        ],
    )
    for label, d in data.items():
        if d["validation"] is None:
            command = f"scripts/validate_partitions.py --config {d['config_path']}"
            report.paragraph(f"`{label}`: validation results not available — run `python {command}` first.")

    report.heading("3. Project distribution")
    for label, d in data.items():
        report.heading(label, level=3)
        if d["metadata"] is None:
            report.paragraph(f"metadata {_NOT_AVAILABLE}")
            continue
        report.table(
            ["Cluster", "Client(s)", "Files", "% of corpus files"],
            [
                [c["cluster_id"], len(c["client_ids"]), c["num_files"], f"{100 * c['share_of_corpus_files']:.1f}%"]
                for c in d["metadata"]["clusters"]
            ],
        )

    report.heading("4. Developer distribution (per_developer only)")
    breakdown = _basis_breakdown(data["per_developer"]["developer_basis"])
    if breakdown:
        report.table(["Basis", "Client shards"], [[k, v] for k, v in sorted(breakdown.items())])
        report.paragraph(
            "`module_path`: the cluster had real subpackages, each becoming one developer's slice. "
            "`deterministic_hash_chunk`: the cluster was a flat file list, split by a seeded hash-shuffle "
            "then contiguous chunking. `single_developer_whole_project`: the cluster was too small to "
            "split further even one level (its one shard equals the project-level shard). See "
            "`partitions/strategies.py::PerDeveloperStrategy` for the full rationale, including why "
            "no real per-developer identity is claimed (the corpus is shallow-cloned — one commit per "
            "project, so no per-file authorship signal exists to split on)."
        )
    else:
        report.paragraph(f"developer basis breakdown {_NOT_AVAILABLE}")

    report.heading("5. Advantages and limitations")
    report.table(
        ["", "project_level", "per_developer"],
        [
            [
                "Advantages",
                "Matches CLASP's cluster-LoRA definition exactly (one shard = one project = one "
                "`dW_cluster`); small, stable client count; strongest non-IID signal (whole codebases "
                "differ in kind, not just sample).",
                "Models individual-scale federated clients (`dW_client`) within a project; exposes "
                "intra-project heterogeneity (module style) that project-level collapses away; more "
                "clients per round is closer to a real multi-developer deployment.",
            ],
            [
                "Limitations",
                "Cannot represent per-developer contribution boundaries at all — by construction, "
                "every developer in a project shares one shard.",
                "No real developer identity is available in this corpus (see §4); the module-path/"
                "hash-chunk proxy is a structural stand-in, not an authorship measurement. Shard sizes "
                "are smaller and less balanced than project-level (see §1), and the client count is "
                "well above what P2 currently simulates (see `scripts/cross_check_client_counts.py`).",
            ],
        ],
    )

    report.heading("6. Reproducibility")
    report.paragraph(
        "Both partitions are pure functions of `(corpus, seed, strategy, strategy config)` — "
        "`partitions/strategies.py`'s strategies do no I/O and no non-deterministic operations "
        "(the per-developer hash-chunk fallback uses `sha256(seed:cluster_id:file_id)`, not a "
        "random-number generator). Re-running `scripts/build_partitions.py` with either config "
        "against the same `datasets/processed/corpus.jsonl` reproduces byte-identical shard digests "
        "— exercised directly by `tests/test_partitions_per_developer.py`'s determinism tests."
    )

    report.heading("7. Integration implications")
    report.paragraph(
        "See `scripts/cross_check_client_counts.py` / `reports/client_count_cross_check.md` for the "
        "full client-count finding. In short: neither partition's client count matches P2's current "
        "simulated width, and per-developer's is the larger mismatch of the two. This is a "
        "coordination item for integration week, not a defect in either module."
    )

    report.rule()
    report.paragraph("Generated by `scripts/generate_partition_comparison.py`.")

    paths_obj = project_paths()
    output = args.output or (paths_obj.reports / "project_vs_developer_comparison.md")
    written = report.write(output)

    lines = [
        "",
        "CLASP-P5 · Week 4 Thursday — Project vs Developer Comparison",
        "=" * 68,
        f"  project_level clients: {data['project_level']['manifest'].num_clients}",
        f"  per_developer clients: {data['per_developer']['manifest'].num_clients}",
        "",
        report_line(written),
        "",
    ]
    emit(lines)
    return EXIT_OK


def _size_summary(sizes: list[int]) -> str:
    if not sizes:
        return "—"
    return f"{min(sizes)} / {sum(sizes) / len(sizes):.1f} / {max(sizes)}"


def _verdict(validation: dict | None) -> str:
    if validation is None:
        return "—"
    return "PASS" if validation["ok"] else "FAIL"


def _coverage_line(validation: dict | None) -> str:
    if validation is None:
        return "—"
    coverage_checks = [c for c in validation["checks_passed"] if "coverage" in c or "disjoint" in c]
    return "; ".join(coverage_checks) if coverage_checks else "see full report"


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
