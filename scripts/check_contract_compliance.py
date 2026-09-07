#!/usr/bin/env python3
"""Sanity-check partitions against the interface contracts.

"Sanity-check partitions against Deepak's interface contract."

Runs five independent checks, each targeting a different way the P5 <-> P1-P4
seam can break:

1. **Schema** — ``manifest.json`` validates against
   ``interfaces/schemas/partition_manifest.schema.json`` structurally and
   semantically.
2. **Provenance** — the manifest's ``corpus_sha256`` matches the corpus
   manifest the collector wrote, and every declared shard file exists with the
   declared file count.
3. **P2 (Cluster)** — ``MockClusterClient`` accepts the manifest as a
   federated round plan under P2's admission rules.
4. **P4 (Registry)** — a snapshot round-trips through ``MockRegistryClient``
   and validates against the snapshot-metadata schema, proving P5 can read
   what P4 will write.
5. **P5 -> P4** — a representative ``EvalResult`` serialises and validates,
   proving the shape P5 will publish is the shape P4's rollback logic expects.

Exit code is non-zero on any failure, so this is the Week-2 sign-off gate.

Usage::

    python scripts/check_contract_compliance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from interfaces.cluster_client import MockClusterClient
from interfaces.contracts import (
    AdapterKind,
    AdapterRef,
    BenchmarkName,
    EvalResult,
    PartitionManifest,
)
from interfaces.registry_client import MockRegistryClient
from interfaces.validation import (
    ValidationReport,
    jsonschema_available,
    validate_document,
    validate_file,
)
from partitions.partitioner import load_partition_config
from utils.errors import ClaspP5Error
from utils.io_utils import read_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.timing import utc_timestamp

_LOG = get_logger(__name__)


class ComplianceRun:
    """Accumulates the outcome of every contract check."""

    def __init__(self) -> None:
        self.reports: list[ValidationReport] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors and all(report.ok for report in self.reports)

    def add(self, report: ValidationReport) -> None:
        self.reports.append(report)
        _LOG.info(report.summary())

    def fail(self, message: str) -> None:
        self.errors.append(message)
        _LOG.error(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        _LOG.warning(message)

    def note(self, message: str) -> None:
        self.notes.append(message)
        _LOG.info(message)


def check_manifest_schema(run: ComplianceRun, manifest_path: Path) -> PartitionManifest | None:
    """Check 1 — the partition manifest satisfies its published schema."""
    report = validate_file("partition_manifest", manifest_path)
    run.add(report)
    if not report.ok:
        return None
    return PartitionManifest.from_dict(read_json(manifest_path))


def check_provenance(run: ComplianceRun, manifest: PartitionManifest, corpus_manifest_path: Path | None) -> None:
    """Check 2 — the manifest points at real, unchanged inputs and outputs."""
    paths = project_paths()

    if corpus_manifest_path and Path(corpus_manifest_path).is_file():
        corpus_manifest = read_json(corpus_manifest_path)
        recorded = corpus_manifest.get("corpus_sha256")
        if recorded != manifest.corpus_sha256:
            run.fail(
                "Partition manifest corpus_sha256 does not match the corpus manifest "
                f"({manifest.corpus_sha256[:12]}… vs {str(recorded)[:12]}…). "
                "The partition is stale — rebuild it."
            )
        else:
            run.note(f"corpus provenance matches ({manifest.corpus_sha256[:12]}…)")
        if corpus_manifest.get("synthetic"):
            run.warn(
                "The underlying corpus is synthetic. Contract shapes are valid, but these "
                "partitions must not be used for training or reported results."
            )
    else:
        run.warn("Corpus manifest not found; skipped the corpus_sha256 provenance check")

    from utils.io_utils import read_jsonl

    for shard in manifest.shards:
        shard_path = paths.root / shard.files_path
        if not shard_path.is_file():
            run.fail(f"Shard file declared in the manifest is missing: {shard.files_path}")
            continue
        actual = sum(1 for _ in read_jsonl(shard_path))
        if actual != shard.num_files:
            run.fail(
                f"Shard '{shard.client_id}' declares {shard.num_files} file(s) "
                f"but {shard.files_path} contains {actual}"
            )
    if not run.errors:
        run.note(f"all {len(manifest.shards)} shard file(s) present with the declared counts")


def check_cluster_consumer(run: ComplianceRun, manifest: PartitionManifest, expected: int, maximum: int) -> None:
    """Check 3 — P2's Cluster Layer would accept this plan."""
    client = MockClusterClient(expected_client_count=expected, max_client_count=maximum)
    acceptance = client.register_partition_plan(manifest)

    for error in acceptance.errors:
        run.fail(f"P2 (Cluster) rejected the plan: {error}")
    for warning in acceptance.warnings:
        run.warn(f"P2 (Cluster): {warning}")
    if acceptance.accepted:
        run.note(acceptance.summary())


def check_registry_reader(run: ComplianceRun, manifest: PartitionManifest) -> None:
    """Check 4 — P5 can read the snapshot shape P4 will write."""
    registry = MockRegistryClient()
    cluster_id = manifest.cluster_ids[0]
    ref = AdapterRef(name=cluster_id, version=1, kind=AdapterKind.CLUSTER, cluster_id=cluster_id)
    snapshot = registry.add_snapshot(
        ref,
        round_number=1,
        artifact_path=f"registry/{cluster_id}/v1/adapter.safetensors",
        artifact_sha256="0" * 64,
        training_loss=1.234,
    )

    run.add(
        validate_document(
            "snapshot_metadata",
            snapshot.to_dict(),
            document_name=f"MockRegistryClient snapshot {ref.uri}",
        )
    )

    latest = registry.latest(cluster_id, AdapterKind.CLUSTER)
    if latest.adapter != ref:
        run.fail("MockRegistryClient.latest() did not return the snapshot that was written")
    else:
        run.note(f"registry read path round-trips ({ref.uri})")


def check_eval_result_shape(run: ComplianceRun, manifest: PartitionManifest) -> None:
    """Check 5 — the EvalResult P5 will publish matches the contract."""
    cluster_id = manifest.cluster_ids[0]
    result = EvalResult(
        adapter=AdapterRef(name=cluster_id, version=1, kind=AdapterKind.CLUSTER, cluster_id=cluster_id),
        benchmark=BenchmarkName.HUMANEVAL,
        # Representative value. Real Pass@k lands in Week 3; this exercises
        # the shape P4's rollback threshold will read.
        pass_at_k={1: 0.0},
        num_tasks=len(manifest.shards),
        num_samples_per_task=1,
        created_at=utc_timestamp(),
        run_id="contract-check",
    )
    run.add(
        validate_document(
            "eval_result", result.to_dict(), document_name="representative EvalResult"
        )
    )
    run.note(
        "EvalResult values are placeholders; scoring runs against the mock backend; "
        "only the shape is being certified here"
    )


def render_report(run: ComplianceRun, manifest: PartitionManifest | None) -> MarkdownReport:
    report = MarkdownReport(
        title="CLASP-P5 · Interface Contract Compliance Report",
        subtitle="partition sign-off against the P1–P4 seam",
    )

    report.heading("1. Verdict")
    report.status_line(
        run.ok,
        f"{len(run.reports)} contract check(s), {len(run.errors)} error(s), "
        f"{len(run.warnings)} warning(s)",
    )
    if manifest:
        report.key_values(
            {
                "Manifest version": manifest.manifest_version,
                "Contract version": manifest.contract_version,
                "Strategy": manifest.strategy.value,
                "Clients": manifest.num_clients,
                "Clusters": manifest.num_clusters,
                "Total files": manifest.total_files,
                "JSON Schema validation": "enabled" if jsonschema_available() else "SKIPPED (jsonschema absent)",
            }
        )

    report.heading("2. Contract checks")
    report.table(
        ["Contract", "Document", "Result", "Passed", "Errors", "Warnings", "Skipped"],
        [
            [
                item.contract,
                item.document,
                "PASS" if item.ok else "FAIL",
                len(item.checks_run),
                len(item.errors),
                len(item.warnings),
                len(item.skipped_checks),
            ]
            for item in run.reports
        ],
    )

    detail_errors = [f"`{i.contract}` — {msg}" for i in run.reports for msg in i.errors]
    if detail_errors or run.errors:
        report.heading("Errors", level=3)
        report.bullets(detail_errors + run.errors)

    detail_warnings = [f"`{i.contract}` — {msg}" for i in run.reports for msg in i.warnings]
    if detail_warnings or run.warnings:
        report.heading("Warnings", level=3)
        report.bullets(detail_warnings + run.warnings)

    skipped = [f"`{i.contract}` — {msg}" for i in run.reports for msg in i.skipped_checks]
    if skipped:
        report.heading("Skipped checks", level=3)
        report.bullets(skipped)

    report.heading("3. Cross-module findings")
    report.bullets(run.notes or ["—"])

    report.heading("4. Seam coverage")
    report.table(
        ["Owner", "Module", "Direction", "Contract", "Verified by"],
        [
            ["P5", "Eval & Data", "produces", "PartitionManifest", "schema + semantic validation"],
            ["P2", "Cluster Layer", "consumes", "PartitionManifest", "MockClusterClient admission rules"],
            ["P4", "State Registry", "produces", "SnapshotMetadata", "MockRegistryClient read round-trip"],
            ["P5", "Eval & Data", "produces", "EvalResult", "schema validation of a representative document"],
            ["P1", "Edge Layer", "consumes", "PartitionShard", "shard files present with declared counts"],
        ],
    )

    report.heading("5. Standing caveats")
    report.bullets(
        [
            "`interfaces/contracts.py` is P5's **mirror** of the contracts package P4 owns. "
            "When P4 freezes `/contracts`, re-point the mirror and re-run this check — a "
            "divergence will surface here rather than at integration time.",
            "Pass@k values in the certified `EvalResult` are placeholders (mock backend).",
            "`SnapshotMetadata` is exercised against `MockRegistryClient`, not P4's live "
            "FastAPI service, which is integrated separately.",
        ]
    )

    report.rule()
    report.paragraph("Generated by `scripts/check_contract_compliance.py`.")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="Partition config. Defaults to configs/partition.yaml."
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Override the manifest path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. Defaults to reports/contract_compliance_report.md.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_partition_config(args.config)
    manifest_path = Path(args.manifest or config.output.manifest_path)
    paths = project_paths()

    if not manifest_path.is_file():
        raise ClaspP5Error(
            f"Partition manifest not found at {manifest_path}. "
            f"Run `python scripts/build_partitions.py` first."
        )

    run = ComplianceRun()
    if not jsonschema_available():
        run.warn(
            "`jsonschema` is not installed — structural validation was skipped. "
            "Install it before treating this run as a sign-off."
        )

    manifest = check_manifest_schema(run, manifest_path)
    if manifest is not None:
        check_provenance(run, manifest, config.input.corpus_manifest_path)
        check_cluster_consumer(
            run,
            manifest,
            config.integration.expected_client_count,
            config.integration.max_client_count,
        )
        check_registry_reader(run, manifest)
        check_eval_result_shape(run, manifest)

    lines = [
        "",
        "CLASP-P5 · Interface Contract Compliance",
        "=" * 68,
        f"  manifest: {paths.relative(manifest_path)}",
        f"  verdict:  {'PASS' if run.ok else 'FAIL'}",
        "",
    ]
    for item in run.reports:
        lines.append(f"    {item.summary()}")
        for message in item.errors:
            lines.append(f"        FAIL  {message}")
    lines.append("")
    for note in run.notes:
        lines.append(f"    ok    {note}")
    for warning in run.warnings:
        lines.append(f"    warn  {warning}")
    for error in run.errors:
        lines.append(f"    FAIL  {error}")
    lines.append("")

    if not args.no_report:
        output = args.output or (paths.reports / "contract_compliance_report.md")
        path = render_report(run, manifest).write(output)
        lines.append(report_line(path))
        lines.append("")

    emit(lines)
    return EXIT_OK if run.ok else EXIT_FAILURE


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
