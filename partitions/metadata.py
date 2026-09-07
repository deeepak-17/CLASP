"""Partition metadata — Week 2, Thursday.

"Write partition metadata (sizes, project labels) to disk."

``manifest.json`` (written by :mod:`partitions.partitioner`) is the *contract*
artefact: minimal, schema-validated, and consumed by P1/P2/P4. This module
writes the *descriptive* artefact alongside it —
``datasets/partitions/partition_metadata.json`` — carrying the derived
statistics teammates and the dashboard want but which do not belong in a
frozen interface: size distributions, per-cluster aggregates, licence
roll-ups and balance metrics.

Splitting them matters. The contract document can then stay stable while the
descriptive one grows freely, so adding a statistic never forces a contract
version bump on four other people.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from corpus.models import CorpusRecord
from interfaces.contracts import PartitionManifest
from partitions.validation import BalanceMetrics, compute_balance
from utils.io_utils import write_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.timing import utc_timestamp

_LOG = get_logger(__name__)

#: Version of the descriptive metadata document (independent of the contract).
METADATA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ClusterMetadata:
    """Descriptive statistics for one cluster (== one project)."""

    cluster_id: str
    project_label: str
    client_ids: list[str]
    num_files: int
    num_lines: int
    num_code_lines: int
    total_bytes: int
    mean_code_lines_per_file: float
    median_code_lines_per_file: float
    max_code_lines_in_file: int
    licenses: list[str]
    source_refs: list[str]
    share_of_corpus_files: float
    share_of_corpus_code_lines: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartitionMetadata:
    """Full descriptive metadata for a partition set."""

    metadata_version: str
    created_at: str
    strategy: str
    seed: int
    manifest_path: str
    corpus_path: str
    corpus_sha256: str
    num_clients: int
    num_clusters: int
    total_files: int
    total_lines: int
    total_code_lines: int
    total_bytes: int
    balance: dict[str, Any]
    clusters: list[ClusterMetadata] = field(default_factory=list)
    license_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clusters"] = [c.to_dict() for c in self.clusters]
        return payload


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_partition_metadata(
    manifest: PartitionManifest,
    shard_records: dict[str, list[CorpusRecord]],
    *,
    manifest_path: Path | str,
) -> PartitionMetadata:
    """Derive descriptive metadata from a manifest and its shard contents.

    Args:
        manifest: The partition manifest.
        shard_records: Client id -> records written to that shard.
        manifest_path: Path of ``manifest.json``, recorded as a back-reference.
    """
    paths = project_paths()
    balance: BalanceMetrics = compute_balance(manifest)

    total_files = sum(shard.num_files for shard in manifest.shards)
    total_code_lines = sum(shard.num_code_lines for shard in manifest.shards)

    # Group shards by cluster. Week 2 has one client per cluster, but grouping
    # now means Week 8's multi-client clusters need no change here.
    by_cluster: dict[str, list[str]] = {}
    for shard in manifest.shards:
        by_cluster.setdefault(shard.cluster_id, []).append(shard.client_id)

    clusters: list[ClusterMetadata] = []
    license_counter: Counter[str] = Counter()

    for cluster_id in sorted(by_cluster):
        client_ids = sorted(by_cluster[cluster_id])
        records = [record for cid in client_ids for record in shard_records.get(cid, [])]
        if not records:
            _LOG.warning("Cluster '%s' has no readable shard records; skipping metadata", cluster_id)
            continue

        code_lines = [record.num_code_lines for record in records]
        cluster_code_lines = sum(code_lines)
        licenses = sorted({record.license for record in records})
        for record in records:
            license_counter[record.license] += 1

        clusters.append(
            ClusterMetadata(
                cluster_id=cluster_id,
                project_label=records[0].project_label,
                client_ids=client_ids,
                num_files=len(records),
                num_lines=sum(r.num_lines for r in records),
                num_code_lines=cluster_code_lines,
                total_bytes=sum(r.num_bytes for r in records),
                mean_code_lines_per_file=round(cluster_code_lines / len(records), 3),
                median_code_lines_per_file=_median(code_lines),
                max_code_lines_in_file=max(code_lines),
                licenses=licenses,
                source_refs=sorted({r.source_ref for r in records if r.source_ref}),
                share_of_corpus_files=round(len(records) / total_files, 6) if total_files else 0.0,
                share_of_corpus_code_lines=(
                    round(cluster_code_lines / total_code_lines, 6) if total_code_lines else 0.0
                ),
            )
        )

    return PartitionMetadata(
        metadata_version=METADATA_VERSION,
        created_at=utc_timestamp(),
        strategy=manifest.strategy.value,
        seed=manifest.seed,
        manifest_path=paths.relative(manifest_path),
        corpus_path=manifest.corpus_path,
        corpus_sha256=manifest.corpus_sha256,
        num_clients=manifest.num_clients,
        num_clusters=manifest.num_clusters,
        total_files=total_files,
        total_lines=sum(shard.num_lines for shard in manifest.shards),
        total_code_lines=total_code_lines,
        total_bytes=sum(shard.total_bytes for shard in manifest.shards),
        balance=balance.to_dict(),
        clusters=clusters,
        license_summary=dict(sorted(license_counter.items())),
    )


def write_partition_metadata(metadata: PartitionMetadata, path: Path | str | None = None) -> Path:
    """Write descriptive metadata to disk and return the path used."""
    target = Path(path) if path else default_metadata_path()
    write_json(target, metadata.to_dict())
    _LOG.info("Wrote partition metadata -> %s", project_paths().relative(target))
    return target


def default_metadata_path() -> Path:
    """Canonical location of ``partition_metadata.json``."""
    return project_paths().partitions / "partition_metadata.json"


def render_metadata_report(metadata: PartitionMetadata) -> MarkdownReport:
    """Render partition metadata as a human-readable Markdown summary."""
    report = MarkdownReport(
        title="CLASP-P5 · Partition Metadata",
        subtitle="sizes, project labels and provenance",
    )

    report.heading("1. Partition summary")
    report.key_values(
        {
            "Strategy": metadata.strategy,
            "Seed": metadata.seed,
            "Federated clients": metadata.num_clients,
            "Clusters (projects)": metadata.num_clusters,
            "Total files": metadata.total_files,
            "Total lines": metadata.total_lines,
            "Total code lines": metadata.total_code_lines,
            "Total bytes": metadata.total_bytes,
            "Corpus": f"`{metadata.corpus_path}`",
            "Corpus SHA-256": f"`{metadata.corpus_sha256[:16]}…`",
            "Manifest": f"`{metadata.manifest_path}`",
        }
    )

    report.heading("2. Per-cluster sizes and labels")
    report.table(
        [
            "Cluster",
            "Project label",
            "Client(s)",
            "Files",
            "Code lines",
            "Bytes",
            "Mean SLOC/file",
            "Median SLOC/file",
            "% of files",
            "% of code",
        ],
        [
            [
                cluster.cluster_id,
                cluster.project_label,
                ", ".join(cluster.client_ids),
                cluster.num_files,
                cluster.num_code_lines,
                cluster.total_bytes,
                f"{cluster.mean_code_lines_per_file:.1f}",
                f"{cluster.median_code_lines_per_file:.1f}",
                f"{100 * cluster.share_of_corpus_files:.1f}%",
                f"{100 * cluster.share_of_corpus_code_lines:.1f}%",
            ]
            for cluster in metadata.clusters
        ],
    )

    report.heading("3. Distribution")
    report.key_values({k: v for k, v in metadata.balance.items()})

    report.heading("4. Licence roll-up")
    report.paragraph(
        "Every file carries the licence of its source repository. All entries must be "
        "permissive for CLASP's on-premise enterprise framing to hold."
    )
    report.table(
        ["Licence", "Files"],
        [[name, count] for name, count in metadata.license_summary.items()],
    )

    report.heading("5. Provenance")
    report.table(
        ["Cluster", "Source ref(s)"],
        [[c.cluster_id, ", ".join(c.source_refs) or "—"] for c in metadata.clusters],
    )

    report.rule()
    report.paragraph(
        "Generated by `partitions/metadata.py`. `manifest.json` is the frozen contract "
        "artefact consumed by P1/P2/P4; this document is descriptive and may grow freely."
    )
    return report
