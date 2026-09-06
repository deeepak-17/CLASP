"""Partition validation — Week 2, Wednesday.

"Validate partitions are non-overlapping and balanced enough."

Two classes of check, kept deliberately distinct:

**Correctness (hard errors).** A violation makes the federation invalid or
falsifies CLASP's central isolation claim, so the run fails.

* *Disjointness* — no ``file_id`` appears in two shards.
* *Content disjointness* — no identical file content appears in two shards.
  Stronger than the above and the one that actually protects the Week-13
  claim "Project A's LoRA never reaches B": a file vendored into two projects
  would leak content across clusters even with distinct ``file_id`` values.
* *Coverage* — every corpus record is assigned exactly once. Silent record
  loss would quietly shrink the training set.
* *Project purity* — every file in a shard shares the shard's ``cluster_id``.
  This is what makes the partition non-IID *by project* rather than merely
  uneven.
* *Minimum shard size* — a near-empty Flower client stalls a FedProx round.

**Balance (warnings).** D1 is genuinely uneven, and that unevenness *is* the
non-IID signal FedProx exists to handle. Reporting it as a hard failure would
be measuring the corpus against the wrong standard, so imbalance ratio, Gini
and coefficient of variation are diagnostics — promotable to errors via
``validation.fail_on_warning`` for CI.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence

from corpus.models import CorpusRecord
from interfaces.contracts import PartitionManifest
from partitions.models import ValidationConfig
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Balance metrics
# ---------------------------------------------------------------------------
def gini_coefficient(values: Sequence[float]) -> float:
    """Gini coefficient of ``values`` — 0.0 is perfectly equal, 1.0 maximally unequal.

    Computed with the sorted-rank formulation::

        G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n

    Returns 0.0 for an empty or all-zero input, which is the correct degenerate
    answer (no inequality can be observed).
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    total = sum(ordered)
    if total <= 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return round((2.0 * weighted) / (n * total) - (n + 1) / n, 6)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Population standard deviation divided by the mean.

    A scale-free dispersion measure; unlike the max/min ratio it is not
    dominated by a single outlier shard.
    """
    if len(values) < 2:
        return 0.0
    numbers = [float(v) for v in values]
    mean = fmean(numbers)
    if mean == 0:
        return 0.0
    return round(pstdev(numbers) / mean, 6)


def imbalance_ratio(values: Sequence[float]) -> float:
    """Largest value divided by the smallest. ``inf`` if the smallest is zero."""
    if not values:
        return 0.0
    smallest = min(float(v) for v in values)
    largest = max(float(v) for v in values)
    if smallest <= 0:
        return float("inf")
    return round(largest / smallest, 6)


@dataclass(frozen=True)
class BalanceMetrics:
    """Distribution summary over shard sizes."""

    num_shards: int
    min_files: int
    max_files: int
    mean_files: float
    median_files: float
    imbalance_ratio: float
    gini: float
    coefficient_of_variation: float
    min_client: str
    max_client: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_shards": self.num_shards,
            "min_files": self.min_files,
            "max_files": self.max_files,
            "mean_files": self.mean_files,
            "median_files": self.median_files,
            "imbalance_ratio": self.imbalance_ratio,
            "gini": self.gini,
            "coefficient_of_variation": self.coefficient_of_variation,
            "min_client": self.min_client,
            "max_client": self.max_client,
        }


def compute_balance(manifest: PartitionManifest) -> BalanceMetrics:
    """Summarise how evenly files are spread across shards."""
    counts = [shard.num_files for shard in manifest.shards]
    ordered = sorted(counts)
    midpoint = len(ordered) // 2
    median = (
        float(ordered[midpoint])
        if len(ordered) % 2 == 1
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    smallest = min(manifest.shards, key=lambda s: s.num_files)
    largest = max(manifest.shards, key=lambda s: s.num_files)

    return BalanceMetrics(
        num_shards=len(counts),
        min_files=min(counts),
        max_files=max(counts),
        mean_files=round(fmean(counts), 4),
        median_files=median,
        imbalance_ratio=imbalance_ratio(counts),
        gini=gini_coefficient(counts),
        coefficient_of_variation=coefficient_of_variation(counts),
        min_client=smallest.client_id,
        max_client=largest.client_id,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class PartitionValidationResult:
    """Outcome of validating a partition set."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    balance: BalanceMetrics | None = None
    purity_by_cluster: dict[str, float] = field(default_factory=dict)
    overlap_examples: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"[{'PASS' if self.ok else 'FAIL'}] partition validation: "
            f"{len(self.checks_passed)} check(s) passed, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks_passed": list(self.checks_passed),
            "balance": self.balance.to_dict() if self.balance else None,
            "purity_by_cluster": dict(self.purity_by_cluster),
        }


class PartitionValidator:
    """Runs the Week-2 Wednesday acceptance checks over a partition set."""

    def __init__(self, config: ValidationConfig) -> None:
        self._config = config

    def validate(
        self,
        manifest: PartitionManifest,
        shard_records: dict[str, list[CorpusRecord]],
        *,
        corpus_records: Iterable[CorpusRecord] | None = None,
    ) -> PartitionValidationResult:
        """Validate a partition set.

        Args:
            manifest: The manifest under test.
            shard_records: Client id -> the records actually written to that
                shard. Read back from disk by the CLI, so the check covers
                what was *written*, not only what was computed in memory.
            corpus_records: The full source corpus, when available. Required
                for the coverage check; omitting it downgrades coverage to a
                reported skip.
        """
        result = PartitionValidationResult()

        self._check_manifest_consistency(manifest, shard_records, result)
        self._check_disjointness(shard_records, result)
        self._check_content_disjointness(shard_records, result)
        self._check_project_purity(manifest, shard_records, result)
        self._check_min_shard_size(manifest, result)
        self._check_coverage(shard_records, corpus_records, result)
        self._check_balance(manifest, result)

        if self._config.fail_on_warning and result.warnings:
            promoted = [f"(promoted from warning) {w}" for w in result.warnings]
            result.errors.extend(promoted)
            result.warnings.clear()

        _LOG.info(result.summary())
        for message in result.errors:
            _LOG.error("  %s", message)
        for message in result.warnings:
            _LOG.warning("  %s", message)
        return result

    # --- individual checks -------------------------------------------------
    def _check_manifest_consistency(
        self,
        manifest: PartitionManifest,
        shard_records: dict[str, list[CorpusRecord]],
        result: PartitionValidationResult,
    ) -> None:
        """The manifest's declared counts must match the shards on disk."""
        declared = {shard.client_id for shard in manifest.shards}
        actual = set(shard_records)

        if declared != actual:
            missing = declared - actual
            extra = actual - declared
            if missing:
                result.errors.append(f"Manifest declares shard(s) not present on disk: {sorted(missing)}")
            if extra:
                result.errors.append(f"Shard file(s) present but not declared in manifest: {sorted(extra)}")
            return

        for shard in manifest.shards:
            written = len(shard_records[shard.client_id])
            if written != shard.num_files:
                result.errors.append(
                    f"Shard '{shard.client_id}' declares {shard.num_files} file(s) "
                    f"but contains {written}"
                )
        if not result.errors:
            result.checks_passed.append("manifest counts match shards on disk")

    def _check_disjointness(
        self, shard_records: dict[str, list[CorpusRecord]], result: PartitionValidationResult
    ) -> None:
        """No ``file_id`` may appear in more than one shard."""
        if not self._config.require_disjoint_files:
            return
        owner: dict[str, str] = {}
        collisions: list[str] = []
        for client_id, records in sorted(shard_records.items()):
            for record in records:
                previous = owner.get(record.file_id)
                if previous is not None and previous != client_id:
                    collisions.append(f"{record.file_id} in both '{previous}' and '{client_id}'")
                else:
                    owner[record.file_id] = client_id

        if collisions:
            result.overlap_examples = collisions[:10]
            result.errors.append(
                f"Shards overlap on {len(collisions)} file_id(s); e.g. {collisions[0]}"
            )
        else:
            result.checks_passed.append(f"shards are disjoint by file_id ({len(owner)} unique files)")

    def _check_content_disjointness(
        self, shard_records: dict[str, list[CorpusRecord]], result: PartitionValidationResult
    ) -> None:
        """No identical file *content* may appear in more than one shard."""
        if not self._config.require_content_disjoint:
            return
        owner: dict[str, tuple[str, str]] = {}
        leaks: list[str] = []
        for client_id, records in sorted(shard_records.items()):
            for record in records:
                previous = owner.get(record.content_sha256)
                if previous is not None and previous[0] != client_id:
                    leaks.append(
                        f"{record.file_id} ('{client_id}') duplicates {previous[1]} ('{previous[0]}')"
                    )
                else:
                    owner[record.content_sha256] = (client_id, record.file_id)

        if leaks:
            result.errors.append(
                f"Identical content appears in {len(leaks)} cross-shard pair(s), which would "
                f"leak code between clusters; e.g. {leaks[0]}"
            )
        else:
            result.checks_passed.append("no identical content shared across shards")

    def _check_project_purity(
        self,
        manifest: PartitionManifest,
        shard_records: dict[str, list[CorpusRecord]],
        result: PartitionValidationResult,
    ) -> None:
        """Every file in a shard must belong to that shard's cluster."""
        if not self._config.require_project_purity:
            return

        impure: list[str] = []
        per_cluster: dict[str, list[float]] = defaultdict(list)

        for shard in manifest.shards:
            records = shard_records.get(shard.client_id, [])
            if not records:
                continue
            matching = sum(1 for r in records if r.cluster_id == shard.cluster_id)
            purity = matching / len(records)
            per_cluster[shard.cluster_id].append(purity)
            if purity < 1.0:
                foreign = sorted({r.cluster_id for r in records if r.cluster_id != shard.cluster_id})
                impure.append(
                    f"'{shard.client_id}' purity={purity:.4f}, contains foreign cluster(s): {foreign}"
                )

        result.purity_by_cluster = {
            cluster_id: round(fmean(values), 6) for cluster_id, values in sorted(per_cluster.items())
        }

        if impure:
            result.errors.append(
                f"Partition is not non-IID by project — {len(impure)} impure shard(s): {impure[0]}"
            )
        else:
            result.checks_passed.append("project purity is 1.0 for every shard (non-IID by project)")

    def _check_min_shard_size(
        self, manifest: PartitionManifest, result: PartitionValidationResult
    ) -> None:
        """No shard may be smaller than the configured floor."""
        undersized = [
            f"'{shard.client_id}' has {shard.num_files} file(s)"
            for shard in manifest.shards
            if shard.num_files < self._config.min_files_per_shard
        ]
        if undersized:
            result.errors.append(
                f"Shard(s) below the {self._config.min_files_per_shard}-file minimum "
                f"(a near-empty Flower client stalls a FedProx round): {'; '.join(undersized)}"
            )
        else:
            result.checks_passed.append(
                f"every shard has >= {self._config.min_files_per_shard} file(s)"
            )

    def _check_coverage(
        self,
        shard_records: dict[str, list[CorpusRecord]],
        corpus_records: Iterable[CorpusRecord] | None,
        result: PartitionValidationResult,
    ) -> None:
        """Every corpus record must be assigned exactly once."""
        if not self._config.require_full_coverage:
            return
        if corpus_records is None:
            result.warnings.append("coverage check skipped: source corpus not supplied")
            return

        corpus_ids = {record.file_id for record in corpus_records}
        assigned_ids = {record.file_id for records in shard_records.values() for record in records}

        unassigned = corpus_ids - assigned_ids
        unknown = assigned_ids - corpus_ids

        if unassigned:
            sample = sorted(unassigned)[:3]
            result.errors.append(
                f"{len(unassigned)} corpus file(s) were never assigned to a shard; e.g. {sample}"
            )
        if unknown:
            sample = sorted(unknown)[:3]
            result.errors.append(
                f"{len(unknown)} shard file(s) do not exist in the corpus; e.g. {sample}"
            )
        if not unassigned and not unknown:
            result.checks_passed.append(f"full coverage: all {len(corpus_ids)} corpus file(s) assigned once")

    def _check_balance(
        self, manifest: PartitionManifest, result: PartitionValidationResult
    ) -> None:
        """Report distribution metrics; breaches are warnings by default."""
        balance = compute_balance(manifest)
        result.balance = balance

        if balance.imbalance_ratio > self._config.max_imbalance_ratio:
            result.warnings.append(
                f"imbalance ratio {balance.imbalance_ratio:.2f} exceeds the "
                f"{self._config.max_imbalance_ratio:.2f} threshold "
                f"('{balance.max_client}' has {balance.max_files} files vs "
                f"'{balance.min_client}' with {balance.min_files})"
            )
        else:
            result.checks_passed.append(
                f"imbalance ratio {balance.imbalance_ratio:.2f} within {self._config.max_imbalance_ratio:.2f}"
            )

        if balance.gini > self._config.max_gini:
            result.warnings.append(
                f"Gini coefficient {balance.gini:.4f} exceeds the {self._config.max_gini:.2f} threshold"
            )
        else:
            result.checks_passed.append(f"Gini {balance.gini:.4f} within {self._config.max_gini:.2f}")

        if balance.coefficient_of_variation > self._config.max_coefficient_of_variation:
            result.warnings.append(
                f"coefficient of variation {balance.coefficient_of_variation:.4f} exceeds the "
                f"{self._config.max_coefficient_of_variation:.2f} threshold"
            )
        else:
            result.checks_passed.append(
                f"coefficient of variation {balance.coefficient_of_variation:.4f} within "
                f"{self._config.max_coefficient_of_variation:.2f}"
            )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_validation_report(
    manifest: PartitionManifest,
    result: PartitionValidationResult,
    config: ValidationConfig,
) -> MarkdownReport:
    """Render the Week-2 Wednesday validation outcome as Markdown."""
    report = MarkdownReport(
        title="CLASP-P5 · Partition Validation Report",
        subtitle="Week 2 · Wednesday deliverable — non-overlap and balance",
    )

    report.heading("1. Verdict")
    report.status_line(
        result.ok,
        f"{len(result.checks_passed)} check(s) passed, {len(result.errors)} error(s), "
        f"{len(result.warnings)} warning(s)",
    )
    report.key_values(
        {
            "Strategy": manifest.strategy.value,
            "Seed": manifest.seed,
            "Clients": manifest.num_clients,
            "Clusters": manifest.num_clusters,
            "Total files": manifest.total_files,
            "Corpus": f"`{manifest.corpus_path}`",
            "Corpus SHA-256": f"`{manifest.corpus_sha256[:16]}…`",
        }
    )

    report.heading("2. Correctness checks")
    report.paragraph(
        "Hard constraints. A failure here invalidates the federation or falsifies CLASP's "
        "project-isolation claim."
    )
    if result.errors:
        report.heading("Errors", level=3)
        report.bullets(result.errors)
    if result.checks_passed:
        report.heading("Passed", level=3)
        report.bullets(result.checks_passed)
    if result.overlap_examples:
        report.heading("Overlap examples", level=3)
        report.bullets(result.overlap_examples)

    report.heading("3. Balance")
    report.paragraph(
        "Reported as diagnostics rather than failures: D1's unevenness is the non-IID "
        "client-drift condition FedProx's proximal term is designed to absorb. Set "
        "`validation.fail_on_warning: true` to enforce these thresholds in CI."
    )
    if result.balance:
        balance = result.balance
        report.table(
            ["Metric", "Value", "Threshold", "Within"],
            [
                [
                    "Imbalance ratio (max/min files)",
                    f"{balance.imbalance_ratio:.3f}",
                    f"{config.max_imbalance_ratio:.2f}",
                    balance.imbalance_ratio <= config.max_imbalance_ratio,
                ],
                ["Gini coefficient", f"{balance.gini:.4f}", f"{config.max_gini:.2f}", balance.gini <= config.max_gini],
                [
                    "Coefficient of variation",
                    f"{balance.coefficient_of_variation:.4f}",
                    f"{config.max_coefficient_of_variation:.2f}",
                    balance.coefficient_of_variation <= config.max_coefficient_of_variation,
                ],
                ["Smallest shard", f"{balance.min_client} ({balance.min_files} files)", "—", None],
                ["Largest shard", f"{balance.max_client} ({balance.max_files} files)", "—", None],
                ["Mean files/shard", f"{balance.mean_files:.2f}", "—", None],
                ["Median files/shard", f"{balance.median_files:.1f}", "—", None],
            ],
        )

    report.heading("4. Per-shard breakdown")
    report.table(
        ["Client", "Cluster", "Project", "Files", "Lines", "Code lines", "Bytes", "Purity"],
        [
            [
                shard.client_id,
                shard.cluster_id,
                shard.project_label,
                shard.num_files,
                shard.num_lines,
                shard.num_code_lines,
                shard.total_bytes,
                f"{result.purity_by_cluster.get(shard.cluster_id, float('nan')):.4f}"
                if shard.cluster_id in result.purity_by_cluster
                else "—",
            ]
            for shard in manifest.shards
        ],
    )

    if result.warnings:
        report.heading("5. Warnings")
        report.bullets(result.warnings)

    report.rule()
    report.paragraph(
        "Generated by `partitions/validation.py`. Regenerate with "
        "`python scripts/validate_partitions.py`."
    )
    return report


def default_report_path() -> Path:
    """Canonical location of the generated validation report."""
    return project_paths().reports / "partition_validation_report.md"
