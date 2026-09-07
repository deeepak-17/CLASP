"""Tests for the Week-2 partitioning, validation and metadata pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from corpus.models import CorpusRecord
from interfaces.contracts import PartitionStrategyName
from partitions.metadata import build_partition_metadata, render_metadata_report
from partitions.models import (
    ClientNamingConfig,
    PartitionConfig,
    PartitionOutputConfig,
    ValidationConfig,
)
from partitions.partitioner import Partitioner, load_partition_config, load_partition_manifest
from partitions.strategies import (
    PerDeveloperStrategy,
    ProjectLevelStrategy,
    build_strategy,
)
from partitions.validation import (
    PartitionValidator,
    coefficient_of_variation,
    compute_balance,
    gini_coefficient,
    imbalance_ratio,
    render_validation_report,
)
from tests.conftest import make_record
from utils.errors import ConfigError, PartitionError


# ---------------------------------------------------------------------------
# Strategies (Week 2 · Monday)
# ---------------------------------------------------------------------------
class TestProjectLevelStrategy:
    @pytest.fixture
    def strategy(self) -> ProjectLevelStrategy:
        return ProjectLevelStrategy(ClientNamingConfig(template="client-{cluster_id}"), seed=1)

    def test_one_shard_per_cluster(
        self, strategy: ProjectLevelStrategy, sample_records: list[CorpusRecord]
    ) -> None:
        assignments = strategy.assign(sample_records)
        assert len(assignments) == 3
        assert [a.cluster_id for a in assignments] == ["alpha", "beta", "gamma"]

    def test_client_ids_follow_the_template(
        self, strategy: ProjectLevelStrategy, sample_records: list[CorpusRecord]
    ) -> None:
        assignments = strategy.assign(sample_records)
        assert [a.client_id for a in assignments] == ["client-alpha", "client-beta", "client-gamma"]

    def test_every_record_is_assigned_exactly_once(
        self, strategy: ProjectLevelStrategy, sample_records: list[CorpusRecord]
    ) -> None:
        assigned = [r.file_id for a in strategy.assign(sample_records) for r in a.records]
        assert sorted(assigned) == sorted(r.file_id for r in sample_records)
        assert len(assigned) == len(set(assigned))

    def test_shards_are_project_pure(
        self, strategy: ProjectLevelStrategy, sample_records: list[CorpusRecord]
    ) -> None:
        for assignment in strategy.assign(sample_records):
            assert {r.cluster_id for r in assignment.records} == {assignment.cluster_id}

    def test_intra_shard_order_is_canonical(
        self, strategy: ProjectLevelStrategy, sample_records: list[CorpusRecord]
    ) -> None:
        for assignment in strategy.assign(sample_records):
            ids = [r.file_id for r in assignment.records]
            assert ids == sorted(ids)

    def test_is_deterministic_under_input_reordering(
        self, strategy: ProjectLevelStrategy, sample_records: list[CorpusRecord]
    ) -> None:
        forward = strategy.assign(sample_records)
        reversed_ = strategy.assign(list(reversed(sample_records)))
        assert [a.client_id for a in forward] == [a.client_id for a in reversed_]
        assert [[r.file_id for r in a.records] for a in forward] == [
            [r.file_id for r in a.records] for a in reversed_
        ]

    def test_aggregates_are_correct(self, strategy: ProjectLevelStrategy) -> None:
        records = [make_record("alpha", i, code_lines=10) for i in range(3)]
        assignment = strategy.assign(records)[0]
        assert assignment.num_files == 3
        assert assignment.num_code_lines == 30

    def test_empty_corpus_raises(self, strategy: ProjectLevelStrategy) -> None:
        with pytest.raises(PartitionError, match="empty corpus"):
            strategy.assign([])

    def test_conflicting_project_labels_raise(self, strategy: ProjectLevelStrategy) -> None:
        records = [
            make_record("alpha", 0, project_label="Alpha"),
            make_record("alpha", 1, project_label="Different"),
        ]
        with pytest.raises(PartitionError, match="conflicting project labels"):
            strategy.assign(records)


class TestPerDeveloperStrategy:
    """Week-4 Monday: per-developer strategy is now implemented. See
    tests/test_partitions_per_developer.py for the full behavioural suite
    (module-path grouping, hash-chunk fallback, leakage, determinism); these
    two tests just replace the old "raises Week-4-not-implemented" checks
    that were correct for Week 2 and are no longer correct now.
    """

    def test_is_now_implemented(self, sample_records: list[CorpusRecord]) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1)
        assignments = strategy.assign(sample_records)
        assert assignments  # no longer raises

    def test_registry_returns_the_right_strategy(self) -> None:
        built = build_strategy(PartitionStrategyName.PROJECT_LEVEL, ClientNamingConfig(), 1)
        assert isinstance(built, ProjectLevelStrategy)

    def test_registry_returns_per_developer_strategy(self) -> None:
        built = build_strategy(PartitionStrategyName.PER_DEVELOPER, ClientNamingConfig(), 1)
        assert isinstance(built, PerDeveloperStrategy)


# ---------------------------------------------------------------------------
# Balance metrics
# ---------------------------------------------------------------------------
class TestBalanceMetrics:
    def test_gini_of_equal_values_is_zero(self) -> None:
        assert gini_coefficient([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)

    def test_gini_increases_with_inequality(self) -> None:
        assert gini_coefficient([1, 1, 100]) > gini_coefficient([1, 1, 3])

    def test_gini_handles_degenerate_input(self) -> None:
        assert gini_coefficient([]) == 0.0
        assert gini_coefficient([0, 0]) == 0.0

    def test_gini_is_bounded(self) -> None:
        assert 0.0 <= gini_coefficient([1, 1, 1, 1_000_000]) <= 1.0

    def test_coefficient_of_variation(self) -> None:
        assert coefficient_of_variation([5, 5, 5]) == pytest.approx(0.0)
        assert coefficient_of_variation([1]) == 0.0
        assert coefficient_of_variation([1, 9]) > 0.5

    def test_imbalance_ratio(self) -> None:
        assert imbalance_ratio([2, 10]) == pytest.approx(5.0)
        assert imbalance_ratio([0, 5]) == float("inf")
        assert imbalance_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# Partitioner (Week 2 · Tuesday)
# ---------------------------------------------------------------------------
class TestPartitioner:
    def test_repository_config_loads(self) -> None:
        config = load_partition_config()
        assert config.strategy_name is PartitionStrategyName.PROJECT_LEVEL
        assert config.seed >= 0

    def test_writes_shards_and_manifest(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        assert result.num_clients == 3
        assert result.manifest_path.is_file()
        for path in result.shard_paths.values():
            assert path.is_file()

    def test_manifest_totals_are_consistent(self, partition_config: PartitionConfig) -> None:
        manifest = Partitioner(partition_config).run().manifest
        assert manifest.total_files == sum(s.num_files for s in manifest.shards)
        assert manifest.num_clients == len(manifest.shards)
        assert manifest.num_clusters == len(set(manifest.cluster_ids))

    def test_manifest_round_trips_through_disk(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        reloaded = load_partition_manifest(result.manifest_path)
        assert reloaded.to_dict() == result.manifest.to_dict()

    def test_shard_digests_are_stable_across_runs(self, partition_config: PartitionConfig) -> None:
        first = Partitioner(partition_config).run().manifest
        second = Partitioner(partition_config).run().manifest
        assert [s.content_sha256 for s in first.shards] == [s.content_sha256 for s in second.shards]

    def test_shard_digests_differ_between_shards(self, partition_config: PartitionConfig) -> None:
        manifest = Partitioner(partition_config).run().manifest
        digests = [s.content_sha256 for s in manifest.shards]
        assert len(set(digests)) == len(digests)

    def test_dry_run_writes_nothing(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run(dry_run=True)
        assert result.num_clients == 3
        assert not result.manifest_path.exists()
        assert not any(path.exists() for path in result.shard_paths.values())

    def test_shard_lookup_by_client_id(self, partition_config: PartitionConfig) -> None:
        manifest = Partitioner(partition_config).run().manifest
        assert manifest.shard_for("client-alpha").cluster_id == "alpha"

    def test_shard_lookup_rejects_unknown_client(self, partition_config: PartitionConfig) -> None:
        from utils.errors import ContractViolationError

        manifest = Partitioner(partition_config).run().manifest
        with pytest.raises(ContractViolationError, match="No shard"):
            manifest.shard_for("client-nope")

    def test_per_developer_strategy_now_runs(self, partition_config: PartitionConfig) -> None:
        # partition_config's tmp_path is fresh per test, so reusing the same
        # output paths as the project-level tests in this class is safe.
        config = replace(partition_config, strategy="per_developer")
        result = Partitioner(config).run()
        assert result.num_clients >= 3  # at least one developer shard per of the 3 sample clusters
        assert all(a.developer_basis is not None for a in result.assignments)

    def test_unknown_strategy_is_rejected(self, partition_config: PartitionConfig) -> None:
        with pytest.raises(ConfigError, match="Unknown partition.strategy"):
            replace(partition_config, strategy="random_shuffle")

    def test_shard_template_must_contain_client_id(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\{client_id\}"):
            PartitionOutputConfig(
                partitions_dir=tmp_path,
                manifest_path=tmp_path / "m.json",
                shard_filename_template="shard.jsonl",
            )


# ---------------------------------------------------------------------------
# Validation (Week 2 · Wednesday)
# ---------------------------------------------------------------------------
def _load_shards(result) -> dict[str, list[CorpusRecord]]:
    from utils.io_utils import read_jsonl

    return {
        client_id: [CorpusRecord.from_dict(row) for row in read_jsonl(path)]
        for client_id, path in result.shard_paths.items()
    }


class TestPartitionValidator:
    def test_clean_partition_passes(
        self, partition_config: PartitionConfig, sample_records: list[CorpusRecord]
    ) -> None:
        result = Partitioner(partition_config).run()
        outcome = PartitionValidator(partition_config.validation).validate(
            result.manifest, _load_shards(result), corpus_records=sample_records
        )
        assert outcome.ok, outcome.errors
        assert outcome.balance is not None

    def test_detects_file_id_overlap(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        shards = _load_shards(result)
        # Inject a file from alpha into beta's shard.
        shards["client-beta"].append(shards["client-alpha"][0])

        outcome = PartitionValidator(partition_config.validation).validate(result.manifest, shards)
        assert not outcome.ok
        assert any("overlap" in error for error in outcome.errors)

    def test_detects_content_leak_across_shards(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        shards = _load_shards(result)
        donor = shards["client-alpha"][0]
        # Same content, different file_id and cluster: a vendored-file leak.
        shards["client-beta"].append(
            replace(donor, file_id="beta:src/beta/copy.py", cluster_id="beta")
        )

        outcome = PartitionValidator(partition_config.validation).validate(result.manifest, shards)
        assert not outcome.ok
        assert any("leak code between clusters" in error for error in outcome.errors)

    def test_detects_impure_shard(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        shards = _load_shards(result)
        foreign = make_record("gamma", 99)
        shards["client-alpha"].append(foreign)

        outcome = PartitionValidator(partition_config.validation).validate(result.manifest, shards)
        assert not outcome.ok
        assert any("non-IID by project" in error for error in outcome.errors)

    def test_detects_unassigned_corpus_files(
        self, partition_config: PartitionConfig, sample_records: list[CorpusRecord]
    ) -> None:
        result = Partitioner(partition_config).run()
        extra = sample_records + [make_record("delta", 0)]

        outcome = PartitionValidator(partition_config.validation).validate(
            result.manifest, _load_shards(result), corpus_records=extra
        )
        assert not outcome.ok
        assert any("never assigned" in error for error in outcome.errors)

    def test_detects_manifest_count_drift(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        shards = _load_shards(result)
        shards["client-alpha"].pop()

        outcome = PartitionValidator(partition_config.validation).validate(result.manifest, shards)
        assert not outcome.ok
        assert any("but contains" in error for error in outcome.errors)

    def test_detects_undersized_shard(self, partition_config: PartitionConfig) -> None:
        config = replace(partition_config, validation=ValidationConfig(min_files_per_shard=99))
        result = Partitioner(config).run()

        outcome = PartitionValidator(config.validation).validate(result.manifest, _load_shards(result))
        assert not outcome.ok
        assert any("minimum" in error for error in outcome.errors)

    def test_imbalance_is_a_warning_not_an_error(self, partition_config: PartitionConfig) -> None:
        config = replace(
            partition_config, validation=ValidationConfig(min_files_per_shard=1, max_imbalance_ratio=1.0)
        )
        result = Partitioner(config).run()
        outcome = PartitionValidator(config.validation).validate(result.manifest, _load_shards(result))
        assert outcome.ok
        assert any("imbalance ratio" in warning for warning in outcome.warnings)

    def test_fail_on_warning_promotes_warnings(self, partition_config: PartitionConfig) -> None:
        config = replace(
            partition_config,
            validation=ValidationConfig(
                min_files_per_shard=1, max_imbalance_ratio=1.0, fail_on_warning=True
            ),
        )
        result = Partitioner(config).run()
        outcome = PartitionValidator(config.validation).validate(result.manifest, _load_shards(result))
        assert not outcome.ok
        assert any("promoted from warning" in error for error in outcome.errors)
        assert not outcome.warnings

    def test_missing_corpus_downgrades_coverage_to_a_warning(
        self, partition_config: PartitionConfig
    ) -> None:
        result = Partitioner(partition_config).run()
        outcome = PartitionValidator(partition_config.validation).validate(
            result.manifest, _load_shards(result), corpus_records=None
        )
        assert outcome.ok
        assert any("coverage check skipped" in warning for warning in outcome.warnings)

    def test_balance_metrics_are_computed(self, partition_config: PartitionConfig) -> None:
        manifest = Partitioner(partition_config).run().manifest
        balance = compute_balance(manifest)
        assert balance.num_shards == 3
        assert balance.max_files == 6
        assert balance.min_files == 3
        assert balance.max_client == "client-alpha"
        assert balance.min_client == "client-gamma"

    def test_validation_report_renders(
        self, partition_config: PartitionConfig, sample_records: list[CorpusRecord]
    ) -> None:
        result = Partitioner(partition_config).run()
        outcome = PartitionValidator(partition_config.validation).validate(
            result.manifest, _load_shards(result), corpus_records=sample_records
        )
        text = render_validation_report(result.manifest, outcome, partition_config.validation).render()
        assert "Partition Validation Report" in text
        assert "**PASS**" in text
        assert "client-alpha" in text


# ---------------------------------------------------------------------------
# Metadata (Week 2 · Thursday)
# ---------------------------------------------------------------------------
class TestPartitionMetadata:
    def test_metadata_describes_every_cluster(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        metadata = build_partition_metadata(
            result.manifest, _load_shards(result), manifest_path=result.manifest_path
        )
        assert metadata.num_clusters == 3
        assert {c.cluster_id for c in metadata.clusters} == {"alpha", "beta", "gamma"}
        assert all(c.project_label for c in metadata.clusters)

    def test_shares_sum_to_one(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        metadata = build_partition_metadata(
            result.manifest, _load_shards(result), manifest_path=result.manifest_path
        )
        assert sum(c.share_of_corpus_files for c in metadata.clusters) == pytest.approx(1.0, abs=1e-6)
        assert sum(c.share_of_corpus_code_lines for c in metadata.clusters) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_licence_roll_up_counts_every_file(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        metadata = build_partition_metadata(
            result.manifest, _load_shards(result), manifest_path=result.manifest_path
        )
        assert sum(metadata.license_summary.values()) == metadata.total_files

    def test_metadata_is_json_serialisable(self, partition_config: PartitionConfig, tmp_path: Path) -> None:
        from partitions.metadata import write_partition_metadata
        from utils.io_utils import read_json

        result = Partitioner(partition_config).run()
        metadata = build_partition_metadata(
            result.manifest, _load_shards(result), manifest_path=result.manifest_path
        )
        path = write_partition_metadata(metadata, tmp_path / "metadata.json")
        assert read_json(path)["num_clusters"] == 3

    def test_metadata_report_renders(self, partition_config: PartitionConfig) -> None:
        result = Partitioner(partition_config).run()
        metadata = build_partition_metadata(
            result.manifest, _load_shards(result), manifest_path=result.manifest_path
        )
        text = render_metadata_report(metadata).render()
        assert "Partition Metadata" in text
        assert "Licence roll-up" in text
