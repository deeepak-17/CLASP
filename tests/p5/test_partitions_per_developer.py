"""Tests for the Week-4 per-developer (individual-style) partitioning.

Kept in its own file (rather than folded into tests/test_partitions.py)
because it needs its own fixture corpus with real subdirectory structure to
exercise the module-path basis — the shared ``sample_records`` fixture in
conftest.py is deliberately flat (see its docstring intent: exercising
balance metrics), which is exactly the shape that should fall back to
hash-chunking, not the shape that exercises module-path grouping.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from corpus.models import CorpusRecord
from interfaces.contracts import PartitionStrategyName
from partitions.models import ClientNamingConfig, PartitionConfig, PerDeveloperConfig
from partitions.partitioner import Partitioner
from partitions.strategies import PerDeveloperStrategy, ProjectLevelStrategy, build_strategy
from partitions.validation import PartitionValidator
from utils.errors import ConfigError, PartitionError
from utils.io_utils import sha256_text


def make_nested_record(cluster_id: str, relative_path: str, *, code_lines: int = 20) -> CorpusRecord:
    """A corpus record at an explicit (possibly nested) path, for module-path tests."""
    body = f"# {cluster_id} {relative_path}\n" * code_lines
    return CorpusRecord(
        file_id=f"{cluster_id}:{relative_path}",
        cluster_id=cluster_id,
        project_label=cluster_id.title(),
        relative_path=relative_path,
        language="python",
        num_lines=body.count("\n"),
        num_code_lines=code_lines,
        num_bytes=len(body.encode("utf-8")),
        content_sha256=sha256_text(body),
        license="BSD-3-Clause",
        source_url=f"https://example.invalid/{cluster_id}.git",
        source_ref="deadbeef",
        content=body,
    )


@pytest.fixture
def structured_records() -> list[CorpusRecord]:
    """One cluster with real subpackages (module-path should apply), one flat
    (hash-chunk should apply), one tiny (single-developer fallback should apply).
    """
    records: list[CorpusRecord] = []
    # "modular": 3 subpackages of 3 files each -> module_path basis.
    for pkg in ("routing", "debug", "http"):
        for i in range(3):
            records.append(make_nested_record("modular", f"src/modular/{pkg}/mod_{i}.py"))
    # "flat": 12 files, no subdirectories -> deterministic_hash_chunk basis.
    for i in range(12):
        records.append(make_nested_record("flat", f"src/flat/mod_{i:02d}.py"))
    # "tiny": 3 files, too small to split -> single_developer_whole_project basis.
    for i in range(3):
        records.append(make_nested_record("tiny", f"src/tiny/mod_{i}.py"))
    return records


@pytest.fixture
def per_dev_config() -> PerDeveloperConfig:
    return PerDeveloperConfig(target_files_per_developer=4, min_files_per_developer=2)


class TestModulePathBasis:
    def test_modular_cluster_splits_by_subpackage(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config)
        assignments = strategy.assign(structured_records)
        modular = [a for a in assignments if a.cluster_id == "modular"]
        assert len(modular) == 3
        assert all(a.developer_basis == "module_path" for a in modular)
        assert {a.developer_label for a in modular} == {
            "src/modular/routing",
            "src/modular/debug",
            "src/modular/http",
        }

    def test_module_path_groups_stay_within_their_subpackage(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config)
        for assignment in strategy.assign(structured_records):
            if assignment.cluster_id == "modular":
                paths = {r.relative_path.rsplit("/", 1)[0] for r in assignment.records}
                assert len(paths) == 1  # every file in this shard shares one subpackage dir


class TestHashChunkFallback:
    def test_flat_cluster_falls_back_to_hash_chunk(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config)
        flat = [a for a in strategy.assign(structured_records) if a.cluster_id == "flat"]
        assert len(flat) == 3  # 12 files / target 4 per dev = 3 chunks
        assert all(a.developer_basis == "deterministic_hash_chunk" for a in flat)

    def test_hash_chunk_is_deterministic_for_a_fixed_seed(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        strategy_a = PerDeveloperStrategy(ClientNamingConfig(), seed=42, per_developer=per_dev_config)
        strategy_b = PerDeveloperStrategy(ClientNamingConfig(), seed=42, per_developer=per_dev_config)
        first = [tuple(r.file_id for r in a.records) for a in strategy_a.assign(structured_records)]
        second = [tuple(r.file_id for r in a.records) for a in strategy_b.assign(structured_records)]
        assert first == second

    def test_hash_chunk_differs_across_seeds(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        by_seed = {}
        for seed in (1, 2):
            strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=seed, per_developer=per_dev_config)
            flat = next(a for a in strategy.assign(structured_records) if a.cluster_id == "flat" and a.developer_label == "hash-chunk 0")
            by_seed[seed] = tuple(sorted(r.file_id for r in flat.records))
        assert by_seed[1] != by_seed[2]

    def test_chunk_membership_is_not_alphabetic_slicing(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        """The hash shuffle must not degrade into 'first N files alphabetically'."""
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=7, per_developer=per_dev_config)
        flat_assignments = [a for a in strategy.assign(structured_records) if a.cluster_id == "flat"]
        first_chunk_ids = sorted(r.file_id for r in flat_assignments[0].records)
        alphabetic_first_chunk = sorted(
            (r.file_id for r in structured_records if r.cluster_id == "flat"), key=str
        )[: len(first_chunk_ids)]
        assert first_chunk_ids != alphabetic_first_chunk

    def test_intra_shard_order_is_still_canonical(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        """Group *membership* is hash-shuffled; order *within* a shard must stay file_id-sorted."""
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=3, per_developer=per_dev_config)
        for assignment in strategy.assign(structured_records):
            ids = [r.file_id for r in assignment.records]
            assert ids == sorted(ids)


class TestSingleDeveloperFallback:
    def test_tiny_cluster_becomes_one_developer(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config)
        tiny = [a for a in strategy.assign(structured_records) if a.cluster_id == "tiny"]
        assert len(tiny) == 1
        assert tiny[0].developer_basis == "single_developer_whole_project"
        assert tiny[0].num_files == 3

    def test_min_files_per_developer_prevents_over_splitting(self, per_dev_config: PerDeveloperConfig) -> None:
        # 5 flat files, target 4/dev -> a naive floor(5/4)=1 chunk anyway, but
        # explicitly verify a stricter min forces the single-developer fallback.
        records = [make_nested_record("small", f"src/small/m{i}.py") for i in range(5)]
        strict = PerDeveloperConfig(target_files_per_developer=2, min_files_per_developer=3)
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=strict)
        assignments = strategy.assign(records)
        # floor(5/2)=2 chunks of ~2-3 files; with min=3 that can't clear for both
        # chunks (2 and 3), so it must fall back to one whole-project developer.
        assert len(assignments) == 1
        assert assignments[0].developer_basis == "single_developer_whole_project"


class TestPerDeveloperGeneral:
    def test_empty_corpus_raises(self, per_dev_config: PerDeveloperConfig) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config)
        with pytest.raises(PartitionError, match="empty corpus"):
            strategy.assign([])

    def test_no_cross_project_leakage(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        """Every developer shard's records share exactly one cluster_id, and
        every file_id is assigned to exactly one developer shard overall —
        the two invariants that together rule out cross-project leakage.
        """
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config)
        assignments = strategy.assign(structured_records)

        seen: dict[str, str] = {}
        for assignment in assignments:
            cluster_ids = {r.cluster_id for r in assignment.records}
            assert cluster_ids == {assignment.cluster_id}
            for record in assignment.records:
                assert record.file_id not in seen, f"{record.file_id} assigned twice"
                seen[record.file_id] = assignment.client_id
        assert set(seen) == {r.file_id for r in structured_records}

    def test_client_ids_follow_the_per_developer_template(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        custom = replace(per_dev_config, client_id_template="dev-{cluster_id}-{dev_index}")
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=custom)
        for assignment in strategy.assign(structured_records):
            assert assignment.client_id.startswith(f"dev-{assignment.cluster_id}-")

    def test_default_config_is_used_when_none_given(self, structured_records: list[CorpusRecord]) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=1)
        assignments = strategy.assign(structured_records)  # must not raise
        assert assignments

    def test_deterministic_under_input_reordering(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        strategy = PerDeveloperStrategy(ClientNamingConfig(), seed=5, per_developer=per_dev_config)
        forward = strategy.assign(structured_records)
        reversed_ = strategy.assign(list(reversed(structured_records)))
        assert [a.client_id for a in forward] == [a.client_id for a in reversed_]
        assert [[r.file_id for r in a.records] for a in forward] == [
            [r.file_id for r in a.records] for a in reversed_
        ]

    def test_project_level_and_per_developer_agree_on_total_files(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        project_level = ProjectLevelStrategy(ClientNamingConfig(), seed=1).assign(structured_records)
        per_developer = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config).assign(
            structured_records
        )
        assert sum(a.num_files for a in project_level) == sum(a.num_files for a in per_developer)

    def test_per_developer_client_count_is_at_least_project_level(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        project_level = ProjectLevelStrategy(ClientNamingConfig(), seed=1).assign(structured_records)
        per_developer = PerDeveloperStrategy(ClientNamingConfig(), seed=1, per_developer=per_dev_config).assign(
            structured_records
        )
        assert len(per_developer) >= len(project_level)

    def test_registry_wires_per_developer_config_through(
        self, structured_records: list[CorpusRecord], per_dev_config: PerDeveloperConfig
    ) -> None:
        built = build_strategy(PartitionStrategyName.PER_DEVELOPER, ClientNamingConfig(), 1, per_dev_config)
        assert isinstance(built, PerDeveloperStrategy)
        modular = [a for a in built.assign(structured_records) if a.cluster_id == "modular"]
        assert len(modular) == 3  # confirms per_dev_config's thresholds were actually used


class TestPerDeveloperConfigValidation:
    def test_rejects_non_positive_target(self) -> None:
        with pytest.raises(ConfigError, match="target_files_per_developer"):
            PerDeveloperConfig(target_files_per_developer=0)

    def test_rejects_non_positive_minimum(self) -> None:
        with pytest.raises(ConfigError, match="min_files_per_developer"):
            PerDeveloperConfig(min_files_per_developer=0)

    def test_rejects_template_missing_placeholders(self) -> None:
        with pytest.raises(ConfigError, match="client_id_template"):
            PerDeveloperConfig(client_id_template="client-{cluster_id}")

    def test_client_id_formats_correctly(self) -> None:
        config = PerDeveloperConfig()
        assert config.client_id("flask", 2) == "client-flask-dev2"


# ---------------------------------------------------------------------------
# End-to-end: Partitioner + PartitionValidator together (Week 4 Monday/Tuesday)
# ---------------------------------------------------------------------------
class TestPerDeveloperEndToEnd:
    @pytest.fixture
    def per_dev_partition_config(self, tmp_path: Path, structured_records: list[CorpusRecord]) -> PartitionConfig:
        from partitions.models import PartitionInputConfig, PartitionOutputConfig, ValidationConfig
        from utils.io_utils import write_jsonl

        corpus_path = tmp_path / "processed" / "corpus.jsonl"
        write_jsonl(corpus_path, (r.to_dict() for r in structured_records))

        return PartitionConfig(
            strategy="per_developer",
            seed=1,
            manifest_version="1.0.0",
            input=PartitionInputConfig(corpus_path=corpus_path, corpus_manifest_path=None),
            output=PartitionOutputConfig(
                partitions_dir=tmp_path / "partitions" / "per_developer",
                manifest_path=tmp_path / "partitions" / "per_developer" / "manifest.json",
                shard_filename_template="{client_id}.jsonl",
                include_content=True,
            ),
            client_naming=ClientNamingConfig(template="client-{cluster_id}"),
            validation=ValidationConfig(min_files_per_shard=2, max_imbalance_ratio=20.0),
            per_developer=PerDeveloperConfig(target_files_per_developer=4, min_files_per_developer=2),
        )

    def test_builds_a_valid_manifest(self, per_dev_partition_config: PartitionConfig) -> None:
        result = Partitioner(per_dev_partition_config).run()
        assert result.manifest.strategy is PartitionStrategyName.PER_DEVELOPER
        assert result.manifest.num_clients == len(result.manifest.shards)
        assert result.manifest_path.is_file()

    def test_passes_full_leakage_validation(
        self, per_dev_partition_config: PartitionConfig, structured_records: list[CorpusRecord]
    ) -> None:
        from utils.io_utils import read_jsonl

        result = Partitioner(per_dev_partition_config).run()
        shard_records = {
            client_id: [CorpusRecord.from_dict(row) for row in read_jsonl(path)]
            for client_id, path in result.shard_paths.items()
        }
        outcome = PartitionValidator(per_dev_partition_config.validation).validate(
            result.manifest, shard_records, corpus_records=structured_records
        )
        assert outcome.ok, outcome.errors
        assert "no identical content shared across shards" in outcome.checks_passed
        assert any("non-IID by project" in c for c in outcome.checks_passed)

    def test_reproducible_shard_digests(self, per_dev_partition_config: PartitionConfig) -> None:
        first = Partitioner(per_dev_partition_config).run().manifest
        second = Partitioner(per_dev_partition_config).run().manifest
        assert [s.content_sha256 for s in first.shards] == [s.content_sha256 for s in second.shards]

    def test_assignments_carry_developer_basis(self, per_dev_partition_config: PartitionConfig) -> None:
        result = Partitioner(per_dev_partition_config).run()
        assert result.assignments
        assert all(a.developer_basis is not None for a in result.assignments)
