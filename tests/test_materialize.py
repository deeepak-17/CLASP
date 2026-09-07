"""Tests for scripts/materialize_client_repo.py (E3.1 support).

Every test builds its own manifest+shards under ``tmp_path`` via the
``partition_config`` fixture (see ``tests/conftest.py``); none of these tests
read or write the repository's real ``datasets/`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from partitions.materialize import (
    MaterializeConfig,
    materialize_client_repo,
    split_held_out,
)
from partitions.models import PartitionConfig
from partitions.partitioner import Partitioner
from utils.errors import ConfigError, PartitionError
from interfaces.contracts import ContractViolationError


@pytest.fixture
def manifest(partition_config: PartitionConfig):
    return Partitioner(partition_config).run().manifest


@pytest.fixture
def materialize_config(tmp_path: Path, partition_config: PartitionConfig) -> MaterializeConfig:
    return MaterializeConfig(
        manifest_path=partition_config.output.manifest_path,
        output_root=tmp_path / "materialized",
        seed=42,
        held_out_fraction=0.10,
    )


class TestMaterializeConfig:
    def test_rejects_out_of_range_fraction(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            MaterializeConfig(manifest_path=tmp_path / "m.json", output_root=tmp_path, held_out_fraction=1.0)

    def test_rejects_negative_seed(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            MaterializeConfig(manifest_path=tmp_path / "m.json", output_root=tmp_path, seed=-1)

    def test_rejects_matching_dirnames(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            MaterializeConfig(
                manifest_path=tmp_path / "m.json",
                output_root=tmp_path,
                repo_dirname="x",
                held_out_dirname="x",
            )


class TestSplitHeldOut:
    def test_kept_and_held_out_partition_the_shard(self, manifest, materialize_config) -> None:
        from partitions.partitioner import load_shard_records

        records = load_shard_records(manifest, "client-alpha")
        kept, held_out = split_held_out(
            records, client_id="client-alpha", seed=42, fraction=0.10
        )
        assert sorted(r.file_id for r in kept + held_out) == sorted(r.file_id for r in records)
        assert not set(r.file_id for r in kept) & set(r.file_id for r in held_out)
        assert len(kept) > 0
        assert len(held_out) > 0

    def test_is_deterministic_for_a_fixed_seed(self, manifest) -> None:
        from partitions.partitioner import load_shard_records

        records = load_shard_records(manifest, "client-alpha")
        first = split_held_out(records, client_id="client-alpha", seed=42, fraction=0.10)
        second = split_held_out(records, client_id="client-alpha", seed=42, fraction=0.10)
        assert [r.file_id for r in first[0]] == [r.file_id for r in second[0]]
        assert [r.file_id for r in first[1]] == [r.file_id for r in second[1]]

    def test_different_seeds_can_change_the_split(self, manifest) -> None:
        # 6 files is too few for every seed pair to disagree by chance (two
        # seeds can land on the same 50/50 cut), so sweep several seeds and
        # require at least one disagreement rather than asserting on a
        # single arbitrary pair.
        from partitions.partitioner import load_shard_records

        records = load_shard_records(manifest, "client-alpha")  # 6 files
        splits = [
            frozenset(r.file_id for r in split_held_out(records, client_id="client-alpha", seed=s, fraction=0.5)[1])
            for s in range(10)
        ]
        assert len(set(splits)) > 1

    def test_rejects_shards_too_small_to_split(self) -> None:
        with pytest.raises(PartitionError):
            split_held_out([], client_id="client-x", seed=1, fraction=0.1)


class TestMaterializeClientRepo:
    def test_writes_kept_files_at_their_relative_paths(self, manifest, materialize_config) -> None:
        result = materialize_client_repo(manifest, "client-alpha", materialize_config)
        for relative_path in result.kept_files:
            assert (result.repo_dir / relative_path).is_file()
        for relative_path in result.held_out_files:
            assert (result.held_out_dir / relative_path).is_file()

    def test_written_content_matches_the_shard(self, manifest, materialize_config) -> None:
        from partitions.partitioner import load_shard_records

        records = {r.relative_path: r for r in load_shard_records(manifest, "client-alpha")}
        result = materialize_client_repo(manifest, "client-alpha", materialize_config)
        for relative_path in result.kept_files:
            on_disk = (result.repo_dir / relative_path).read_text(encoding="utf-8")
            assert on_disk == records[relative_path].content

    def test_kept_and_held_out_are_disjoint_and_cover_the_shard(self, manifest, materialize_config) -> None:
        result = materialize_client_repo(manifest, "client-alpha", materialize_config)
        assert not set(result.kept_files) & set(result.held_out_files)
        shard = manifest.shard_for("client-alpha")
        assert result.num_kept + result.num_held_out == shard.num_files

    def test_holds_out_roughly_ten_percent(self, manifest, materialize_config) -> None:
        result = materialize_client_repo(manifest, "client-alpha", materialize_config)
        shard = manifest.shard_for("client-alpha")
        assert result.num_held_out == max(1, round(shard.num_files * 0.10))

    def test_writes_materialize_manifest(self, manifest, materialize_config) -> None:
        result = materialize_client_repo(manifest, "client-alpha", materialize_config)
        assert result.manifest_path.is_file()

    def test_dry_run_writes_nothing(self, manifest, materialize_config) -> None:
        result = materialize_client_repo(manifest, "client-alpha", materialize_config, dry_run=True)
        assert not result.repo_dir.exists()
        assert not result.held_out_dir.exists()
        assert not result.manifest_path.exists()

    def test_does_not_touch_the_partition_shard(self, manifest, materialize_config, partition_config) -> None:
        shard_path = partition_config.output.shard_path("client-alpha")
        before = shard_path.read_bytes()
        materialize_client_repo(manifest, "client-alpha", materialize_config)
        assert shard_path.read_bytes() == before

    def test_unknown_client_id_raises(self, manifest, materialize_config) -> None:
        with pytest.raises(ContractViolationError):
            materialize_client_repo(manifest, "client-does-not-exist", materialize_config)

    def test_is_reproducible_across_runs(self, manifest, materialize_config) -> None:
        first = materialize_client_repo(manifest, "client-alpha", materialize_config)
        second = materialize_client_repo(manifest, "client-alpha", materialize_config)
        assert first.kept_files == second.kept_files
        assert first.held_out_files == second.held_out_files

    def test_rejects_relative_path_escaping_output_dir(self, manifest, materialize_config) -> None:
        from dataclasses import replace

        from partitions.partitioner import load_shard_records

        records = load_shard_records(manifest, "client-alpha")
        bad = replace(records[0], relative_path="../../escaped.py")
        from partitions.materialize import _safe_write

        with pytest.raises(PartitionError):
            _safe_write(materialize_config.repo_dir("client-alpha"), bad.relative_path, bad.content)
