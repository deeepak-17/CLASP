"""Partition construction — Week 2, Tuesday and Thursday.

Tue: "Implement partition script v1 on collected dataset"
Thu: "Write partition metadata (sizes, project labels) to disk"

Pipeline
--------
::

    datasets/processed/corpus.jsonl
        -> PartitionStrategy.assign()             (partitions.strategies)
        -> one JSONL shard per client             (datasets/partitions/*.jsonl)
        -> PartitionManifest                      (datasets/partitions/manifest.json)

The manifest is the artefact P1, P2 and P4 consume. It carries the corpus
SHA-256 so a shard set can always be traced back to the exact corpus bytes it
came from, and a per-shard ``content_sha256`` so a tampered or truncated shard
is detectable without re-reading the corpus.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from corpus.collector import load_corpus_records
from corpus.models import CorpusRecord
from interfaces.contracts import PartitionManifest, PartitionShard
from partitions.models import PartitionConfig
from partitions.strategies import ClientAssignment, build_strategy
from utils.config import load_config
from utils.errors import PartitionError
from utils.io_utils import ensure_dir, read_json, sha256_file, write_json, write_jsonl
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.timing import Stopwatch, utc_timestamp

_LOG = get_logger(__name__)


def load_partition_config(path: Path | str | None = None) -> PartitionConfig:
    """Load ``configs/partition.yaml`` into a :class:`PartitionConfig`."""
    target = path or (project_paths().configs / "partition.yaml")
    return load_config(PartitionConfig, target, section="partition")


@dataclass(frozen=True)
class PartitionResult:
    """What a partitioning run produced."""

    manifest: PartitionManifest
    manifest_path: Path
    shard_paths: dict[str, Path]
    elapsed_seconds: float
    #: The strategy's own ClientAssignment objects, in manifest order. Not
    #: part of any contract — carries P5-local traceability (e.g.
    #: PerDeveloperStrategy's developer_basis/developer_label) that the
    #: frozen PartitionShard/PartitionManifest deliberately does not.
    assignments: tuple[ClientAssignment, ...] = ()

    @property
    def num_clients(self) -> int:
        return self.manifest.num_clients


class Partitioner:
    """Turns a collected corpus into federated client shards plus a manifest."""

    def __init__(self, config: PartitionConfig) -> None:
        self._config = config
        self._strategy = build_strategy(
            config.strategy_name, config.client_naming, config.seed, config.per_developer
        )

    # --- public API --------------------------------------------------------
    def run(self, *, dry_run: bool = False) -> PartitionResult:
        """Build shards and the manifest.

        Args:
            dry_run: Compute everything but write nothing. Shard content
                hashes are still computed, so a dry run is a genuine check.
        """
        paths = project_paths()
        corpus_path = Path(self._config.input.corpus_path)

        records = load_corpus_records(corpus_path)
        _LOG.info("Loaded %d corpus record(s) from %s", len(records), paths.relative(corpus_path))
        self._warn_if_synthetic()

        with Stopwatch("partition") as watch:
            assignments = self._strategy.assign(records)

            shard_paths: dict[str, Path] = {}
            shards: list[PartitionShard] = []

            for assignment in assignments:
                shard_path = self._config.output.shard_path(assignment.client_id)
                shard_paths[assignment.client_id] = shard_path

                if not dry_run:
                    ensure_dir(shard_path.parent)
                    write_jsonl(
                        shard_path,
                        (
                            record.to_dict(include_content=self._config.output.include_content)
                            for record in assignment.records
                        ),
                    )

                shards.append(
                    PartitionShard(
                        client_id=assignment.client_id,
                        cluster_id=assignment.cluster_id,
                        project_label=assignment.project_label,
                        num_files=assignment.num_files,
                        num_lines=assignment.num_lines,
                        num_code_lines=assignment.num_code_lines,
                        total_bytes=assignment.total_bytes,
                        files_path=paths.relative(shard_path),
                        content_sha256=self._shard_digest(assignment),
                    )
                )
                _LOG.info(
                    "  %-20s cluster=%-10s files=%4d code_lines=%6d",
                    assignment.client_id,
                    assignment.cluster_id,
                    assignment.num_files,
                    assignment.num_code_lines,
                )

            manifest = PartitionManifest(
                manifest_version=self._config.manifest_version,
                strategy=self._config.strategy_name,
                seed=self._config.seed,
                created_at=utc_timestamp(),
                corpus_path=paths.relative(corpus_path),
                corpus_sha256=self._corpus_digest(corpus_path),
                num_clients=len(shards),
                num_clusters=len({shard.cluster_id for shard in shards}),
                total_files=sum(shard.num_files for shard in shards),
                shards=shards,
                cluster_ids=sorted({shard.cluster_id for shard in shards}),
            )

            manifest_path = Path(self._config.output.manifest_path)
            if dry_run:
                _LOG.info("Dry run: %d shard(s) computed, nothing written", len(shards))
            else:
                write_json(manifest_path, manifest.to_dict())
                _LOG.info("Wrote partition manifest -> %s", paths.relative(manifest_path))

        return PartitionResult(
            manifest=manifest,
            manifest_path=manifest_path,
            shard_paths=shard_paths,
            elapsed_seconds=watch.elapsed_seconds,
            assignments=tuple(assignments),
        )

    # --- internals ---------------------------------------------------------
    @staticmethod
    def _shard_digest(assignment: ClientAssignment) -> str:
        """SHA-256 over the shard's ordered per-file content hashes.

        Hashing the file hashes rather than the shard file itself means the
        digest identifies the *content set*, independent of whether the shard
        was written with contents embedded or index-only.
        """
        digest = hashlib.sha256()
        for record in assignment.records:
            digest.update(record.file_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(record.content_sha256.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _corpus_digest(self, corpus_path: Path) -> str:
        """Prefer the digest recorded by the collector; fall back to hashing.

        Preferring the manifest value means a mismatch between the recorded
        corpus and the file on disk becomes visible to
        :mod:`partitions.validation` instead of being papered over.
        """
        manifest_path = self._config.input.corpus_manifest_path
        if manifest_path and Path(manifest_path).is_file():
            payload = read_json(manifest_path)
            recorded = payload.get("corpus_sha256")
            if isinstance(recorded, str) and len(recorded) == 64:
                return recorded
            _LOG.warning("Corpus manifest has no usable corpus_sha256; hashing the corpus directly")
        return sha256_file(corpus_path)

    def _warn_if_synthetic(self) -> None:
        """Emit a loud warning when partitioning a synthetic corpus."""
        manifest_path = self._config.input.corpus_manifest_path
        if not manifest_path or not Path(manifest_path).is_file():
            return
        payload = read_json(manifest_path)
        if payload.get("synthetic"):
            _LOG.warning(
                "Corpus manifest is marked synthetic=true. The resulting partitions are "
                "pipeline fixtures and MUST NOT be used for training or reported results."
            )


def load_partition_manifest(path: Path | str) -> PartitionManifest:
    """Read a partition manifest back into its contract dataclass."""
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise PartitionError(f"Partition manifest at {path} is not a JSON object")
    return PartitionManifest.from_dict(payload)


def load_shard_records(manifest: PartitionManifest, client_id: str) -> list[CorpusRecord]:
    """Load the corpus records belonging to one client from its shard file."""
    from utils.io_utils import read_jsonl  # local import keeps module import light

    shard = manifest.shard_for(client_id)
    shard_path = project_paths().root / shard.files_path
    return [CorpusRecord.from_dict(row) for row in read_jsonl(shard_path)]
