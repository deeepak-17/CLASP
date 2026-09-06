"""Partitioning strategies — Week 2, Monday.

"Define project-level partition logic (non-IID by project)."

Why project-level is the correct non-IID axis for CLASP
-------------------------------------------------------
CLASP composes ``W_base + alpha*dW_cluster + beta*dW_client``. The cluster
adapter is *defined* as "what this project's code has in common", so the
federated partition boundary must be the project boundary. Any partition that
mixes projects within a shard makes ``dW_cluster`` an average over unrelated
codebases, which is precisely the thing CLASP claims to avoid.

This is genuinely non-IID rather than nominally so: each repository has its
own import graph, naming conventions, exception hierarchy and API idiom, so
the per-shard token distributions differ in kind and not just in sample.
Shard *sizes* also differ substantially (Colorama vs Werkzeug), which is the
client-drift condition FedProx's proximal term exists to stabilise.

A :class:`PartitionStrategy` is deliberately a pure function from corpus
records to a client assignment: no I/O, no config side effects. Writing
shards, hashing and manifest construction all live in
:mod:`partitions.partitioner`, which keeps strategies trivial to unit test and
makes the Week-4 per-developer strategy a drop-in addition.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from corpus.models import CorpusRecord
from interfaces.contracts import PartitionStrategyName
from partitions.models import ClientNamingConfig, PerDeveloperConfig
from utils.errors import PartitionError
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class ClientAssignment:
    """The records assigned to one federated client, in deterministic order.

    ``developer_basis``/``developer_label`` are set only by
    :class:`PerDeveloperStrategy`, to record *how* a "developer" boundary was
    derived (see that class's docstring). They are P5-local traceability,
    deliberately not part of the frozen ``PartitionShard``/``PartitionManifest``
    contract — :mod:`partitions.metadata` and the Week-4 comparison report
    are what surface them.
    """

    client_id: str
    cluster_id: str
    project_label: str
    records: list[CorpusRecord] = field(default_factory=list)
    developer_basis: str | None = None
    developer_label: str | None = None

    def __post_init__(self) -> None:
        if not self.records:
            raise PartitionError(f"Client '{self.client_id}' was assigned no records")

    @property
    def num_files(self) -> int:
        return len(self.records)

    @property
    def num_lines(self) -> int:
        return sum(r.num_lines for r in self.records)

    @property
    def num_code_lines(self) -> int:
        return sum(r.num_code_lines for r in self.records)

    @property
    def total_bytes(self) -> int:
        return sum(r.num_bytes for r in self.records)


class PartitionStrategy(ABC):
    """Assigns corpus records to federated clients."""

    #: Contract enum value this strategy implements.
    name: PartitionStrategyName

    def __init__(self, naming: ClientNamingConfig, seed: int) -> None:
        self._naming = naming
        self._seed = seed

    @abstractmethod
    def assign(self, records: list[CorpusRecord]) -> list[ClientAssignment]:
        """Partition ``records`` into per-client assignments.

        Implementations must be deterministic: the same records and seed must
        always produce the same assignment, in the same order.
        """

    @staticmethod
    def _sort_records(records: list[CorpusRecord]) -> list[CorpusRecord]:
        """Canonical intra-shard ordering: by ``file_id``.

        Ordering is part of the contract, because each shard's
        ``content_sha256`` is computed over the ordered file hashes. Sorting
        by ``file_id`` makes that hash stable across filesystems whose
        directory iteration order differs.
        """
        return sorted(records, key=lambda r: r.file_id)


class ProjectLevelStrategy(PartitionStrategy):
    """One shard per project. **The Week-2 deliverable.**

    Each source repository becomes exactly one cluster and, in Week 2, exactly
    one Flower client. Week 8 keeps the same cluster boundary and splits each
    cluster into several clients; the manifest schema already distinguishes
    ``cluster_id`` from ``client_id`` so that change requires no contract
    revision.
    """

    name = PartitionStrategyName.PROJECT_LEVEL

    def assign(self, records: list[CorpusRecord]) -> list[ClientAssignment]:
        if not records:
            raise PartitionError("Cannot partition an empty corpus")

        grouped: dict[str, list[CorpusRecord]] = defaultdict(list)
        labels: dict[str, str] = {}
        for record in records:
            grouped[record.cluster_id].append(record)
            existing = labels.setdefault(record.cluster_id, record.project_label)
            if existing != record.project_label:
                raise PartitionError(
                    f"Cluster '{record.cluster_id}' carries conflicting project labels: "
                    f"'{existing}' and '{record.project_label}'"
                )

        assignments = [
            ClientAssignment(
                client_id=self._naming.client_id(cluster_id),
                cluster_id=cluster_id,
                project_label=labels[cluster_id],
                records=self._sort_records(grouped[cluster_id]),
            )
            # Sorting by cluster_id makes shard order — and therefore the
            # manifest — reproducible.
            for cluster_id in sorted(grouped)
        ]

        _LOG.info(
            "project_level: %d record(s) -> %d client(s) across %d cluster(s)",
            len(records),
            len(assignments),
            len(grouped),
        )
        return assignments


class PerDeveloperStrategy(PartitionStrategy):
    """Per-developer (individual-style) partitioning. **The Week-4 deliverable.**

    What "developer" means in this corpus
    --------------------------------------
    There is no real per-developer identity to split on. ``configs/dataset.yaml``
    shallow-clones every source repository (``acquisition.git_depth: 1``), so
    every file in a project resolves to the *same single commit* — verified
    directly against this repository's own checkouts: ``git log --oneline``
    under ``datasets/raw/flask`` (and every other cluster) returns exactly one
    commit, authored by whoever cut that release tag, not by the people who
    actually wrote each file. Splitting on that would not be "per-developer",
    it would be "per-release-tagger", which is a fabricated identity wearing
    a real-sounding label. Per the Week-4 requirement to not fabricate
    developer identities, this strategy uses only structure the corpus
    genuinely carries:

    1. **Module path** (``basis: module_path``) — when a project's files
       form real subpackages (e.g. ``werkzeug``'s ``routing/``, ``debug/``,
       ``middleware/``), each subpackage becomes one developer's slice. This
       is a legitimate, common proxy for contribution boundaries — module
       ownership is how many real engineering orgs *do* divide work — and it
       is directly present in ``CorpusRecord.relative_path``, unlike
       authorship.
    2. **Deterministic hash-chunking** (``basis: deterministic_hash_chunk``)
       — for projects with no subpackages at all (a flat file list — three
       of D1's six projects: Requests, Click, Colorama, plus Jinja2, which
       is also flat despite its size), files are ordered by
       ``sha256(seed:cluster_id:file_id)`` and cut into contiguous,
       roughly-equal chunks. Hashing before chunking (rather than just
       slicing the alphabetically-sorted list) avoids the artefact of
       grouping files by name prefix, which is not a meaningful boundary
       either.

    Either basis can degrade to a single developer (== the whole project,
    identical to :class:`ProjectLevelStrategy`'s shard) when the project is
    too small to clear ``per_developer.min_files_per_developer`` more than
    once — Colorama's 5 files are the concrete case in D1. That is reported,
    not hidden: see :mod:`partitions.metadata` and the Week-4 comparison
    report.

    Every developer shard stays inside its project's ``cluster_id`` — this
    strategy only ever *subdivides* what :class:`ProjectLevelStrategy`
    assigns to a cluster, so the disjointness/content-disjointness/purity
    checks in :mod:`partitions.validation` apply unchanged and continue to
    prove no cross-project leakage.
    """

    name = PartitionStrategyName.PER_DEVELOPER

    def __init__(
        self,
        naming: ClientNamingConfig,
        seed: int,
        per_developer: PerDeveloperConfig | None = None,
    ) -> None:
        super().__init__(naming, seed)
        self._config = per_developer or PerDeveloperConfig()

    def assign(self, records: list[CorpusRecord]) -> list[ClientAssignment]:
        if not records:
            raise PartitionError("Cannot partition an empty corpus")

        grouped: dict[str, list[CorpusRecord]] = defaultdict(list)
        labels: dict[str, str] = {}
        for record in records:
            grouped[record.cluster_id].append(record)
            existing = labels.setdefault(record.cluster_id, record.project_label)
            if existing != record.project_label:
                raise PartitionError(
                    f"Cluster '{record.cluster_id}' carries conflicting project labels: "
                    f"'{existing}' and '{record.project_label}'"
                )

        assignments: list[ClientAssignment] = []
        for cluster_id in sorted(grouped):
            cluster_records = self._sort_records(grouped[cluster_id])
            groups = self._module_groups(cluster_records)
            basis = "module_path"
            if len(groups) < 2 or not self._clears_minimum(groups):
                groups = self._hash_chunks(cluster_records, cluster_id)
                basis = "deterministic_hash_chunk"
            if len(groups) < 2 or not self._clears_minimum(groups):
                # Too small to split even one level further than the whole
                # project: the honest answer is one developer == the project.
                groups = [cluster_records]
                basis = "single_developer_whole_project"

            for dev_index, group_records in enumerate(groups):
                label = self._group_label(basis, group_records, dev_index)
                assignments.append(
                    ClientAssignment(
                        client_id=self._config.client_id(cluster_id, dev_index),
                        cluster_id=cluster_id,
                        project_label=labels[cluster_id],
                        records=group_records,
                        developer_basis=basis,
                        developer_label=label,
                    )
                )

        _LOG.info(
            "per_developer: %d record(s) -> %d client(s) across %d cluster(s)",
            len(records),
            len(assignments),
            len(grouped),
        )
        return assignments

    # --- internals ---------------------------------------------------------
    def _clears_minimum(self, groups: list[list[CorpusRecord]]) -> bool:
        return all(len(group) >= self._config.min_files_per_developer for group in groups)

    @staticmethod
    def _module_key(record: CorpusRecord) -> str:
        """The directory a file lives in, as a module/subpackage proxy.

        ``src/werkzeug/routing/map.py`` -> ``src/werkzeug/routing``;
        ``src/werkzeug/http.py`` -> ``src/werkzeug`` (the package root, its
        own bucket). A flat project (every file at the same depth) collapses
        to one key, which is exactly the signal :meth:`assign` uses to fall
        back to hash-chunking instead.
        """
        parts = record.relative_path.rsplit("/", 1)
        return parts[0] if len(parts) == 2 else "."

    def _module_groups(self, records: list[CorpusRecord]) -> list[list[CorpusRecord]]:
        by_module: dict[str, list[CorpusRecord]] = defaultdict(list)
        for record in records:
            by_module[self._module_key(record)].append(record)
        # Sorted by key for determinism; records within each group are
        # already canonically ordered (records passed in are pre-sorted).
        return [by_module[key] for key in sorted(by_module)]

    def _hash_chunks(self, records: list[CorpusRecord], cluster_id: str) -> list[list[CorpusRecord]]:
        """Deterministically shuffle by ``sha256(seed:cluster_id:file_id)``, then chunk.

        Hashing before chunking avoids the alphabetic-prefix artefact a plain
        slice of the (file_id-sorted) input would have — files are already
        canonically sorted by file_id for the shard's *internal* order, but
        that same order must not determine *group membership*, or the split
        would just be "files starting with a-m" vs "n-z", which is no more
        meaningful than the flat-directory case it exists to fix.
        """
        target = self._config.target_files_per_developer
        num_chunks = max(1, len(records) // target)

        def sort_key(record: CorpusRecord) -> str:
            digest = hashlib.sha256(f"{self._seed}:{cluster_id}:{record.file_id}".encode("utf-8"))
            return digest.hexdigest()

        shuffled = sorted(records, key=sort_key)

        chunks: list[list[CorpusRecord]] = [[] for _ in range(num_chunks)]
        for index, record in enumerate(shuffled):
            chunks[index % num_chunks].append(record)
        # Restore canonical (file_id) order within each chunk — the hash
        # shuffle is only for group *membership*, not intra-shard ordering,
        # which the rest of the pipeline (shard digests, tests) assumes.
        return [self._sort_records(chunk) for chunk in chunks if chunk]

    @staticmethod
    def _group_label(basis: str, records: list[CorpusRecord], dev_index: int) -> str:
        if basis == "module_path":
            return PerDeveloperStrategy._module_key(records[0])
        if basis == "single_developer_whole_project":
            return "<whole project — too small to split further>"
        return f"hash-chunk {dev_index}"


#: Strategy registry. Adding a strategy means adding one entry here.
STRATEGIES: dict[PartitionStrategyName, type[PartitionStrategy]] = {
    PartitionStrategyName.PROJECT_LEVEL: ProjectLevelStrategy,
    PartitionStrategyName.PER_DEVELOPER: PerDeveloperStrategy,
}


def build_strategy(
    name: PartitionStrategyName,
    naming: ClientNamingConfig,
    seed: int,
    per_developer: PerDeveloperConfig | None = None,
) -> PartitionStrategy:
    """Instantiate the strategy registered under ``name``.

    ``per_developer`` is ignored by every strategy except
    :class:`PerDeveloperStrategy`; kept as one shared optional parameter
    (rather than a strategy-specific factory signature) so callers do not
    need to know which concrete strategy they are building.
    """
    try:
        strategy_cls = STRATEGIES[name]
    except KeyError as exc:  # pragma: no cover - config validation catches this first
        raise PartitionError(f"No strategy registered for '{name}'") from exc
    if strategy_cls is PerDeveloperStrategy:
        return PerDeveloperStrategy(naming, seed, per_developer)
    return strategy_cls(naming, seed)
