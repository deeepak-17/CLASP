"""Cross-module data contracts that CLASP-P5 produces or consumes.

Ownership
---------
The authoritative ``/contracts`` package for the CLASP system is owned by
**P4 (Deepak — State Registry / DevOps & Orchestration)** and is frozen at the
end of Week 2. This module is P5's *local, dependency-free mirror* of the
subset P5 touches, so the Eval & Data pipeline can be built, tested and
validated before P4's package is published.

Contracts mirrored here
-----------------------
======================  =========  ==========  =====================================
Contract                Producer   Consumer    Week-1/2 relevance
======================  =========  ==========  =====================================
``AdapterRef``          P1/P2      P4, P5      Identifies a versioned LoRA adapter.
``SnapshotMetadata``    P4         P5          Registry -> Eval (read snapshot).
``EvalResult``          P5         P4, P5-UI   Pass@k outcome (populated Week 3+).
``PartitionShard``      P5         P1, P2      One federated client's file slice.
``PartitionManifest``   P5         P1, P2, P4  Week-2 Thursday deliverable.
======================  =========  ==========  =====================================

Migration path: when P4 freezes ``/contracts``, replace the bodies below with
re-exports from that package. :mod:`interfaces.validation` and
``scripts/check_contract_compliance.py`` exist to prove the two agree.

These are intentionally plain ``dataclasses`` with no third-party dependency,
matching the style of the upstream contracts package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from utils.errors import ContractViolationError

#: Version of the contract mirror. Bumped whenever a field is added or renamed;
#: written into every artefact so readers can detect a stale producer.
CONTRACT_VERSION = "0.2.0"


class AdapterKind(str, Enum):
    """Which layer of the CLASP stack an adapter belongs to.

    ``W_new = W_base + alpha * dW_cluster + beta * dW_client``
    """

    CLIENT = "client"
    """Per-developer LoRA (``dW_client``), trained locally by P1."""

    CLUSTER = "cluster"
    """Per-project federated LoRA (``dW_cluster``), aggregated by P2."""


class PartitionStrategyName(str, Enum):
    """Partitioning schemes recognised by the CLASP data pipeline."""

    PROJECT_LEVEL = "project_level"
    """Non-IID by project. One shard per repository. **Week 2 scope.**"""

    PER_DEVELOPER = "per_developer"
    """Per-developer slices within a project. Reserved for Week 4 — declared
    here so the manifest schema is stable, but not implemented in Week 2."""


class BenchmarkName(str, Enum):
    """Code-generation benchmarks P5 is responsible for."""

    HUMANEVAL = "HumanEval"
    MBPP = "MBPP"


@dataclass(frozen=True)
class AdapterRef:
    """Immutable reference to a versioned LoRA adapter held by P4's Registry."""

    name: str
    version: int
    kind: AdapterKind
    cluster_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractViolationError("AdapterRef.name must be non-empty")
        if self.version < 0:
            raise ContractViolationError(f"AdapterRef.version must be >= 0, got {self.version}")
        if self.kind is AdapterKind.CLUSTER and not self.cluster_id:
            raise ContractViolationError("Cluster adapters must carry a cluster_id")

    @property
    def uri(self) -> str:
        """Stable string form, e.g. ``cluster/flask@3``."""
        scope = self.cluster_id or self.name
        return f"{self.kind.value}/{scope}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdapterRef":
        try:
            return cls(
                name=str(data["name"]),
                version=int(data["version"]),
                kind=AdapterKind(data["kind"]),
                cluster_id=data.get("cluster_id"),
            )
        except (KeyError, ValueError) as exc:
            raise ContractViolationError(f"Invalid AdapterRef payload: {exc}") from exc


@dataclass(frozen=True)
class SnapshotMetadata:
    """P4 -> P5. Sidecar written next to a ``safetensors`` snapshot.

    P5 reads this to decide *what* to evaluate; the ``artifact_path`` /
    ``artifact_sha256`` pair lets the harness verify it evaluated the exact
    bytes P4 recorded. Populated for real from Week 7 (eval auto-trigger); in
    Weeks 1-2 it is only produced by :class:`interfaces.registry_client.MockRegistryClient`.
    """

    adapter: AdapterRef
    round_number: int
    created_at: str
    artifact_path: str
    artifact_sha256: str | None = None
    training_loss: float | None = None
    eval_score: float | None = None
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adapter"] = self.adapter.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SnapshotMetadata":
        try:
            return cls(
                adapter=AdapterRef.from_dict(data["adapter"]),
                round_number=int(data["round_number"]),
                created_at=str(data["created_at"]),
                artifact_path=str(data["artifact_path"]),
                artifact_sha256=data.get("artifact_sha256"),
                training_loss=data.get("training_loss"),
                eval_score=data.get("eval_score"),
                contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
            )
        except KeyError as exc:
            raise ContractViolationError(f"SnapshotMetadata missing field: {exc}") from exc


@dataclass(frozen=True)
class EvalResult:
    """P5 -> P4 / dashboard. Outcome of one benchmark run.

    ``pass_at_k`` keys are integers in memory but serialise to JSON strings,
    because JSON object keys must be strings; :meth:`from_dict` reverses this.

    Scoring itself is a **Week 3** deliverable — Week 1/2 only fixes the shape
    so P4 can code the rollback threshold against a stable schema.
    """

    adapter: AdapterRef
    benchmark: BenchmarkName
    pass_at_k: dict[int, float] = field(default_factory=dict)
    num_tasks: int = 0
    num_samples_per_task: int = 1
    created_at: str | None = None
    run_id: str | None = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for k, value in self.pass_at_k.items():
            if k < 1:
                raise ContractViolationError(f"pass@k requires k >= 1, got {k}")
            if not 0.0 <= float(value) <= 1.0:
                raise ContractViolationError(f"pass@{k} must lie in [0, 1], got {value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter.to_dict(),
            "benchmark": self.benchmark.value,
            "pass_at_k": {str(k): v for k, v in sorted(self.pass_at_k.items())},
            "num_tasks": self.num_tasks,
            "num_samples_per_task": self.num_samples_per_task,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalResult":
        try:
            return cls(
                adapter=AdapterRef.from_dict(data["adapter"]),
                benchmark=BenchmarkName(data["benchmark"]),
                pass_at_k={int(k): float(v) for k, v in (data.get("pass_at_k") or {}).items()},
                num_tasks=int(data.get("num_tasks", 0)),
                num_samples_per_task=int(data.get("num_samples_per_task", 1)),
                created_at=data.get("created_at"),
                run_id=data.get("run_id"),
                contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
            )
        except (KeyError, ValueError) as exc:
            raise ContractViolationError(f"Invalid EvalResult payload: {exc}") from exc


@dataclass(frozen=True)
class PartitionShard:
    """P5 -> P1/P2. One federated client's slice of the corpus.

    ``client_id`` is what P2 registers as a Flower client; ``cluster_id`` is
    the project it belongs to. Under the Week-2 project-level scheme there is
    exactly one shard per cluster, so ``client_id == cluster_id``-derived; the
    two fields stay distinct because Week 4 splits a cluster into several
    per-developer clients.
    """

    client_id: str
    cluster_id: str
    project_label: str
    num_files: int
    num_lines: int
    num_code_lines: int
    total_bytes: int
    files_path: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.client_id or not self.cluster_id:
            raise ContractViolationError("PartitionShard requires client_id and cluster_id")
        if self.num_files <= 0:
            raise ContractViolationError(f"Shard {self.client_id} has no files")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionShard":
        try:
            return cls(
                client_id=str(data["client_id"]),
                cluster_id=str(data["cluster_id"]),
                project_label=str(data["project_label"]),
                num_files=int(data["num_files"]),
                num_lines=int(data["num_lines"]),
                num_code_lines=int(data["num_code_lines"]),
                total_bytes=int(data["total_bytes"]),
                files_path=str(data["files_path"]),
                content_sha256=str(data["content_sha256"]),
            )
        except (KeyError, ValueError) as exc:
            raise ContractViolationError(f"Invalid PartitionShard payload: {exc}") from exc


@dataclass(frozen=True)
class PartitionManifest:
    """P5 -> P1/P2/P4. The Week-2 Thursday deliverable.

    This is the single document the rest of the team reads to answer: how many
    federated clients exist, which project each one represents, and how large
    its slice is. ``corpus_sha256`` pins the exact corpus the partition was
    derived from, so a stale manifest is detectable rather than silently wrong.
    """

    manifest_version: str
    strategy: PartitionStrategyName
    seed: int
    created_at: str
    corpus_path: str
    corpus_sha256: str
    num_clients: int
    num_clusters: int
    total_files: int
    shards: list[PartitionShard] = field(default_factory=list)
    cluster_ids: list[str] = field(default_factory=list)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.num_clients != len(self.shards):
            raise ContractViolationError(
                f"num_clients ({self.num_clients}) disagrees with shard count ({len(self.shards)})"
            )
        counted = sum(shard.num_files for shard in self.shards)
        if counted != self.total_files:
            raise ContractViolationError(
                f"total_files ({self.total_files}) disagrees with sum over shards ({counted})"
            )

    def shard_for(self, client_id: str) -> PartitionShard:
        """Look up a shard by client id.

        Raises:
            ContractViolationError: If no such client exists in the manifest.
        """
        for shard in self.shards:
            if shard.client_id == client_id:
                return shard
        raise ContractViolationError(f"No shard for client_id '{client_id}'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "strategy": self.strategy.value,
            "seed": self.seed,
            "created_at": self.created_at,
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "num_clients": self.num_clients,
            "num_clusters": self.num_clusters,
            "total_files": self.total_files,
            "cluster_ids": list(self.cluster_ids),
            "shards": [shard.to_dict() for shard in self.shards],
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionManifest":
        try:
            return cls(
                manifest_version=str(data["manifest_version"]),
                strategy=PartitionStrategyName(data["strategy"]),
                seed=int(data["seed"]),
                created_at=str(data["created_at"]),
                corpus_path=str(data["corpus_path"]),
                corpus_sha256=str(data["corpus_sha256"]),
                num_clients=int(data["num_clients"]),
                num_clusters=int(data["num_clusters"]),
                total_files=int(data["total_files"]),
                cluster_ids=list(data.get("cluster_ids", [])),
                shards=[PartitionShard.from_dict(s) for s in data.get("shards", [])],
                contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
            )
        except (KeyError, ValueError) as exc:
            raise ContractViolationError(f"Invalid PartitionManifest payload: {exc}") from exc
