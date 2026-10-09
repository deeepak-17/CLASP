"""Core data contracts shared across CLASP services — the integration seam.

Intentionally dependency-free dataclasses so every module can consume them
without pulling heavy deps (torch, fastapi, flower). Promote to pydantic models
at the API boundary in ``services/registry`` where request/response validation
is needed; these dataclasses remain the canonical field definitions.

Frozen at **v1.0.0**; **v1.1.0** is an additive, backward-compatible minor bump
(composite adapters for D6, lossless JSON for every seam). Changes require a
semver bump + all-hands sign-off. The three cross-module schemas below are the
wire seams:

    Edge  ->  Cluster   : AdapterUpload      (a client's trained LoRA)
    Cluster -> Registry : ClusterSnapshot    (aggregated cluster adapter)
    Registry -> Eval    : EvalResult         (metrics that drive promotion)

Composition model:

    Wnew = Wbase + alpha * dWcluster + beta * dWclient

Every value object has ``to_json()`` / ``from_json()``. Use them on both sides
of an HTTP hop: JSON has no integer keys and no tuples, so a plain
``asdict`` + ``json.loads`` silently turns ``pass_at_k[1]`` into ``pass_at_k["1"]``.
``from_json`` also accepts every v1.0 payload unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

#: Semantic version of this contract package. Bump on any breaking change.
CONTRACTS_VERSION = "1.1.0"

#: Base model a composite was merged against, recorded on its provenance (D6).
DEFAULT_BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"


def utcnow_iso() -> str:
    """UTC timestamp in ISO-8601, used everywhere a ``timestamp`` field appears."""
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class AdapterKind(str, Enum):
    CLIENT = "client"        # per-developer LoRA (dW_client)
    CLUSTER = "cluster"      # per-project federated LoRA (dW_cluster)
    COMPOSITE = "composite"  # v1.1: pre-merged alpha*dW_cluster + beta*dW_client (D6)


class AggregationMethod(str, Enum):
    """How the cluster server combined client adapters (D2)."""
    SVD_EXACT = "svd_exact"    # reconstruct dW=B.A, exact-average, re-factorize via SVD
    NAIVE_AVG = "naive_avg"    # average A/B factors directly — ABLATION BASELINE ONLY


class PromotionAction(str, Enum):
    PROMOTE = "promote"    # repoint `active` tag to this version
    ROLLBACK = "rollback"  # repoint `active` tag to the previous version


# --------------------------------------------------------------------------- #
# Shared value objects (the "extended types" — D7/D8/D9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoRAHyperParams:
    """LoRA + composition hyperparameters carried on every adapter (D8 caps)."""
    rank: int = 16                                  # D8: rank 16
    lora_alpha: int = 32                            # LoRA scaling (distinct from composition alpha)
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Composition coefficients for Wnew = Wbase + alpha*dWcluster + beta*dWclient
    alpha: float = 1.0                              # cluster contribution
    beta: float = 1.0                               # client contribution

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "target_modules": list(self.target_modules)}

    @classmethod
    def from_json(cls, d: dict[str, Any] | None) -> LoRAHyperParams:
        d = d or {}
        base = cls()
        return cls(
            rank=d.get("rank", base.rank),
            lora_alpha=d.get("lora_alpha", base.lora_alpha),
            dropout=d.get("dropout", base.dropout),
            target_modules=tuple(d.get("target_modules", base.target_modules)),
            alpha=d.get("alpha", base.alpha),
            beta=d.get("beta", base.beta),
        )


@dataclass(frozen=True)
class PrivacySpec:
    """Client-level DP accounting (D7). epsilon logged per round into metadata."""
    epsilon: float | None = None   # spent budget; None => DP disabled (ablation)
    delta: float = 1e-5
    noise_multiplier: float | None = None
    max_grad_norm: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any] | None) -> PrivacySpec:
        return cls(**(d or {}))


@dataclass(frozen=True)
class AdapterRef:
    """An immutable reference to a versioned LoRA adapter in the registry."""
    name: str
    version: int
    kind: AdapterKind
    cluster_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "kind": self.kind.value}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> AdapterRef:
        return cls(
            name=d["name"],
            version=d["version"],
            kind=AdapterKind(d.get("kind", AdapterKind.CLIENT.value)),
            cluster_id=d.get("cluster_id"),
        )


@dataclass(frozen=True)
class CompositeProvenance:
    """v1.1 — exactly what a pre-merged composite adapter was built from (D6).

    alpha/beta are pinned here as well as in ``hparams`` because the composite
    is merged once, at promotion, and the coefficients actually used must stay
    on the immutable artifact even if experiment defaults change later (D9).
    """
    cluster_name: str
    cluster_version: int
    client_name: str
    client_version: int
    alpha: float
    beta: float
    base_model: str = DEFAULT_BASE_MODEL

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CompositeProvenance:
        return cls(**d)


# --------------------------------------------------------------------------- #
# Seam 1 — Edge -> Cluster : a client uploads its trained LoRA adapter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdapterUpload:
    """Payload a client sends to the cluster server after local training.

    The tensor bytes travel out-of-band (safetensors); this struct is the
    metadata envelope. ``num_train_samples`` is required for FedProx-weighted
    exact averaging on the server (D2).
    """
    client_id: str
    cluster_id: str
    kind: AdapterKind                      # CLIENT for uploads
    hparams: LoRAHyperParams
    num_train_samples: int
    round: int
    privacy: PrivacySpec = field(default_factory=PrivacySpec)
    seed: int = 0
    timestamp: str = field(default_factory=utcnow_iso)
    contracts_version: str = CONTRACTS_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "kind": self.kind.value,
            "hparams": self.hparams.to_json(),
            "privacy": self.privacy.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> AdapterUpload:
        return cls(
            client_id=d["client_id"],
            cluster_id=d["cluster_id"],
            kind=AdapterKind(d["kind"]),
            hparams=LoRAHyperParams.from_json(d.get("hparams")),
            num_train_samples=d["num_train_samples"],
            round=d["round"],
            privacy=PrivacySpec.from_json(d.get("privacy")),
            seed=d.get("seed", 0),
            timestamp=d.get("timestamp") or utcnow_iso(),
            contracts_version=d.get("contracts_version", "1.0.0"),
        )


# --------------------------------------------------------------------------- #
# Seam 2 — Cluster -> Registry : aggregated cluster adapter snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterSnapshot:
    """A cluster's aggregated adapter, pushed to the registry after a round."""
    cluster_id: str
    round: int
    kind: AdapterKind                      # CLUSTER for snapshots
    hparams: LoRAHyperParams
    aggregation: AggregationMethod
    participating_clients: tuple[str, ...]
    privacy: PrivacySpec = field(default_factory=PrivacySpec)
    seed: int = 0
    timestamp: str = field(default_factory=utcnow_iso)
    contracts_version: str = CONTRACTS_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "kind": self.kind.value,
            "hparams": self.hparams.to_json(),
            "aggregation": self.aggregation.value,
            "participating_clients": list(self.participating_clients),
            "privacy": self.privacy.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ClusterSnapshot:
        return cls(
            cluster_id=d["cluster_id"],
            round=d["round"],
            kind=AdapterKind(d["kind"]),
            hparams=LoRAHyperParams.from_json(d.get("hparams")),
            aggregation=AggregationMethod(d["aggregation"]),
            participating_clients=tuple(d.get("participating_clients", ())),
            privacy=PrivacySpec.from_json(d.get("privacy")),
            seed=d.get("seed", 0),
            timestamp=d.get("timestamp") or utcnow_iso(),
            contracts_version=d.get("contracts_version", "1.0.0"),
        )


# --------------------------------------------------------------------------- #
# Seam 3 — Registry -> Eval : evaluation outcome that drives promotion (D5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InProjectMetrics:
    """Primary metric (D5): completion quality on the client's held-out files."""
    edit_similarity: float
    exact_match: float
    perplexity: float
    n_examples: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any] | None) -> InProjectMetrics | None:
        if d is None:
            return None
        return cls(
            edit_similarity=d["edit_similarity"],
            exact_match=d["exact_match"],
            perplexity=d["perplexity"],
            n_examples=d["n_examples"],
        )


@dataclass(frozen=True)
class GuardMetrics:
    """Regression guard (D5): HumanEval / MBPP pass@k. Never the sole gate.

    v1.1: keys are normalized to ``int`` on construction (so a dict that came
    off the wire with ``"1"`` keys still answers ``pass_at_k[1]``), and every
    entry is validated — k >= 1, 0 <= pass@k <= 1.
    """
    benchmark: str                 # "HumanEval" | "MBPP"
    pass_at_k: dict[int, float]    # e.g. {1: 0.31, 10: 0.52}

    def __post_init__(self) -> None:
        normalized: dict[int, float] = {}
        for raw_k, raw_v in self.pass_at_k.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError) as e:
                raise ValueError(f"pass@k key {raw_k!r} is not an integer") from e
            v = float(raw_v)
            if k < 1:
                raise ValueError(f"pass@k needs k >= 1, got k={k}")
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"pass@{k} must be a fraction in [0, 1], got {v}")
            normalized[k] = v
        object.__setattr__(self, "pass_at_k", normalized)

    def pass_at(self, k: int) -> float | None:
        return self.pass_at_k.get(k)

    def to_json(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "pass_at_k": {str(k): v for k, v in self.pass_at_k.items()},
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> GuardMetrics:
        return cls(benchmark=d["benchmark"], pass_at_k=dict(d["pass_at_k"]))


@dataclass(frozen=True)
class EvalResult:
    """Full evaluation of a candidate adapter version, consumed by the promotion rule.

    D5 two-sided rule: promote iff in-project improves beyond ``baseline_noise_band``
    AND HumanEval pass@1 drop <= 2 points absolute; otherwise roll back.
    """
    adapter: AdapterRef
    in_project: InProjectMetrics
    guard: tuple[GuardMetrics, ...] = ()
    baseline_in_project: InProjectMetrics | None = None  # previous `active` for delta
    baseline_noise_band: float = 0.0                     # from 3 repeated baseline evals
    seed: int = 0
    timestamp: str = field(default_factory=utcnow_iso)
    contracts_version: str = CONTRACTS_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter.to_json(),
            "in_project": self.in_project.to_json(),
            "guard": [g.to_json() for g in self.guard],
            "baseline_in_project": (
                self.baseline_in_project.to_json() if self.baseline_in_project else None
            ),
            "baseline_noise_band": self.baseline_noise_band,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "contracts_version": self.contracts_version,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EvalResult:
        return cls(
            adapter=AdapterRef.from_json(d["adapter"]),
            in_project=InProjectMetrics.from_json(d["in_project"]),
            guard=tuple(GuardMetrics.from_json(g) for g in d.get("guard", ())),
            baseline_in_project=InProjectMetrics.from_json(d.get("baseline_in_project")),
            baseline_noise_band=d.get("baseline_noise_band", 0.0),
            seed=d.get("seed", 0),
            timestamp=d.get("timestamp") or utcnow_iso(),
            contracts_version=d.get("contracts_version", "1.0.0"),
        )


@dataclass(frozen=True)
class PromotionDecision:
    """Result of applying the D5 two-sided rule; the registry acts on this."""
    adapter: AdapterRef
    action: PromotionAction
    active_version_after: int
    reason: str
    timestamp: str = field(default_factory=utcnow_iso)

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "adapter": self.adapter.to_json(),
            "action": self.action.value,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PromotionDecision:
        return cls(
            adapter=AdapterRef.from_json(d["adapter"]),
            action=PromotionAction(d["action"]),
            active_version_after=d["active_version_after"],
            reason=d["reason"],
            timestamp=d.get("timestamp") or utcnow_iso(),
        )


# --------------------------------------------------------------------------- #
# Registry-owned records (D9): what persists next to each safetensors blob
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdapterMetadata:
    """Stored as ``metadata.json`` beside each versioned safetensors file.

    Immutable once written; a new adapter version = a new directory. Captures
    everything needed to reproduce and to audit privacy budget (D7/D9).
    """
    ref: AdapterRef
    hparams: LoRAHyperParams
    privacy: PrivacySpec
    aggregation: AggregationMethod | None       # set for CLUSTER adapters, None for CLIENT
    round: int | None
    seed: int
    sha256: str                                 # digest of the safetensors payload
    num_bytes: int
    source_clients: tuple[str, ...] = ()
    created_at: str = field(default_factory=utcnow_iso)
    contracts_version: str = CONTRACTS_VERSION
    composed_from: CompositeProvenance | None = None  # v1.1: set iff kind == COMPOSITE

    def to_json(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_json(),
            "hparams": self.hparams.to_json(),
            "privacy": self.privacy.to_json(),
            "aggregation": self.aggregation.value if self.aggregation else None,
            "round": self.round,
            "seed": self.seed,
            "sha256": self.sha256,
            "num_bytes": self.num_bytes,
            "source_clients": list(self.source_clients),
            "created_at": self.created_at,
            "contracts_version": self.contracts_version,
            "composed_from": self.composed_from.to_json() if self.composed_from else None,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> AdapterMetadata:
        composed = d.get("composed_from")
        return cls(
            ref=AdapterRef.from_json(d["ref"]),
            hparams=LoRAHyperParams.from_json(d["hparams"]),
            privacy=PrivacySpec.from_json(d["privacy"]),
            aggregation=AggregationMethod(d["aggregation"]) if d.get("aggregation") else None,
            round=d.get("round"),
            seed=d["seed"],
            sha256=d["sha256"],
            num_bytes=d["num_bytes"],
            source_clients=tuple(d.get("source_clients", ())),
            created_at=d["created_at"],
            contracts_version=d.get("contracts_version", "1.0.0"),
            composed_from=CompositeProvenance.from_json(composed) if composed else None,
        )


@dataclass(frozen=True)
class RunManifest:
    """One per experiment run (D9). Recorded before the run; GPU-hrs estimated first."""
    run_id: str
    seed: int
    config_hash: str                            # sha256 of the resolved experiment config
    adapter_versions: dict[str, int]            # adapter name -> version produced
    gpu_hours_estimate: float
    gpu_hours_actual: float | None = None
    notes: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    contracts_version: str = CONTRACTS_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RunManifest:
        return cls(**d)
