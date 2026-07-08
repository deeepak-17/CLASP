"""Core data contracts shared across CLASP services — the integration seam.

Intentionally dependency-free dataclasses so every module can consume them
without pulling heavy deps (torch, fastapi, flower). Promote to pydantic models
at the API boundary in ``services/registry`` where request/response validation
is needed; these dataclasses remain the canonical field definitions.

Frozen at **v1.0.0** (Phase II W2). Changes after the freeze require a semver
bump + all-hands sign-off (see MASTER_PLAN D9). The three cross-module schemas
below are the wire seams:

    Edge  ->  Cluster   : AdapterUpload      (a client's trained LoRA)
    Cluster -> Registry : ClusterSnapshot    (aggregated cluster adapter)
    Registry -> Eval    : EvalResult         (metrics that drive promotion)

Composition model (MASTER_PLAN):

    Wnew = Wbase + alpha * dWcluster + beta * dWclient
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

#: Semantic version of this contract package. Bump on any breaking change.
CONTRACTS_VERSION = "1.0.0"


def utcnow_iso() -> str:
    """UTC timestamp in ISO-8601, used everywhere a ``timestamp`` field appears."""
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class AdapterKind(str, Enum):
    CLIENT = "client"    # per-developer LoRA (dW_client)
    CLUSTER = "cluster"  # per-project federated LoRA (dW_cluster)


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


@dataclass(frozen=True)
class PrivacySpec:
    """Client-level DP accounting (D7). epsilon logged per round into metadata."""
    epsilon: float | None = None   # spent budget; None => DP disabled (ablation)
    delta: float = 1e-5
    noise_multiplier: float | None = None
    max_grad_norm: float | None = None


@dataclass(frozen=True)
class AdapterRef:
    """An immutable reference to a versioned LoRA adapter in the registry."""
    name: str
    version: int
    kind: AdapterKind
    cluster_id: str | None = None


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


@dataclass(frozen=True)
class GuardMetrics:
    """Regression guard (D5): HumanEval / MBPP pass@k. Never the sole gate."""
    benchmark: str                 # "HumanEval" | "MBPP"
    pass_at_k: dict[int, float]    # e.g. {1: 0.31, 10: 0.52}


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


@dataclass(frozen=True)
class PromotionDecision:
    """Result of applying the D5 two-sided rule; the registry acts on this."""
    adapter: AdapterRef
    action: PromotionAction
    active_version_after: int
    reason: str
    timestamp: str = field(default_factory=utcnow_iso)


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
