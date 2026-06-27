"""Core data contracts shared across CLASP services.

Intentionally dependency-free dataclasses so every module can consume them
without pulling heavy deps. Promote to pydantic models at the API boundary
in services/registry if/when validation is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AdapterKind(str, Enum):
    CLIENT = "client"   # per-developer LoRA (Delta W_client)
    CLUSTER = "cluster"  # per-project federated LoRA (Delta W_cluster)


@dataclass(frozen=True)
class AdapterRef:
    """An immutable reference to a versioned LoRA adapter in the registry."""
    name: str
    version: int
    kind: AdapterKind
    cluster_id: str | None = None


@dataclass(frozen=True)
class EvalResult:
    """Pass@k evaluation outcome used to promote/rollback an adapter."""
    adapter: AdapterRef
    benchmark: str          # "HumanEval" | "MBPP"
    pass_at_k: dict[int, float]  # e.g. {1: 0.31, 10: 0.52}
