"""Client <-> cluster-server message schemas (contracts draft, Week 1).

Wire format: tensors travel as base64-encoded little-endian buffers with an
explicit dtype + shape so the payload is JSON-safe end to end (Flower configs,
registry snapshots, mTLS channels all carry JSON-compatible dicts).

Extended metadata fields (rank, target_modules, alpha, beta, epsilon, seed,
timestamp) follow the Registry->Eval extended-type list frozen in contracts
v1.0.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class TensorPayload(BaseModel):
    """A single tensor encoded for transport."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_b64: str

    @classmethod
    def from_numpy(cls, name: str, array: np.ndarray) -> "TensorPayload":
        array = np.ascontiguousarray(array)
        return cls(
            name=name,
            dtype=str(array.dtype),
            shape=tuple(array.shape),
            data_b64=base64.b64encode(array.tobytes()).decode("ascii"),
        )

    def to_numpy(self) -> np.ndarray:
        raw = base64.b64decode(self.data_b64)
        return np.frombuffer(raw, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()

    @model_validator(mode="after")
    def _check_size(self) -> "TensorPayload":
        expected = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        actual = len(base64.b64decode(self.data_b64))
        if expected != actual:
            raise ValueError(
                f"tensor {self.name!r}: payload is {actual} bytes, "
                f"shape/dtype imply {expected}"
            )
        return self


class AdapterUpload(BaseModel):
    """Edge client -> cluster server: one trained LoRA adapter."""

    schema_version: str = SCHEMA_VERSION
    client_id: str
    round_id: int = Field(ge=0)
    rank: int = Field(gt=0)
    target_modules: tuple[str, ...]
    alpha: float = Field(gt=0)
    num_examples: int = Field(gt=0)
    seed: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tensors: list[TensorPayload]

    @field_validator("target_modules")
    @classmethod
    def _non_empty_modules(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("target_modules must not be empty")
        return v

    @model_validator(mode="after")
    def _check_tensor_coverage(self) -> "AdapterUpload":
        # Every target module must ship exactly one lora_A and one lora_B.
        names = {t.name for t in self.tensors}
        missing = [
            f"{m}.{part}"
            for m in self.target_modules
            for part in ("lora_A", "lora_B")
            if f"{m}.{part}" not in names
        ]
        if missing:
            raise ValueError(f"upload missing tensors: {missing}")
        return self


class ClusterAdapterBroadcast(BaseModel):
    """Cluster server -> edge clients: the aggregated cluster adapter."""

    schema_version: str = SCHEMA_VERSION
    cluster_id: str
    round_id: int = Field(ge=0)
    rank: int = Field(gt=0)
    target_modules: tuple[str, ...]
    alpha: float = Field(gt=0)
    num_clients: int = Field(gt=0)
    aggregation: str = "svd"  # "svd" | "naive" (ablation baseline, D2)
    epsilon: float | None = None  # DP budget spent, filled by P3's accountant
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tensors: list[TensorPayload]


class RoundMetrics(BaseModel):
    """Per-round bookkeeping logged by the server."""

    round_id: int = Field(ge=0)
    num_clients: int = Field(ge=0)
    mean_loss: float | None = None
    duration_s: float | None = None
    aggregation: str = "svd"
