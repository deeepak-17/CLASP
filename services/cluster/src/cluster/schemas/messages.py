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
    def from_numpy(cls, name: str, array: np.ndarray) -> TensorPayload:
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
    def _check_size(self) -> TensorPayload:
        """Validate declared dtype first, as a clean ValueError.

        Review fix: ``np.dtype(self.dtype)`` raises a bare ``TypeError`` on a
        garbled dtype string (e.g. "bogus"), and pydantic v2's validator
        error handling only converts ``ValueError``/``AssertionError`` into a
        proper ``ValidationError`` (-> HTTP 422 at the FastAPI layer). Left
        unguarded, a malformed tensor payload posted to /uploads crashed the
        request with an unhandled 500 instead of a 422. Wrapping the dtype
        parse and re-raising as ValueError restores the clean 422 path.
        """
        try:
            np_dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(
                f"tensor {self.name!r}: invalid dtype {self.dtype!r}"
            ) from exc
        expected = int(np.prod(self.shape)) * np_dtype.itemsize
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
    num_layers: int = Field(1, ge=1)
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
    def _check_tensor_coverage(self) -> AdapterUpload:
        """Structural check only: enough tensors for every (layer, module,
        lora_A/lora_B) slot, no duplicate names. Real key-by-key parsing —
        layer index, module, A vs B — happens in
        ``cluster.adapter_format.LoRAAdapter.from_state_dict`` at the point
        of use; this validator just rejects a malformed upload before it
        gets there.

        Integration Sprint Defect 2: the original version of this validator
        checked for flat '{module}.lora_A' / '{module}.lora_B' names only,
        with no layer index at all — so it rejected every real multi-layer
        upload (192 tensors across 24 layers) outright. It is deliberately
        layer-COUNT based rather than layer-NAME based, so it does not care
        which of the two key conventions ``LoRAAdapter`` accepts
        (the short 'layers.<i>.<module>...' form or a real PEFT dump like
        'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight')
        the caller actually used.
        """
        expected = 2 * len(self.target_modules) * self.num_layers
        if len(self.tensors) != expected:
            raise ValueError(
                f"expected {expected} tensors (lora_A + lora_B per module, "
                f"{self.num_layers} layer(s), {len(self.target_modules)} "
                f"target module(s)), got {len(self.tensors)}"
            )
        names = [t.name for t in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError("upload contains duplicate tensor names")
        return self


class ClusterAdapterBroadcast(BaseModel):
    """Cluster server -> edge clients: the aggregated cluster adapter."""

    schema_version: str = SCHEMA_VERSION
    cluster_id: str
    round_id: int = Field(ge=0)
    rank: int = Field(gt=0)
    target_modules: tuple[str, ...]
    alpha: float = Field(gt=0)
    num_layers: int = Field(1, ge=1)
    num_clients: int = Field(gt=0)
    aggregation: str = "svd"  # "svd" | "naive" (ablation baseline, D2)
    epsilon: float | None = None  # DP budget spent, filled by P3's accountant
    # PEFT adapter_config.json fields (LoRAAdapter.to_peft_config()) so a
    # consumer can materialize a loadable PEFT directory from this broadcast
    # alone, alongside ``tensors`` in PEFT key form (LoRAAdapter.to_peft_state_dict()).
    peft_config: dict[str, object] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tensors: list[TensorPayload]


class RoundMetrics(BaseModel):
    """Per-round bookkeeping logged by the server."""

    round_id: int = Field(ge=0)
    num_clients: int = Field(ge=0)
    mean_loss: float | None = None
    duration_s: float | None = None
    aggregation: str = "svd"
