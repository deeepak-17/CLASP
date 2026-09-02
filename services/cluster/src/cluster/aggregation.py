"""Cluster-side LoRA aggregation (Week 3, D2).

Pipeline per (layer, target module):
    1. reconstruct delta_W_k = B_k @ A_k for every client k        (W3 Tue)
    2. streamed exact weighted average of the delta_Ws             (W3 Wed)
       - one running accumulator; never materializes all clients
    3. truncated SVD re-factorization of the average back to r=16  (W3 Thu)
       - delta_avg ≈ B' @ A' with B' = U_r sqrt(S_r), A' = sqrt(S_r) V_r^T

Naive parameter-space averaging (mean of A_k, mean of B_k) is kept as the
ablation baseline (D2): mean(B_k) @ mean(A_k) != mean(B_k @ A_k) in general,
which is exactly the error the SVD path avoids.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import zip_longest

import numpy as np

from cluster.adapter_format import LoRAAdapter

# Sentinel used to detect a length mismatch between `adapters` and `weights`
# without materializing either into a list (see `_paired`).
_MISSING = object()


class StreamingWeightedMean:
    """Exact weighted mean computed one contribution at a time (W3 Wed).

    Uses the incremental update  m <- m + (w_i / W_i) * (x_i - m), which is
    exact (up to float error) and keeps memory at one tensor regardless of
    the number of clients.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._total_weight: float = 0.0

    def update(self, value: np.ndarray, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")
        value = np.asarray(value, dtype=np.float64)
        self._total_weight += weight
        if self._mean is None:
            self._mean = value.copy()
        else:
            self._mean += (weight / self._total_weight) * (value - self._mean)

    @property
    def total_weight(self) -> float:
        return self._total_weight

    def result(self) -> np.ndarray:
        if self._mean is None:
            raise ValueError("no contributions received")
        return self._mean


def _paired(
    adapters: Iterable[LoRAAdapter], weights: Iterable[float]
) -> Iterator[tuple[LoRAAdapter, float]]:
    """zip(adapters, weights), but raise instead of silently truncating.

    Plain ``zip`` stops at the shorter iterable with no error — if ``weights``
    is ever shorter than ``adapters`` (e.g. an off-by-one when a client drops
    mid-round), fewer clients get aggregated with no signal that anything was
    dropped. This raises as soon as either side runs out first, and — unlike
    materializing ``adapters`` into a list to compare lengths up front — keeps
    the "adapters is only iterated once, streamed" contract intact.
    """
    for i, (adapter, weight) in enumerate(
        zip_longest(adapters, weights, fillvalue=_MISSING)
    ):
        if adapter is _MISSING:
            raise ValueError(
                f"weights has more entries than adapters (adapters exhausted "
                f"after {i} pair(s)) — cannot aggregate a partial pair"
            )
        if weight is _MISSING:
            raise ValueError(
                f"adapters has more entries than weights (weights exhausted "
                f"after {i} pair(s)) — cannot aggregate a partial pair"
            )
        yield adapter, weight


def truncated_svd_refactor(delta_w: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Best rank-``rank`` factorization of delta_w: returns (A, B), B @ A ≈ delta_w.

    Singular values are split symmetrically (sqrt into each factor) so A and B
    stay on comparable scales for subsequent client-side fine-tuning.
    """
    if delta_w.ndim != 2:
        raise ValueError("delta_w must be 2-D")
    u, s, vt = np.linalg.svd(delta_w, full_matrices=False)
    u_r, s_r, vt_r = u[:, :rank], s[:rank], vt[:rank, :]
    sqrt_s = np.sqrt(s_r)
    b = u_r * sqrt_s  # (out, r)
    a = sqrt_s[:, None] * vt_r  # (r, in)
    return a, b


def aggregate_svd(
    adapters: Iterable[LoRAAdapter],
    weights: Iterable[float],
    rank: int | None = None,
) -> LoRAAdapter:
    """SVD aggregation: exact-average delta_W per (layer, module), re-factorized to rank r.

    ``adapters`` is only iterated once, so it may be a generator that loads /
    receives one client adapter at a time (streamed).
    """
    means: dict[tuple[int, str], StreamingWeightedMean] = {}
    template: LoRAAdapter | None = None
    for adapter, weight in _paired(adapters, weights):
        adapter.validate()
        if template is None:
            template = adapter
            means = {
                (layer, m): StreamingWeightedMean()
                for layer in adapter.layer_indices
                for m in adapter.target_modules
            }
        elif adapter.target_modules != template.target_modules:
            raise ValueError("all client adapters must share target_modules")
        elif adapter.num_layers != template.num_layers:
            raise ValueError("all client adapters must share num_layers")
        for layer in adapter.layer_indices:
            for module in adapter.target_modules:
                means[(layer, module)].update(adapter.delta_w(module, layer), weight)
    if template is None:
        raise ValueError("no adapters to aggregate")

    out_rank = rank if rank is not None else template.rank
    modules: dict[int, dict[str, dict[str, np.ndarray]]] = {
        layer: {} for layer in template.layer_indices
    }
    for layer in template.layer_indices:
        for module in template.target_modules:
            a, b = truncated_svd_refactor(means[(layer, module)].result(), out_rank)
            dtype = template.modules[layer][module]["lora_A"].dtype
            modules[layer][module] = {"lora_A": a.astype(dtype), "lora_B": b.astype(dtype)}
    return LoRAAdapter(
        rank=out_rank,
        alpha=template.alpha,
        target_modules=template.target_modules,
        num_layers=template.num_layers,
        modules=modules,
    )


def aggregate_naive(
    adapters: Iterable[LoRAAdapter],
    weights: Iterable[float],
) -> LoRAAdapter:
    """Ablation baseline (D2): weighted average of A and B factors directly."""
    means: dict[tuple[int, str], dict[str, StreamingWeightedMean]] = {}
    template: LoRAAdapter | None = None
    for adapter, weight in _paired(adapters, weights):
        adapter.validate()
        if template is None:
            template = adapter
            means = {
                (layer, m): {"lora_A": StreamingWeightedMean(), "lora_B": StreamingWeightedMean()}
                for layer in adapter.layer_indices
                for m in adapter.target_modules
            }
        elif adapter.target_modules != template.target_modules:
            # mirror the mismatch guard from aggregate_svd; without this,
            # mismatched clients silently corrupt the averaged factors.
            raise ValueError("all client adapters must share target_modules")
        elif adapter.num_layers != template.num_layers:
            raise ValueError("all client adapters must share num_layers")
        for layer in adapter.layer_indices:
            for module in adapter.target_modules:
                for part in ("lora_A", "lora_B"):
                    means[(layer, module)][part].update(
                        adapter.modules[layer][module][part], weight
                    )
    if template is None:
        raise ValueError("no adapters to aggregate")

    modules: dict[int, dict[str, dict[str, np.ndarray]]] = {
        layer: {
            module: {
                part: means[(layer, module)][part]
                .result()
                .astype(template.modules[layer][module][part].dtype)
                for part in ("lora_A", "lora_B")
            }
            for module in template.target_modules
        }
        for layer in template.layer_indices
    }
    return LoRAAdapter(
        rank=template.rank,
        alpha=template.alpha,
        target_modules=template.target_modules,
        num_layers=template.num_layers,
        modules=modules,
    )


def exact_average_delta(
    adapters: Iterable[LoRAAdapter],
    weights: Iterable[float],
    layer: int = 0,
) -> dict[str, np.ndarray]:
    """Reference: the exact weighted-average delta_W per module, no
    re-factorization. Single-layer by default (``layer=0``), matching how
    ``delta_w()`` defaults — pass ``layer`` to inspect another one.
    """
    means: dict[str, StreamingWeightedMean] = {}
    modules_order: tuple[str, ...] | None = None
    for adapter, weight in _paired(adapters, weights):
        if modules_order is None:
            modules_order = adapter.target_modules
            means = {m: StreamingWeightedMean() for m in modules_order}
        for module in modules_order:
            means[module].update(adapter.delta_w(module, layer), weight)
    if modules_order is None:
        raise ValueError("no adapters to aggregate")
    return {m: means[m].result() for m in modules_order}
