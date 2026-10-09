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

from collections.abc import Iterable, Iterator, Sequence
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


def _svd(delta_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``np.linalg.svd`` with a transpose retry on LAPACK non-convergence.

    Integration sprint, real-adapter run: LAPACK's ``gesdd`` (what numpy calls)
    raised ``LinAlgError: SVD did not converge`` on one real 2048x2048 averaged
    delta_W (layers.12.self_attn.o_proj of the three web clients) — a known
    gesdd failure mode on near-degenerate spectra, and one the 32-dim synthetic
    demo could never hit. The same matrix transposed converges and yields
    identical singular values, so the retry is a different LAPACK entry path to
    the *same* decomposition, not different mathematics: for A = U S V^T,
    A^T = V S U^T, so swapping the factors back recovers exactly U, S, V^T.
    """
    try:
        return np.linalg.svd(delta_w, full_matrices=False)
    except np.linalg.LinAlgError:
        u_t, s, vt_t = np.linalg.svd(np.ascontiguousarray(delta_w.T), full_matrices=False)
        return vt_t.T, s, u_t.T


def truncated_svd_refactor(delta_w: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Best rank-``rank`` factorization of delta_w: returns (A, B), B @ A ≈ delta_w.

    Singular values are split symmetrically (sqrt into each factor) so A and B
    stay on comparable scales for subsequent client-side fine-tuning.
    """
    if delta_w.ndim != 2:
        raise ValueError("delta_w must be 2-D")
    u, s, vt = _svd(delta_w)
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


def aggregate_layerwise(
    adapters: Sequence[LoRAAdapter],
    weights: Sequence[float],
    rank: int | None = None,
    aggregation: str = "svd",
) -> LoRAAdapter:
    """Memory-bounded driver over ``aggregate_svd`` / ``aggregate_naive``.

    NO NEW MATHEMATICS. Both aggregators already treat every (layer, module)
    pair independently, so running them one layer at a time and stitching the
    results back together is bit-for-bit the same adapter as running them over
    the whole thing — it just never holds more than one layer's accumulators.

    Why it is needed: ``aggregate_svd`` allocates one float64 accumulator per
    (layer, module) up front. At real width that is 24 x 4 x 2048 x 2048 x 8 B
    = 3.2 GB before the first SVD even starts, which is more RAM than the
    demo machine has (8 GB total, ~1 GB free). Per layer it is 134 MB.

    The per-layer slices reference the SAME numpy arrays as ``adapters`` (no
    copy); only the aggregated output is new memory.
    """
    if aggregation not in ("svd", "naive"):
        raise ValueError(f"unknown aggregation {aggregation!r}")
    adapters = list(adapters)
    weights = list(weights)
    if not adapters:
        raise ValueError("no adapters to aggregate")

    template = adapters[0]
    template.validate()
    out_rank = rank if rank is not None else template.rank
    modules: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    for layer in template.layer_indices:
        slices = [
            LoRAAdapter(
                rank=a.rank,
                alpha=a.alpha,
                target_modules=a.target_modules,
                num_layers=1,
                modules={0: a.modules[layer]},
            )
            for a in adapters
        ]
        if aggregation == "svd":
            merged_layer = aggregate_svd(iter(slices), weights, rank=out_rank)
        else:
            merged_layer = aggregate_naive(iter(slices), weights)
        modules[layer] = merged_layer.modules[0]
        del slices, merged_layer

    return LoRAAdapter(
        rank=out_rank if aggregation == "svd" else template.rank,
        alpha=template.alpha,
        target_modules=template.target_modules,
        num_layers=template.num_layers,
        modules=modules,
    )


def layerwise_exact_average_error(
    adapters: Sequence[LoRAAdapter],
    weights: Sequence[float],
    merged: LoRAAdapter,
) -> dict[str, float]:
    """Relative Frobenius error of ``merged`` against the EXACT weighted average.

    Reuses ``exact_average_delta`` unchanged, one layer at a time (same memory
    argument as ``aggregate_layerwise``). This is the reconstruction-error
    figure the aggregation manifest carries: it is the price rank-r truncation
    pays, measured rather than assumed.
    """
    adapters = list(adapters)
    weights = list(weights)
    worst, total, n = 0.0, 0.0, 0
    for layer in merged.layer_indices:
        exact = exact_average_delta(iter(adapters), weights, layer=layer)
        for module in merged.target_modules:
            ref = exact[module]
            denom = float(np.linalg.norm(ref))
            rel = float(np.linalg.norm(merged.delta_w(module, layer) - ref) / denom) if denom else 0.0
            worst = max(worst, rel)
            total += rel
            n += 1
        del exact
    return {
        "max_rel_err": round(worst, 6),
        "mean_rel_err": round(total / n, 6) if n else 0.0,
        "n_modules": n,
    }


def aggregate_svd_lowrank(
    adapters: Sequence[LoRAAdapter],
    weights: Sequence[float],
    rank: int | None = None,
) -> tuple[LoRAAdapter, dict[str, float]]:
    """``aggregate_svd``, computed through the low-rank factors instead of a
    dense 2048x2048 average. Same mathematics, same result, ~1000x faster.

    THIS IS NOT A DIFFERENT ALGORITHM. It is D2's exact pipeline —
    weighted-average delta_W, then best rank-r truncation — evaluated in a way
    that never forms the dense average. For k clients of rank r:

        dW = sum_i c_i B_i A_i = Bcat @ Acat,
            Bcat = [c_1 B_1 | ... | c_k B_k]   (out x k*r)
            Acat = [A_1 ; ... ; A_k]           (k*r x in)

        Bcat = Q_B R_B            (QR, out x k*r)
        Acat^T = Q_A R_A          (QR, in x k*r)
        dW = Q_B (R_B R_A^T) Q_A^T

    Q_B and Q_A have orthonormal columns, so the SVD of the tiny (k*r x k*r)
    core C = R_B R_A^T lifts to the exact SVD of dW:
    U = Q_B U_c, V = Q_A V_c, same singular values. Truncating to r and
    splitting sqrt(S) symmetrically is then bit-for-bit the same construction
    ``truncated_svd_refactor`` performs — see
    ``tests/test_aggregation.py::test_lowrank_matches_reference_aggregate_svd``,
    which asserts agreement with ``aggregate_svd`` itself.

    Why it exists: at real width the reference path allocates one dense float64
    accumulator per (layer, module) — 3.2 GB for a 24-layer adapter — and then
    runs a 2048x2048 LAPACK SVD per module, measured at ~12 s each, i.e. ~20
    min per cluster on the demo machine against a 30-minute round NFR (D11).
    Neither the memory nor the wall clock fits. ``aggregate_svd`` is untouched
    and remains the reference oracle.

    Returns (aggregated adapter, per-module relative Frobenius truncation
    error). The error is exact and free here: it is
    sqrt(sum_{j>r} s_j^2) / sqrt(sum_j s_j^2) from the singular values already
    computed, rather than a second dense pass.
    """
    adapters = list(adapters)
    weights = [float(w) for w in weights]
    if len(adapters) != len(weights):
        raise ValueError(
            f"got {len(adapters)} adapters and {len(weights)} weights — "
            f"cannot aggregate a partial pair"
        )
    if not adapters:
        raise ValueError("no adapters to aggregate")
    if any(w <= 0 for w in weights):
        raise ValueError(f"weights must be positive, got {weights}")

    template = adapters[0]
    template.validate()
    for other in adapters[1:]:
        other.validate()
        if other.target_modules != template.target_modules:
            raise ValueError("all client adapters must share target_modules")
        if other.num_layers != template.num_layers:
            raise ValueError("all client adapters must share num_layers")

    total = sum(weights)
    coeffs = [w / total for w in weights]
    out_rank = rank if rank is not None else template.rank

    modules: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    errors: dict[str, float] = {}
    for layer in template.layer_indices:
        modules[layer] = {}
        for module in template.target_modules:
            b_cat = np.concatenate(
                [c * a.modules[layer][module]["lora_B"].astype(np.float64)
                 for c, a in zip(coeffs, adapters)],
                axis=1,
            )
            a_cat = np.concatenate(
                [a.modules[layer][module]["lora_A"].astype(np.float64)
                 for a in adapters],
                axis=0,
            )
            q_b, r_b = np.linalg.qr(b_cat)
            q_a, r_a = np.linalg.qr(a_cat.T)
            u_c, s, vt_c = _svd(r_b @ r_a.T)

            keep = min(out_rank, s.shape[0])
            sqrt_s = np.sqrt(s[:keep])
            b = (q_b @ u_c[:, :keep]) * sqrt_s          # (out, keep)
            a = sqrt_s[:, None] * (vt_c[:keep, :] @ q_a.T)  # (keep, in)
            if keep < out_rank:  # pad so every module reports the same rank
                b = np.concatenate([b, np.zeros((b.shape[0], out_rank - keep))], axis=1)
                a = np.concatenate([a, np.zeros((out_rank - keep, a.shape[1]))], axis=0)

            dtype = template.modules[layer][module]["lora_A"].dtype
            modules[layer][module] = {"lora_A": a.astype(dtype), "lora_B": b.astype(dtype)}

            total_energy = float(np.sum(s ** 2))
            tail = float(np.sum(s[keep:] ** 2))
            errors[f"layers.{layer}.{module}"] = (
                float(np.sqrt(tail / total_energy)) if total_energy > 0 else 0.0
            )
            del b_cat, a_cat, q_b, r_b, q_a, r_a, u_c, vt_c

    merged = LoRAAdapter(
        rank=out_rank,
        alpha=template.alpha,
        target_modules=template.target_modules,
        num_layers=template.num_layers,
        modules=modules,
    )
    return merged, errors
