"""Shared helpers for the P2 cluster tests (builders only, no fixtures state)."""

from __future__ import annotations

import numpy as np

from cluster.adapter_format import LoRAAdapter, random_adapter


def trained_adapter(
    seed: int,
    dim: int = 32,
    rank: int = 16,
    num_layers: int = 1,
    direction: np.ndarray | None = None,
    noise: float = 0.0,
) -> LoRAAdapter:
    """An adapter with non-zero B. If ``direction`` is given, every B is
    ``direction + noise * N(0,1)`` so adapters sharing a direction have a high
    delta_W cosine similarity and different directions have a low one."""
    rng = np.random.default_rng(seed)
    adapter = random_adapter(dim, dim, rank=rank, num_layers=num_layers, seed=0)
    for layer in adapter.layer_indices:
        for m in adapter.target_modules:
            if direction is None:
                b = rng.normal(scale=0.1, size=(dim, rank))
            else:
                b = direction + noise * rng.normal(size=(dim, rank))
            adapter.modules[layer][m]["lora_B"] = b.astype(np.float32)
    return adapter


def planted_directions(n_groups: int = 2, dim: int = 32, rank: int = 16, seed: int = 123):
    rng = np.random.default_rng(seed)
    return [rng.normal(scale=0.1, size=(dim, rank)) for _ in range(n_groups)]
