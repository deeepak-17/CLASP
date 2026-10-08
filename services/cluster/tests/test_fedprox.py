"""Week 2 Tue: FedProx proximal term in the client loop (needs torch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from cluster.adapter_format import random_adapter
from cluster.client import LoRAClient, ToyLoRAModel


def _client(mu: float, lr: float = 0.05, steps: int = 40, seed: int = 3) -> LoRAClient:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(64, 32, generator=gen)
    y = x @ (torch.randn(32, 32, generator=gen) / 32**0.5).T
    return LoRAClient("c", ToyLoRAModel(dim=32, seed=seed), (x, y), mu=mu, lr=lr, local_steps=steps)


def _distance_to_global(arrays_out, arrays_in) -> float:
    return float(np.sqrt(sum(((o - i) ** 2).sum() for o, i in zip(arrays_out, arrays_in))))


def test_proximal_term_keeps_params_closer_to_global():
    start = random_adapter(32, 32, seed=0).to_ndarrays()
    dists = {}
    for mu in (0.0, 2.0, 10.0):  # lr*mu stays < 2 (explicit-Euler stability)
        out, _, _ = _client(mu).fit([a.copy() for a in start], {})
        dists[mu] = _distance_to_global(out, start)
    assert dists[0.0] > dists[2.0] > dists[10.0]  # stronger mu => smaller drift


def test_local_training_reduces_loss_and_reports_metrics():
    start = random_adapter(32, 32, seed=0).to_ndarrays()
    client = _client(mu=0.01)
    loss_before, _, _ = client.evaluate([a.copy() for a in start], {})
    out, n, metrics = client.fit([a.copy() for a in start], {})
    loss_after, _, _ = client.evaluate(out, {})
    assert loss_after < loss_before
    assert n == 64 and metrics["client_id"] == "c" and metrics["duration_s"] >= 0
    assert np.isclose(metrics["loss"], loss_after, rtol=0.2)


def test_mu_zero_equals_plain_sgd_and_is_deterministic():
    start = random_adapter(32, 32, seed=0).to_ndarrays()
    a, _, _ = _client(0.0).fit([x.copy() for x in start], {})
    b, _, _ = _client(0.0).fit([x.copy() for x in start], {})
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_simulated_delay_is_reported_not_slept():
    import time

    start = random_adapter(32, 32, seed=0).to_ndarrays()
    c = _client(0.01, steps=1)
    c.simulated_delay_s = 120.0
    t0 = time.monotonic()
    _, _, metrics = c.fit([x.copy() for x in start], {})
    assert time.monotonic() - t0 < 30
    assert metrics["duration_s"] >= 120.0


def test_large_mu_times_lr_diverges_documenting_the_stability_bound():
    """The proximal gradient step is explicit Euler on a quadratic: it is stable
    only while lr * mu < 2. The Week-9 'FedProx mu tuning for stability' must
    therefore be done against the client lr, not in isolation."""
    start = random_adapter(32, 32, seed=0).to_ndarrays()
    from cluster.adapter_format import AdapterFormatError

    # lr*mu = 2.5: the client diverges, and its (NaN) adapter is refused at the
    # format boundary instead of being uploaded to poison the aggregate
    with pytest.raises(AdapterFormatError, match="non-finite"):
        _client(mu=50.0, lr=0.05).fit([a.copy() for a in start], {})
    out, _, metrics = _client(mu=10.0, lr=0.05).fit([a.copy() for a in start], {})  # lr*mu = 0.5
    assert np.isfinite(metrics["loss"]) and all(np.isfinite(o).all() for o in out)
