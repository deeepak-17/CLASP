"""Tests for the Week 5 demo (services/cluster/src/cluster/demo.py).

These exercise the demo's own bookkeeping (engine selection, round metrics,
reproducibility, JSON shape) without duplicating aggregation-math coverage
that belongs to test_aggregation.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster.adapter_format import TARGET_MODULES
from cluster.client import TORCH_AVAILABLE
from cluster.demo import run_demo
from cluster.simulation import make_dummy_clients, random_adapter, run_round


def test_demo_dummy_engine_completes_successfully():
    result = run_demo(clients=3, seed=42, engine="dummy", verbose=False)

    assert result["status"] == "success"
    assert result["engine_used"] == "dummy"
    assert result["round_metrics"]["num_clients"] == 3
    assert result["round_metrics"]["aggregation"] == "svd"
    assert set(result["final_adapter"]) == set(TARGET_MODULES)
    for shapes in result["final_adapter"].values():
        assert shapes["lora_A_shape"] == [16, 32]
        assert shapes["lora_B_shape"] == [32, 16]


def test_demo_dummy_engine_deterministic_shapes_and_status():
    r1 = run_demo(seed=7, engine="dummy", verbose=False)
    r2 = run_demo(seed=7, engine="dummy", verbose=False)
    assert r1["final_adapter"] == r2["final_adapter"]
    assert r1["status"] == r2["status"] == "success"


def test_demo_fedprox_engine_without_torch_gives_clear_error(monkeypatch):
    """engine='fedprox' must fail loudly (not silently degrade) if torch is missing."""
    if TORCH_AVAILABLE:
        pytest.skip("torch is installed in this environment; nothing to assert here")
    with pytest.raises(RuntimeError, match="torch"):
        run_demo(engine="fedprox", verbose=False)


def test_demo_auto_engine_falls_back_when_torch_missing():
    result = run_demo(engine="auto", seed=1, verbose=False)
    expected = "fedprox" if TORCH_AVAILABLE else "dummy"
    assert result["engine_used"] == expected
    assert result["status"] == "success"


def test_round_result_reproducible_end_to_end():
    """Same seed -> byte-identical aggregated adapter (the demo's reproducibility claim)."""

    def one_run(seed: int):
        clients = make_dummy_clients(n=3)
        global_adapter = random_adapter(32, 32, rank=16, seed=seed)
        return run_round(clients, global_adapter, round_id=0, aggregation="svd")

    r1 = one_run(42)
    r2 = one_run(42)
    for module in r1.adapter.target_modules:
        assert np.array_equal(
            r1.adapter.modules[module]["lora_A"], r2.adapter.modules[module]["lora_A"]
        )
        assert np.array_equal(
            r1.adapter.modules[module]["lora_B"], r2.adapter.modules[module]["lora_B"]
        )
