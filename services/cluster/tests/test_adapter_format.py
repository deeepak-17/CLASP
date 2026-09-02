"""Tests for cluster.adapter_format: the LoRAAdapter <-> ndarray/state_dict
round trip (Week 2 Thu).

Before this file, to_ndarrays/from_ndarrays/to_state_dict/from_state_dict
were only exercised indirectly (via test_aggregation.py / test_demo.py), and
always with single-layer synthetic adapters — so a bug in how layers are
keyed (see adapter_format.py's module docstring: modules used to be a flat
{module_name: pair} dict with no layer index at all) had no test that could
catch it. These tests build multi-layer adapters explicitly and round-trip
them through every wire/storage format.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster.adapter_format import (
    TARGET_MODULES,
    AdapterFormatError,
    LoRAAdapter,
    random_adapter,
)


def _peft_style_state_dict(adapter: LoRAAdapter) -> dict[str, np.ndarray]:
    """Keys shaped like a real PEFT dump, e.g.
    'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight' —
    deliberately NOT the bare 'layers.<i>.<module>.lora_A.weight' that
    ``to_state_dict()`` itself emits, to prove ``from_state_dict()`` tolerates
    a realistic wrapper prefix around the layer segment.
    """
    sd: dict[str, np.ndarray] = {}
    for layer in adapter.layer_indices:
        for name in adapter.target_modules:
            for part in ("lora_A", "lora_B"):
                key = f"base_model.model.model.layers.{layer}.self_attn.{name}.{part}.weight"
                sd[key] = adapter.modules[layer][name][part]
    return sd


def _distinguish_layers(adapter: LoRAAdapter, seed: int) -> None:
    """Overwrite lora_B with per-layer-distinct values so a bug that aliases
    or mixes up layers (e.g. every layer silently sharing layer 0's pair)
    shows up as a value mismatch, not just a shape mismatch."""
    rng = np.random.default_rng(seed)
    out_features = next(iter(adapter.modules[0].values()))["lora_B"].shape[0]
    for layer in adapter.layer_indices:
        for name in adapter.target_modules:
            adapter.modules[layer][name]["lora_B"] = (
                rng.normal(size=(out_features, adapter.rank)).astype(np.float32)
                + layer * 100.0
            )


def test_random_adapter_multi_layer_has_one_pair_per_layer():
    adapter = random_adapter(32, 32, rank=8, num_layers=6, seed=0)
    adapter.validate()
    assert adapter.num_layers == 6
    assert set(adapter.modules) == set(range(6))
    for layer in range(6):
        assert set(adapter.modules[layer]) == set(TARGET_MODULES)
    # every layer's lora_A is independently sampled, not the same array reused
    a0 = adapter.modules[0]["q_proj"]["lora_A"]
    a1 = adapter.modules[1]["q_proj"]["lora_A"]
    assert not np.array_equal(a0, a1)


def test_ndarrays_round_trip_multi_layer():
    original = random_adapter(24, 24, rank=4, num_layers=5, seed=1)
    _distinguish_layers(original, seed=2)

    arrays = original.to_ndarrays()
    assert len(arrays) == 2 * len(TARGET_MODULES) * 5

    recovered = LoRAAdapter.from_ndarrays(
        arrays, rank=4, num_layers=5, target_modules=TARGET_MODULES
    )
    for layer in original.layer_indices:
        for name in original.target_modules:
            for part in ("lora_A", "lora_B"):
                np.testing.assert_array_equal(
                    original.modules[layer][name][part],
                    recovered.modules[layer][name][part],
                )
    # layer 1's data must not have landed in layer 0's slot (or vice versa)
    assert not np.array_equal(
        recovered.modules[0]["q_proj"]["lora_B"], recovered.modules[1]["q_proj"]["lora_B"]
    )


def test_state_dict_round_trip_multi_layer_realistic_keys():
    """A round trip through PEFT-shaped, multi-layer keys must recover each
    layer's own pair — the exact case a flat {module: pair} dict (no layer
    index) cannot represent."""
    original = random_adapter(16, 16, rank=4, num_layers=3, seed=3)
    _distinguish_layers(original, seed=4)

    sd = _peft_style_state_dict(original)
    assert len(sd) == 2 * len(TARGET_MODULES) * 3

    recovered = LoRAAdapter.from_state_dict(
        sd, rank=4, num_layers=3, target_modules=TARGET_MODULES
    )
    for layer in original.layer_indices:
        for name in original.target_modules:
            np.testing.assert_array_equal(
                original.modules[layer][name]["lora_B"],
                recovered.modules[layer][name]["lora_B"],
            )
    b0 = recovered.modules[0]["q_proj"]["lora_B"]
    b1 = recovered.modules[1]["q_proj"]["lora_B"]
    assert not np.array_equal(b0, b1)


def test_to_state_dict_from_state_dict_self_round_trip_multi_layer():
    original = random_adapter(12, 12, rank=4, num_layers=4, seed=5)
    _distinguish_layers(original, seed=6)

    sd = original.to_state_dict()
    assert len(sd) == 2 * len(TARGET_MODULES) * 4
    recovered = LoRAAdapter.from_state_dict(sd, rank=4, num_layers=4)

    for layer in original.layer_indices:
        for name in original.target_modules:
            for part in ("lora_A", "lora_B"):
                np.testing.assert_array_equal(
                    original.modules[layer][name][part],
                    recovered.modules[layer][name][part],
                )


def test_from_state_dict_single_layer_accepts_legacy_layerless_keys():
    """A single-layer adapter (num_layers=1) must still accept old-style
    keys with no 'layers.<i>.' segment at all, read as layer 0."""
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=7)
    legacy_sd = {
        f"{name}.{part}.weight": adapter.modules[0][name][part]
        for name in adapter.target_modules
        for part in ("lora_A", "lora_B")
    }
    recovered = LoRAAdapter.from_state_dict(legacy_sd, rank=4, num_layers=1)
    for name in adapter.target_modules:
        for part in ("lora_A", "lora_B"):
            np.testing.assert_array_equal(
                adapter.modules[0][name][part], recovered.modules[0][name][part]
            )


def test_from_state_dict_rejects_duplicate_key_for_same_layer_and_module():
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=8)
    sd = adapter.to_state_dict()
    sd["extra.layers.0.q_proj.lora_A.weight"] = adapter.modules[0]["q_proj"]["lora_A"]
    with pytest.raises(AdapterFormatError):
        LoRAAdapter.from_state_dict(sd, rank=4, num_layers=1)


def test_from_state_dict_rejects_layer_outside_declared_range():
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=9)
    sd = adapter.to_state_dict()
    sd["layers.5.q_proj.lora_A.weight"] = adapter.modules[0]["q_proj"]["lora_A"]
    with pytest.raises(AdapterFormatError):
        LoRAAdapter.from_state_dict(sd, rank=4, num_layers=1)


def test_from_state_dict_multi_layer_key_without_layer_segment_rejected():
    adapter = random_adapter(8, 8, rank=4, num_layers=2, seed=10)
    sd = adapter.to_state_dict()
    sd["q_proj.lora_A.weight"] = adapter.modules[0]["q_proj"]["lora_A"]
    with pytest.raises(AdapterFormatError):
        LoRAAdapter.from_state_dict(sd, rank=4, num_layers=2)


def test_from_ndarrays_wrong_length_raises():
    with pytest.raises(AdapterFormatError):
        LoRAAdapter.from_ndarrays([np.zeros((4, 4), dtype=np.float32)], rank=4, num_layers=2)


def test_validate_rejects_mismatched_rank_in_a_non_zero_layer():
    """The bug this file guards against would only show up past layer 0 —
    make sure validation actually looks at every layer, not just the first."""
    adapter = random_adapter(8, 8, rank=4, num_layers=2, seed=11)
    adapter.modules[1]["q_proj"]["lora_A"] = np.zeros((3, 8), dtype=np.float32)
    with pytest.raises(AdapterFormatError):
        adapter.validate()


def test_delta_w_defaults_to_layer_zero_and_matches_explicit_layer():
    adapter = random_adapter(8, 8, rank=4, num_layers=2, seed=12)
    _distinguish_layers(adapter, seed=13)
    np.testing.assert_array_equal(
        adapter.delta_w("q_proj"), adapter.delta_w("q_proj", layer=0)
    )
    assert not np.array_equal(adapter.delta_w("q_proj", layer=0), adapter.delta_w("q_proj", layer=1))
