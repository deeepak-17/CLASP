"""LoRAAdapter <-> real PEFT key/config conversion (Integration Sprint
Defect 3, write direction).

edge.merge.py (services/edge, branch origin/services/edge — read directly
from git, not imported: that module hard-requires torch + safetensors.torch
at import time, and is out of Cluster's scope to modify or depend on)
expects a PEFT adapter directory: adapter_config.json matching
CONTRACT_HYPERPARAMS, plus adapter_model.safetensors keyed like
'base_model.model.model.layers.<i>.self_attn.<module>.lora_<A|B>.weight'
(see edge.merge.PREFIXES / edge.merge.module_prefixes in its own test file,
services/edge/tests/test_merge.py).

These tests pin those exact literal values/shapes (copied from the real
source, not guessed) and check LoRAAdapter.to_peft_state_dict() /
to_peft_config() against them directly, since importing edge.merge itself
is not possible without torch in this environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster.adapter_format import TARGET_MODULES, LoRAAdapter, random_adapter

# Copied verbatim from services/edge/src/edge/merge.py (origin/services/edge)
EDGE_CONTRACT_HYPERPARAMS = {
    "r": 16,
    "lora_alpha": 16,
    "target_modules": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "base_model_name_or_path": "deepseek-ai/deepseek-coder-1.3b-base",
}
EDGE_STRUCTURAL_FIELDS = ("peft_type", "use_rslora", "use_dora", "fan_in_fan_out", "lora_bias")


def _edge_module_prefixes(sd: dict[str, np.ndarray]) -> list[str]:
    """Reimplements edge.merge.module_prefixes's key-splitting logic exactly
    (pure string manipulation, no torch needed) so this test checks against
    edge's real parsing rule rather than a guess at it."""
    return sorted({k.rsplit(".lora_A", 1)[0] for k in sd if ".lora_A" in k})


def test_to_peft_state_dict_uses_full_edge_key_convention():
    adapter = random_adapter(32, 32, rank=16, num_layers=24, seed=1)
    sd = adapter.to_peft_state_dict()

    assert len(sd) == 2 * len(TARGET_MODULES) * 24
    for layer in range(24):
        for name in TARGET_MODULES:
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{name}"
            assert f"{prefix}.lora_A.weight" in sd
            assert f"{prefix}.lora_B.weight" in sd

    # edge.merge's own prefix-extraction must find exactly one prefix per
    # (layer, module) pair, with the '.self_attn.' segment intact
    prefixes = _edge_module_prefixes(sd)
    assert len(prefixes) == 24 * len(TARGET_MODULES)
    assert all(".self_attn." in p and p.startswith("base_model.model.model.layers.") for p in prefixes)


def test_to_peft_state_dict_shapes_match_lora_a_b_convention():
    """edge.merge._ab reads lora_A as (r, in) and lora_B as (out, r) off the
    same keys — verify that convention survives the conversion."""
    adapter = random_adapter(64, 48, rank=16, num_layers=1, seed=2)
    sd = adapter.to_peft_state_dict()
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    a, b = sd[f"{prefix}.lora_A.weight"], sd[f"{prefix}.lora_B.weight"]
    assert a.shape == (16, 64)  # (r, in_features)
    assert b.shape == (48, 16)  # (out_features, r)


def test_to_peft_config_matches_edge_contract_hyperparams():
    adapter = random_adapter(
        32, 32, rank=16, num_layers=24, alpha=16.0,
        target_modules=TARGET_MODULES, seed=3,
    )
    cfg = adapter.to_peft_config()

    assert cfg["r"] == EDGE_CONTRACT_HYPERPARAMS["r"]
    assert cfg["lora_alpha"] == EDGE_CONTRACT_HYPERPARAMS["lora_alpha"]
    assert set(cfg["target_modules"]) == EDGE_CONTRACT_HYPERPARAMS["target_modules"]
    assert cfg["base_model_name_or_path"] == EDGE_CONTRACT_HYPERPARAMS["base_model_name_or_path"]
    for field in EDGE_STRUCTURAL_FIELDS:
        assert field in cfg
    assert cfg["peft_type"] == "LORA"
    assert cfg["use_rslora"] is False
    assert cfg["use_dora"] is False
    assert cfg["fan_in_fan_out"] is False
    assert cfg["lora_bias"] is False


def test_to_peft_config_default_alpha_diverges_from_edge_contract():
    """Documents the cross-team mismatch found while inspecting the repo:
    Cluster's own DEFAULT_ALPHA (32.0) is NOT edge's contract lora_alpha
    (16). This is not a Cluster bug to silently patch — it is a value the
    caller must supply correctly (alpha=16.0) to produce an edge-loadable
    config; this test pins that the method is honest about the adapter's
    *actual* alpha rather than hardcoding edge's 16."""
    adapter = random_adapter(8, 8, rank=16, num_layers=1, seed=4)  # default alpha=32.0
    cfg = adapter.to_peft_config()
    assert cfg["lora_alpha"] == pytest.approx(32.0)
    assert cfg["lora_alpha"] != EDGE_CONTRACT_HYPERPARAMS["lora_alpha"]


def test_from_state_dict_still_parses_the_output_of_to_peft_state_dict():
    """Full round trip: LoRAAdapter -> real PEFT keys -> LoRAAdapter, values
    preserved exactly. Confirms the read direction (already working before
    this sprint) and the new write direction agree with each other."""
    original = random_adapter(16, 16, rank=8, num_layers=4, seed=5)
    for layer in original.layer_indices:
        for name in original.target_modules:
            original.modules[layer][name]["lora_B"] = np.random.default_rng(
                layer * 10
            ).normal(size=original.modules[layer][name]["lora_B"].shape).astype(np.float32)

    peft_sd = original.to_peft_state_dict()
    recovered = LoRAAdapter.from_state_dict(
        peft_sd, rank=8, alpha=original.alpha, num_layers=4, target_modules=TARGET_MODULES
    )
    for layer in original.layer_indices:
        for name in original.target_modules:
            for part in ("lora_A", "lora_B"):
                np.testing.assert_array_equal(
                    original.modules[layer][name][part],
                    recovered.modules[layer][name][part],
                )
