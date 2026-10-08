"""Composite builder (D6): alpha*dW_cluster + beta*dW_client as ONE adapter.

The builder concatenates LoRA factors along the rank axis with each part's
coefficient and scaling folded into its B block, so the stored composite's
B @ A is exactly the weighted sum of the parts' deltas. These tests check
that claim against the obvious reference (sum of coef * scale * B @ A).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from contracts import LoRAHyperParams
from registry.composite import CompositeError, PartSpec, build_composite
from safetensors.numpy import load, save

LAYERS, MODULES, R, D = 2, ("q_proj", "v_proj"), 4, 8


def _peft(layer: int, module: str, part: str) -> str:
    return f"base_model.model.model.layers.{layer}.self_attn.{module}.{part}.weight"


def _cluster_key(layer: int, module: str, part: str) -> str:
    return f"layers.{layer}.{module}.{part}.weight"


def make_adapter(seed: int, *, rank: int = R, lora_alpha: int = R, keyer=_peft,
                 embed_cfg: bool = True, modules=MODULES) -> bytes:
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(LAYERS):
        for m in modules:
            sd[keyer(layer, m, "lora_A")] = rng.standard_normal((rank, D)).astype(np.float32)
            sd[keyer(layer, m, "lora_B")] = rng.standard_normal((D, rank)).astype(np.float32)
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": lora_alpha,
           "target_modules": list(modules), "use_rslora": False}
    meta = {"adapter_config": json.dumps(cfg), "format": "pt"} if embed_cfg else None
    return save(sd, metadata=meta)


def _hp(rank: int = R, lora_alpha: int = R) -> LoRAHyperParams:
    return LoRAHyperParams(rank=rank, lora_alpha=lora_alpha, target_modules=MODULES)


def _deltas(payload: bytes, scale: float, keyer=_peft) -> dict[tuple[int, str], np.ndarray]:
    sd = load(payload)
    return {
        (layer, m): scale * (sd[keyer(layer, m, "lora_B")].astype(np.float64)
                             @ sd[keyer(layer, m, "lora_A")].astype(np.float64))
        for layer in range(LAYERS) for m in MODULES
    }


def test_composite_delta_equals_weighted_sum_of_parts():
    cluster, client = make_adapter(1, lora_alpha=8), make_adapter(2, lora_alpha=4)
    alpha, beta = 0.5, 1.0
    out = build_composite(PartSpec(cluster, _hp(lora_alpha=8), alpha),
                          PartSpec(client, _hp(lora_alpha=4), beta))
    want_c = _deltas(cluster, alpha * 8 / R)
    want_l = _deltas(client, beta * 4 / R)
    got = _deltas(out.payload, 1.0)  # composite scaling is pinned to 1.0
    for key, dw in got.items():
        np.testing.assert_allclose(dw, want_c[key] + want_l[key], rtol=1e-5, atol=1e-5)
    assert out.rank == 2 * R
    assert out.hparams.rank == 2 * R and out.hparams.lora_alpha == 2 * R
    assert out.hparams.alpha == alpha and out.hparams.beta == beta


def test_embedded_config_pins_scaling_to_one():
    out = build_composite(PartSpec(make_adapter(1), _hp(), 0.5),
                          PartSpec(make_adapter(2), _hp(), 1.0))
    header_len = int.from_bytes(out.payload[:8], "little")
    meta = json.loads(out.payload[8:8 + header_len])["__metadata__"]
    cfg = json.loads(meta["adapter_config"])
    assert cfg["r"] == cfg["lora_alpha"] == 2 * R
    assert cfg["use_rslora"] is False
    assert sorted(cfg["target_modules"]) == sorted(MODULES)


def test_zero_alpha_returns_client_bitwise():
    """alpha=0 prunes the cluster block: same delta, rank r, identical tensors."""
    client = make_adapter(2)
    out = build_composite(PartSpec(make_adapter(1), _hp(), 0.0), PartSpec(client, _hp(), 1.0))
    got, want = load(out.payload), load(client)
    assert out.rank == R
    assert set(got) == set(want)
    for k in want:
        assert np.array_equal(got[k], want[k])


def test_cluster_short_keys_are_mapped_onto_client_keys():
    """The cluster may publish `layers.<i>.<module>...` keys; output uses PEFT keys."""
    cluster = make_adapter(1, keyer=_cluster_key)
    client = make_adapter(2)
    out = build_composite(PartSpec(cluster, _hp(), 1.0), PartSpec(client, _hp(), 1.0))
    keys = set(load(out.payload))
    assert keys == set(load(client))
    got = _deltas(out.payload, 1.0)
    want_c, want_l = _deltas(cluster, 1.0, keyer=_cluster_key), _deltas(client, 1.0)
    for key in got:
        np.testing.assert_allclose(got[key], want_c[key] + want_l[key], rtol=1e-5, atol=1e-5)


def test_works_without_embedded_config_using_registry_hparams():
    cluster = make_adapter(1, embed_cfg=False)
    client = make_adapter(2, embed_cfg=False)
    out = build_composite(PartSpec(cluster, _hp(lora_alpha=8), 1.0),
                          PartSpec(client, _hp(lora_alpha=8), 1.0))
    got = _deltas(out.payload, 1.0)
    want_c, want_l = _deltas(cluster, 2.0), _deltas(client, 2.0)
    for key in got:
        np.testing.assert_allclose(got[key], want_c[key] + want_l[key], rtol=1e-5, atol=1e-5)


def test_deterministic_bytes():
    a = build_composite(PartSpec(make_adapter(1), _hp(), 0.5), PartSpec(make_adapter(2), _hp(), 1.0))
    b = build_composite(PartSpec(make_adapter(1), _hp(), 0.5), PartSpec(make_adapter(2), _hp(), 1.0))
    assert a.payload == b.payload


# --------------------------------------------------------------------------- #
# refusals — a composite that cannot be exact must not be stored
# --------------------------------------------------------------------------- #
def test_rejects_registry_vs_embedded_rank_mismatch():
    with pytest.raises(CompositeError, match="rank"):
        build_composite(PartSpec(make_adapter(1), _hp(rank=16), 1.0),
                        PartSpec(make_adapter(2), _hp(), 1.0))


def test_rejects_registry_vs_embedded_lora_alpha_mismatch():
    with pytest.raises(CompositeError, match="lora_alpha"):
        build_composite(PartSpec(make_adapter(1), _hp(lora_alpha=32), 1.0),
                        PartSpec(make_adapter(2), _hp(), 1.0))


def test_rejects_shape_mismatch_between_parts():
    cluster = make_adapter(1)
    sd = load(make_adapter(2))
    key = _peft(0, "q_proj", "lora_A")
    sd[key] = np.zeros((R, D + 1), dtype=np.float32)
    with pytest.raises(CompositeError, match="shape"):
        build_composite(PartSpec(cluster, _hp(), 1.0), PartSpec(save(sd), _hp(), 1.0))


def test_rejects_partial_module_overlap():
    """Cluster on {q, v}, client on {q}: v would keep rank R under a declared 2R."""
    client = make_adapter(2, modules=("q_proj",))
    hp_client = LoRAHyperParams(rank=R, lora_alpha=R, target_modules=("q_proj",))
    with pytest.raises(CompositeError, match="different modules.*only in cluster: 0.v_proj"):
        build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(client, hp_client, 1.0))


def test_rejects_disjoint_layers():
    sd = {k: v for k, v in load(make_adapter(2)).items() if ".layers.1." not in k}
    with pytest.raises(CompositeError, match="only in cluster: 1.q_proj"):
        build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(save(sd), _hp(), 1.0))


def test_pruned_part_may_cover_other_modules():
    """alpha=0 drops the cluster entirely, so its module set is irrelevant."""
    cluster = make_adapter(1, modules=("q_proj",))
    hp_cluster = LoRAHyperParams(rank=R, lora_alpha=R, target_modules=("q_proj",))
    out = build_composite(PartSpec(cluster, hp_cluster, 0.0), PartSpec(make_adapter(2), _hp(), 1.0))
    assert out.rank == R


def test_every_composite_tensor_matches_the_declared_rank():
    out = build_composite(PartSpec(make_adapter(1), _hp(), 0.5),
                          PartSpec(make_adapter(2), _hp(), 1.0))
    header_len = int.from_bytes(out.payload[:8], "little")
    cfg = json.loads(json.loads(out.payload[8:8 + header_len])["__metadata__"]["adapter_config"])
    for key, t in load(out.payload).items():
        assert (t.shape[0] if "lora_A" in key else t.shape[1]) == cfg["r"], key


def test_rejects_unpaired_lora_factor():
    sd = load(make_adapter(2))
    del sd[_peft(1, "v_proj", "lora_B")]
    with pytest.raises(CompositeError, match="lora_B"):
        build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(save(sd), _hp(), 1.0))


def test_rejects_bad_coefficients():
    with pytest.raises(CompositeError, match="coefficient"):
        build_composite(PartSpec(make_adapter(1), _hp(), float("nan")),
                        PartSpec(make_adapter(2), _hp(), 1.0))
    with pytest.raises(CompositeError, match="both"):
        build_composite(PartSpec(make_adapter(1), _hp(), 0.0), PartSpec(make_adapter(2), _hp(), 0.0))


def test_rslora_scaling_is_folded():
    sd = load(make_adapter(2))
    cfg = {"peft_type": "LORA", "r": R, "lora_alpha": R, "use_rslora": True,
           "target_modules": list(MODULES)}
    payload = save(sd, metadata={"adapter_config": json.dumps(cfg)})
    out = build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(payload, _hp(), 1.0))
    # rsLoRA scales by alpha/sqrt(r); the fold must use that scale, not alpha/r.
    got = _deltas(out.payload, 1.0)
    want_c = _deltas(make_adapter(1), 1.0)
    want_l = _deltas(payload, R / np.sqrt(R))
    for key in got:
        np.testing.assert_allclose(got[key], want_c[key] + want_l[key], rtol=1e-5, atol=1e-5)
