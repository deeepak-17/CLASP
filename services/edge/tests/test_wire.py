"""Edge <-> Cluster wire format (Integration Sprint · B3).

Unit-level proof of the bidirectional key map, the fp32 contract and the
structural checks. Runs on tiny synthetic tensors — CPU only, no cluster
service, no registry, no torch — plus one test against a real trained adapter
that skips itself when the (gitignored) weights are not on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edge import wire

REAL_ADAPTER = (Path(__file__).resolve().parents[1]
                / "artifacts" / "round1" / "client-flask" / "adapter")
LAYERS, MODULES, RANK, DIM = 3, wire.TARGET_MODULES, 4, 8


def tiny(convention: str = "peft", dtype=np.float32) -> dict:
    rng = np.random.default_rng(0)
    key = wire.peft_key if convention == "peft" else wire.cluster_key
    sd = {}
    for layer in range(LAYERS):
        for module in MODULES:
            sd[key(layer, module, "lora_A")] = rng.normal(size=(RANK, DIM)).astype(dtype)
            sd[key(layer, module, "lora_B")] = rng.normal(size=(DIM, RANK)).astype(dtype)
    return sd


# --------------------------------------------------------------------------- #
# key translation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key,expected", [
    ("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight", (3, "q_proj", "lora_A")),
    ("layers.3.q_proj.lora_A.weight", (3, "q_proj", "lora_A")),
    ("base_model.model.model.layers.23.self_attn.o_proj.lora_B.weight", (23, "o_proj", "lora_B")),
    ("model.layers.0.self_attn.v_proj.lora_B.weight", (0, "v_proj", "lora_B")),
])
def test_parse_key_accepts_both_conventions(key, expected):
    assert wire.parse_key(key) == expected


@pytest.mark.parametrize("key", [
    "base_model.model.model.q_proj.lora_A.weight",   # no layer index
    "layers.3.q_proj.lora_C.weight",                 # not an A/B factor
    "layers.3.q_proj.lora_A.bias",                   # not a weight
    "",
])
def test_parse_key_raises_rather_than_guessing(key):
    """A key the map cannot read is a key the aggregator would silently drop."""
    with pytest.raises(wire.WireFormatError):
        wire.parse_key(key)


def test_peft_to_cluster_to_peft_round_trips():
    original = tiny("peft")
    there = wire.to_cluster_keys(original)
    assert all(k.startswith("layers.") for k in there)
    back = wire.to_peft_keys(there)
    assert set(back) == set(original)
    for key, arr in original.items():
        np.testing.assert_array_equal(back[key], arr)


def test_cluster_to_peft_to_cluster_round_trips():
    original = tiny("cluster")
    back = wire.to_cluster_keys(wire.to_peft_keys(original))
    assert set(back) == set(original)
    for key, arr in original.items():
        np.testing.assert_array_equal(back[key], arr)


def test_translation_is_layer_generic_not_hardcoded():
    """24 layers, not just the one the sprint doc used as an example."""
    rng = np.random.default_rng(1)
    sd = {wire.cluster_key(i, "q_proj", "lora_A"): rng.normal(size=(2, 2)).astype("f4")
          for i in range(24)}
    sd.update({wire.cluster_key(i, "q_proj", "lora_B"): rng.normal(size=(2, 2)).astype("f4")
               for i in range(24)})
    peft = wire.to_peft_keys(sd)
    assert len(peft) == 48
    assert wire.peft_key(23, "q_proj", "lora_B") in peft


def test_translation_refuses_to_silently_overwrite_on_collision():
    sd = {
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": np.zeros((2, 2), "f4"),
        "layers.1.q_proj.lora_A.weight": np.ones((2, 2), "f4"),
    }
    with pytest.raises(wire.WireFormatError, match="collision"):
        wire.to_peft_keys(sd)


# --------------------------------------------------------------------------- #
# structure + dtype contract
# --------------------------------------------------------------------------- #
def test_describe_reads_shape_off_the_tensors():
    info = wire.describe(tiny())
    assert info["num_layers"] == LAYERS
    assert info["rank"] == RANK
    assert info["target_modules"] == MODULES
    assert info["n_tensors"] == 2 * len(MODULES) * LAYERS
    assert info["dtypes"] == ["float32"]


def test_describe_rejects_a_truncated_adapter():
    sd = tiny()
    sd.pop(wire.peft_key(2, "o_proj", "lora_B"))
    with pytest.raises(wire.WireFormatError):
        wire.describe(sd)


def test_describe_rejects_a_hole_in_the_layer_range():
    sd = {k: v for k, v in tiny().items() if wire.parse_key(k)[0] != 1}
    with pytest.raises(wire.WireFormatError, match="non-contiguous"):
        wire.describe(sd)


def test_as_fp32_casts_by_default_and_refuses_when_strict():
    sd = tiny(dtype=np.float16)
    assert all(v.dtype == np.float32 for v in wire.as_fp32(sd).values())
    with pytest.raises(wire.WireFormatError, match="fp32"):
        wire.as_fp32(sd, strict=True)


def test_encode_decode_round_trips_and_checks_length():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    payload = wire.encode_tensor("t", arr)
    assert payload["dtype"] == "float32" and payload["shape"] == [3, 4]
    np.testing.assert_array_equal(wire.decode_tensor(payload), arr)
    payload["shape"] = [3, 5]
    with pytest.raises(wire.WireFormatError):
        wire.decode_tensor(payload)


def test_serialize_is_byte_reproducible():
    """safetensors orders its header from a Rust hash map, so the same adapter
    serializes to different bytes — and a different sha256 — call to call.
    wire.serialize canonicalizes the header so that stops being true."""
    cfg = {"r": RANK, "lora_alpha": RANK, "target_modules": list(MODULES)}
    sd = tiny()
    blobs = {wire.serialize(sd, cfg) for _ in range(8)}
    assert len(blobs) == 1, "serialize() is not byte-stable"
    back_sd, back_cfg = wire.deserialize(blobs.pop())
    assert back_cfg == cfg
    for key, arr in sd.items():
        np.testing.assert_array_equal(back_sd[key], arr)


def test_canonicalize_rejects_a_blob_that_is_not_safetensors():
    with pytest.raises(wire.WireFormatError):
        wire.canonicalize_safetensors(b"nope")
    with pytest.raises(wire.WireFormatError):
        wire.canonicalize_safetensors((10 ** 9).to_bytes(8, "little") + b"{}")


def test_canonicalize_preserves_tensors_and_metadata():
    from safetensors.numpy import save

    sd = {wire.peft_key(0, "q_proj", p): np.arange(4, dtype=np.float32).reshape(2, 2)
          for p in ("lora_A", "lora_B")}
    raw = save(sd, metadata={"b": "2", "a": "1"})
    canon = wire.canonicalize_safetensors(raw)
    header_len = int.from_bytes(canon[:8], "little")
    header = json.loads(canon[8:8 + header_len])
    assert list(header) == sorted(header)          # sorted, including __metadata__
    assert header["__metadata__"] == {"a": "1", "b": "2"}
    back, _cfg = wire.deserialize(canon)
    for key, arr in sd.items():
        np.testing.assert_array_equal(back[key], arr)


# --------------------------------------------------------------------------- #
# the upload envelope
# --------------------------------------------------------------------------- #
def test_upload_payload_carries_what_the_cluster_needs():
    cfg = {"r": RANK, "lora_alpha": RANK, "target_modules": list(MODULES)}
    payload = wire.upload_payload(tiny(), cfg, client_id="client-flask",
                                  cluster_id="web", round_id=0, num_examples=170, seed=0)
    assert payload["client_id"] == "client-flask"
    assert payload["cluster_id"] == "web"
    assert payload["num_examples"] == 170
    assert payload["rank"] == RANK
    assert payload["num_layers"] == LAYERS
    assert list(payload["target_modules"]) == list(MODULES)
    assert len(payload["tensors"]) == 2 * len(MODULES) * LAYERS
    assert all(t["dtype"] == "float32" for t in payload["tensors"])
    json.dumps(payload)  # must be JSON-safe end to end


def test_upload_payload_folds_a_non_unit_scaling_into_lora_b():
    """The cluster reconstructs B @ A with no scaling, so an adapter whose
    lora_alpha != r must arrive pre-folded or it is silently mis-weighted."""
    sd = tiny()
    cfg = {"r": RANK, "lora_alpha": 4 * RANK, "target_modules": list(MODULES)}
    before = wire.delta_w(sd, cfg, 0, "q_proj")
    payload = wire.upload_payload(sd, cfg, client_id="c", cluster_id="web",
                                  round_id=0, num_examples=1)
    assert payload["alpha"] == float(RANK)          # scaling now exactly 1.0
    after = wire.delta_w(wire.tensors_from_payload(payload["tensors"]),
                         {"r": payload["rank"], "lora_alpha": payload["alpha"]},
                         0, "q_proj")
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)


def test_upload_payload_rejects_a_config_that_contradicts_the_tensors():
    with pytest.raises(wire.WireFormatError, match="disagrees"):
        wire.upload_payload(tiny(), {"r": RANK + 1, "lora_alpha": RANK},
                            client_id="c", cluster_id="web", round_id=0, num_examples=1)


def test_broadcast_without_a_config_refuses_rather_than_inventing_one():
    broadcast = {"rank": RANK, "num_layers": LAYERS, "peft_config": None,
                 "tensors": [wire.encode_tensor(k, v) for k, v in tiny().items()]}
    with pytest.raises(wire.WireFormatError, match="peft_config"):
        wire.broadcast_to_peft(broadcast)


# --------------------------------------------------------------------------- #
# the real trained adapter
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (REAL_ADAPTER / "adapter_model.safetensors").exists(),
                    reason="trained weights are gitignored; not on this machine")
def test_real_client_adapter_round_trips_through_both_conventions():
    """B3's acceptance case at full size: 192 tensors, 24 layers, r=16, fp32."""
    sd, cfg = wire.load_peft_dir(REAL_ADAPTER)
    info = wire.describe(sd)
    assert info["n_tensors"] == 192
    assert info["num_layers"] == 24
    assert info["rank"] == 16
    assert info["dtypes"] == ["float32"]
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 16   # PEFT scaling exactly 1.0

    back = wire.to_peft_keys(wire.to_cluster_keys(sd))
    assert set(back) == set(sd)
    for key, arr in sd.items():
        np.testing.assert_array_equal(back[key], arr)


@pytest.mark.skipif(not (REAL_ADAPTER / "adapter_model.safetensors").exists(),
                    reason="trained weights are gitignored; not on this machine")
def test_real_adapter_survives_serialize_deserialize_with_its_config():
    sd, cfg = wire.load_peft_dir(REAL_ADAPTER)
    blob = wire.serialize(sd, cfg)
    back_sd, back_cfg = wire.deserialize(blob)
    assert back_cfg == cfg          # config travels inside the safetensors header
    assert len(back_sd) == 192
    for key, arr in sd.items():
        np.testing.assert_array_equal(back_sd[key], arr)
