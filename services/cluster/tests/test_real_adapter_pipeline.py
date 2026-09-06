"""Cluster-side real-adapter round trip (Integration Sprint item 5):

    Edge-style LoRA adapter
       -> Cluster HTTP upload (POST /uploads)
       -> LoRAAdapter conversion (LoRAAdapter.from_state_dict, unchanged)
       -> SVD aggregation (POST /aggregate -> aggregate_svd, unchanged)
       -> aggregated Cluster adapter
       -> output loadable by edge.merge (PEFT directory: adapter_config.json
          + adapter_model.safetensors, via LoRAAdapter.to_peft_state_dict /
          to_peft_config)

HONESTY NOTE, read before trusting this file: this repository has no actual
trained adapter weights anywhere reachable from Cluster's side. `*.safetensors`
and `checkpoints/` are gitignored repo-wide, none exist in this working tree,
and none exist on any fetched branch, including origin/services/edge (whose
furthest commit is code, not weights). So "Edge-style" below means: real
dimensions (24 layers, q/k/v/o, rank 16, fp32) and the real PEFT key
convention edge.merge.py actually uses -- NOT literally-trained tensor
values, which do not exist anywhere this session can reach. hidden_size=2048
is an assumed, representative dimension for deepseek-coder-1.3b-base (not
pinned anywhere in this repo); it does not affect the correctness of the
wiring this test proves, since neither the contract nor the aggregation math
cares what the hidden size is. If real .safetensors files exist on a
teammate's machine, this same test should be re-pointed at them instead of
`_make_edge_style_adapter`.

The safetensors serialization itself is genuine: it uses the real
`safetensors` library (pure-numpy backend, no torch needed) to write and
read back an actual .safetensors file, so at least the file-format half of
"loadable by Edge merge code" is verified for real, not simulated.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster.adapter_format import TARGET_MODULES, LoRAAdapter
from cluster.schemas.messages import TensorPayload
from cluster.server import _state, app

safetensors_numpy = pytest.importorskip("safetensors.numpy")

NUM_LAYERS = 24
# 2048 (deepseek-coder-1.3b-base's real hidden size, not contract-pinned) made
# this test OOM-kill on the verification VM used to run it here: SVD across
# 24 layers x 4 modules x 3 clients on 2048x2048 delta_W matrices, computed
# both inside /aggregate and again independently for the comparison below,
# is several GB of transient float64 allocation on a machine with ~3.8GB RAM.
# 256 keeps every CONTRACT-relevant dimension real (24 layers, rank 16, fp32,
# q/k/v/o) and only shrinks the one dimension that is not part of the
# contract. Re-run with HIDDEN_SIZE = 2048 on a machine with more headroom to
# verify at the literal model width.
HIDDEN_SIZE = 256
RANK = 16
EDGE_BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
EDGE_LORA_ALPHA = 16.0  # edge.merge.CONTRACT_HYPERPARAMS["lora_alpha"]


def _make_edge_style_adapter(seed: int) -> LoRAAdapter:
    """24 layers x q/k/v/o, rank 16, fp32 -- real Edge dimensions and real
    PEFT key shape, non-trained values (see module docstring)."""
    rng = np.random.default_rng(seed)
    modules: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for layer in range(NUM_LAYERS):
        layer_modules: dict[str, dict[str, np.ndarray]] = {}
        for name in TARGET_MODULES:
            # PEFT default init: A ~ N(0, 1/r), B nonzero (a freshly
            # initialized adapter has B=0, which would make every client's
            # contribution and the aggregate indistinguishable from zero and
            # prove nothing about the aggregation actually running).
            a = (rng.normal(0.0, 1.0 / RANK, size=(RANK, HIDDEN_SIZE))).astype(np.float32)
            b = (rng.normal(0.0, 1.0 / HIDDEN_SIZE, size=(HIDDEN_SIZE, RANK))).astype(np.float32)
            layer_modules[name] = {"lora_A": a, "lora_B": b}
        modules[layer] = layer_modules
    return LoRAAdapter(
        rank=RANK, alpha=EDGE_LORA_ALPHA, target_modules=TARGET_MODULES,
        num_layers=NUM_LAYERS, modules=modules,
    )


def _reset_state():
    _state.uploads.clear()
    _state.round_id = 0
    _state.active = None
    _state.last_manifest = None


@pytest.fixture
def tmp_out_dir():
    d = tempfile.mkdtemp(prefix="clasp_cluster_peft_out_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_real_shape_adapter_survives_from_state_dict_conversion():
    """LoRAAdapter conversion step, in isolation, at real 24-layer/rank-16
    dimensions -- confirms the fix for Defect 2/3's read direction holds at
    full scale, not just the small synthetic sizes test_adapter_format.py uses."""
    edge_adapter = _make_edge_style_adapter(seed=1)
    peft_sd = edge_adapter.to_peft_state_dict()  # what a real Edge dump looks like
    assert len(peft_sd) == 2 * len(TARGET_MODULES) * NUM_LAYERS  # 192 tensors

    recovered = LoRAAdapter.from_state_dict(
        peft_sd, rank=RANK, alpha=EDGE_LORA_ALPHA,
        target_modules=TARGET_MODULES, num_layers=NUM_LAYERS,
    )
    for layer in range(NUM_LAYERS):
        for name in TARGET_MODULES:
            for part in ("lora_A", "lora_B"):
                np.testing.assert_array_equal(
                    edge_adapter.modules[layer][name][part],
                    recovered.modules[layer][name][part],
                )
            assert recovered.modules[layer][name]["lora_A"].dtype == np.float32
            assert recovered.modules[layer][name]["lora_B"].dtype == np.float32


def test_full_cluster_http_pipeline_with_real_shape_adapters(tmp_out_dir):
    """The complete path item 5 asks for, run for real over the FastAPI
    TestClient (a real ASGI request/response cycle, not a direct function
    call) with three 192-tensor, rank-16, fp32 client adapters."""
    _reset_state()
    client = TestClient(app)

    edge_adapters = [_make_edge_style_adapter(seed=s) for s in (101, 102, 103)]
    weights = [80.0, 120.0, 40.0]

    for i, adapter in enumerate(edge_adapters):
        sd = adapter.to_peft_state_dict()  # real Edge upload shape, over HTTP
        payload = {
            "client_id": f"web-client-{i}",
            "round_id": 0,
            "rank": RANK,
            "target_modules": list(TARGET_MODULES),
            "alpha": EDGE_LORA_ALPHA,
            "num_layers": NUM_LAYERS,
            "num_examples": int(weights[i]),
            "tensors": [
                TensorPayload.from_numpy(name, arr).model_dump(mode="json")
                for name, arr in sd.items()
            ],
        }
        resp = client.post("/uploads", json=payload)
        assert resp.status_code == 201, resp.text
        assert resp.json()["tensors_received"] == 192

    agg_resp = client.post(
        "/aggregate", json={"aggregation": "svd", "include_manifest": True}
    )
    assert agg_resp.status_code == 200, agg_resp.text
    broadcast = agg_resp.json()
    assert broadcast["num_clients"] == 3
    assert broadcast["rank"] == RANK
    assert broadcast["num_layers"] == NUM_LAYERS
    assert len(broadcast["tensors"]) == 192

    # --- independent numeric check: recompute the SVD aggregate directly
    # from the same three adapters (bypassing HTTP) and compare against what
    # the endpoint returned. This is the "compare against
    # artifacts/round1_out/cluster-web" check from the sprint doc's Thursday
    # gate, minus the artifact that does not exist in this repo -- the
    # closest available substitute is agreement with an independently
    # computed reference over the same real inputs.
    from cluster.aggregation import aggregate_svd

    reference = aggregate_svd(iter(edge_adapters), weights, rank=RANK)
    for layer in range(NUM_LAYERS):
        for name in TARGET_MODULES:
            expected_delta = reference.delta_w(name, layer)
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{name}"
            got_tensors = {t["name"]: t for t in broadcast["tensors"]}
            got_a = TensorPayload(**got_tensors[f"{prefix}.lora_A.weight"]).to_numpy()
            got_b = TensorPayload(**got_tensors[f"{prefix}.lora_B.weight"]).to_numpy()
            got_delta = got_b @ got_a
            np.testing.assert_allclose(got_delta, expected_delta, rtol=1e-5, atol=1e-6)

    # reconstruction-error manifest is present and finite for a real-scale run
    manifest = client.get("/aggregate/manifest").json()
    assert manifest["num_clients"] == 3
    assert np.isfinite(manifest["svd_reconstruction_error"]["mean"])

    # --- write a genuine PEFT directory from the HTTP response and read it
    # back with the real safetensors library (pure-numpy backend, no torch) ---
    peft_config = broadcast["peft_config"]
    assert peft_config["r"] == RANK
    assert peft_config["lora_alpha"] == EDGE_LORA_ALPHA
    assert set(peft_config["target_modules"]) == set(TARGET_MODULES)
    assert peft_config["base_model_name_or_path"] == EDGE_BASE_MODEL

    (tmp_out_dir / "adapter_config.json").write_text(json.dumps(peft_config, indent=2))
    tensor_dict = {
        t["name"]: TensorPayload(**t).to_numpy() for t in broadcast["tensors"]
    }
    safetensors_numpy.save_file(tensor_dict, str(tmp_out_dir / "adapter_model.safetensors"))

    # read back exactly as edge.merge.load_adapter would (its two file reads,
    # reimplemented here with the numpy backend since torch is unavailable)
    loaded_cfg = json.loads((tmp_out_dir / "adapter_config.json").read_text())
    loaded_tensors = safetensors_numpy.load_file(str(tmp_out_dir / "adapter_model.safetensors"))

    assert loaded_cfg == peft_config
    assert len(loaded_tensors) == 192
    for name, arr in loaded_tensors.items():
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, tensor_dict[name])
    # edge.merge.module_prefixes' exact key-splitting rule, applied to what
    # we just read back off disk
    prefixes = sorted({k.rsplit(".lora_A", 1)[0] for k in loaded_tensors if ".lora_A" in k})
    assert len(prefixes) == NUM_LAYERS * len(TARGET_MODULES)
    assert all(p.startswith("base_model.model.model.layers.") and ".self_attn." in p
               for p in prefixes)
