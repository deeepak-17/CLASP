"""End-to-end loop across Cluster and the State Registry (P4, Integration
Sprint item B8: "a CPU-only end-to-end test over tiny synthetic adapters that
exercises all four seams in-process").

Edge is not imported here — it needs torch/transformers/peft, which this test
deliberately avoids so it stays CPU-only and fast. Instead a few
`cluster.adapter_format.random_adapter()` instances stand in for trained
client adapters, exactly the way `services/cluster`'s own HTTP tests already
do. Nothing about the seams below depends on the tensors being real trained
weights — only on the wiring between services being correct.

Seams exercised (naming per Phase-2-Panel-1/CLASP_Integration_Sprint.md figure 2):

    A  Edge -> Cluster    POST /uploads (x3 synthetic clients)
    B  Cluster -> Registry POST /adapters/{name}/versions
                           (a real ClusterAdapterBroadcast, serialized to a
                           real safetensors blob)
    C1 Registry -> Edge    GET /adapters/{name}/active + .../versions/{v}/file
                           — pull the cluster delta back out and reassemble a
                           loadable PEFT config from JUST the two responses
                           (Defect 4: "the registry returns tensors without
                           their config")
    C2 Edge -> Registry    POST /adapters/{name}/promote — an EvalResult in, a
                           PROMOTE or ROLLBACK decision out. Both outcomes are
                           exercised over this real HTTP seam, not only
                           asserted against `registry.promotion.decide`
                           in isolation.

No aggregation or promotion math is reimplemented here: every step calls
straight into `cluster.server` / `cluster.aggregation` and `registry.app` /
`registry.promotion`, unchanged. Seam A/B's own internals (validation, error
paths, reconstruction-error manifests) already have dedicated coverage in
`services/cluster/tests/` and `services/registry/tests/` — this file's only
job is proving the three services agree with each other over a live HTTP
round trip, which none of those per-service suites can see on their own.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from cluster.adapter_format import TARGET_MODULES, random_adapter
from cluster.schemas.messages import TensorPayload
from fastapi.testclient import TestClient

import cluster.server as cluster_server
import registry.app as registry_appmod

safetensors_numpy = pytest.importorskip("safetensors.numpy")

RANK = 4
IN_FEATURES = OUT_FEATURES = 8
NUM_LAYERS = 2
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"  # project-wide constant, not per-adapter


@pytest.fixture
def cluster_client() -> TestClient:
    cluster_server._state.uploads.clear()
    cluster_server._state.round_id = 0
    cluster_server._state.active = None
    cluster_server._state.last_manifest = None
    return TestClient(cluster_server.app)


@pytest.fixture
def registry_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry_e2e"))
    registry_appmod._store = None  # force re-read of CLASP_REGISTRY_DATA
    return TestClient(registry_appmod.app)


def _upload_payload(adapter, client_id: str, round_id: int = 0, num_examples: int = 100) -> dict:
    sd = adapter.to_state_dict()
    return {
        "client_id": client_id,
        "round_id": round_id,
        "rank": adapter.rank,
        "target_modules": list(adapter.target_modules),
        "alpha": adapter.alpha,
        "num_layers": adapter.num_layers,
        "num_examples": num_examples,
        "tensors": [
            TensorPayload.from_numpy(name, arr).model_dump(mode="json")
            for name, arr in sd.items()
        ],
    }


def _eval_body(name: str, version: int, *, edit_similarity: float, pass_at_1: float) -> dict:
    """A synthetic EvalResult + baseline_guard, shaped exactly as
    `registry.app._eval_result_from` parses it."""
    return {
        "eval": {
            "adapter": {"name": name, "version": version, "kind": "cluster", "cluster_id": name},
            "in_project": {
                "edit_similarity": edit_similarity, "exact_match": 0.3,
                "perplexity": 3.1, "n_examples": 40,
            },
            "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": pass_at_1}}],
            "baseline_in_project": {
                "edit_similarity": 0.50, "exact_match": 0.25,
                "perplexity": 3.4, "n_examples": 40,
            },
            "baseline_noise_band": 0.01,
            "seed": 0,
        },
        "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.31}}],
    }


def test_full_loop_edge_to_cluster_to_registry_and_back(cluster_client, registry_client):
    # ---- Seam A: three synthetic clients upload to Cluster -----------------
    clients = [
        random_adapter(IN_FEATURES, OUT_FEATURES, rank=RANK, num_layers=NUM_LAYERS, seed=s)
        for s in (1, 2, 3)
    ]
    weights = [30.0, 50.0, 20.0]
    for i, adapter in enumerate(clients):
        # random_adapter's default lora_B is all-zero; give it real signal so
        # aggregation isn't just summing zeros (matches cluster's own tests).
        rng = np.random.default_rng(i)
        for layer in adapter.layer_indices:
            for name in adapter.target_modules:
                adapter.modules[layer][name]["lora_B"] = rng.normal(
                    size=adapter.modules[layer][name]["lora_B"].shape
                ).astype(np.float32)
        payload = _upload_payload(adapter, f"client-{i}", num_examples=int(weights[i]))
        resp = cluster_client.post("/uploads", json=payload)
        assert resp.status_code == 201, resp.text

    agg_resp = cluster_client.post("/aggregate", json={"aggregation": "svd"})
    assert agg_resp.status_code == 200, agg_resp.text
    broadcast = agg_resp.json()
    assert broadcast["num_clients"] == 3
    name = broadcast["cluster_id"]

    # ---- Seam B: Cluster -> Registry, a real safetensors blob --------------
    tensor_dict = {t["name"]: TensorPayload(**t).to_numpy() for t in broadcast["tensors"]}
    payload_bytes = safetensors_numpy.save(tensor_dict)
    meta = {
        "kind": "cluster",
        "hparams": {
            "rank": broadcast["rank"],
            "lora_alpha": int(broadcast["alpha"]),
            "target_modules": broadcast["target_modules"],
        },
        "aggregation": "svd_exact",
        "round": broadcast["round_id"],
        "seed": 0,
        "cluster_id": name,
        "source_clients": [f"client-{i}" for i in range(3)],
        "set_active": True,
    }
    save_resp = registry_client.post(
        f"/adapters/{name}/versions",
        files={"file": ("adapter.safetensors", payload_bytes, "application/octet-stream")},
        data={"meta": json.dumps(meta)},
    )
    assert save_resp.status_code == 201, save_resp.text
    assert save_resp.json()["ref"]["version"] == 1

    # ---- Seam C1: Registry -> Edge, pull the cluster delta back out --------
    active_meta = registry_client.get(f"/adapters/{name}/active").json()
    assert active_meta["ref"]["version"] == 1
    assert active_meta["source_clients"] == meta["source_clients"]

    file_resp = registry_client.get(f"/adapters/{name}/versions/1/file")
    assert file_resp.status_code == 200
    recovered = safetensors_numpy.load(file_resp.content)
    for k, v in tensor_dict.items():
        np.testing.assert_array_equal(recovered[k], v)

    # Defect 4 ("the registry returns tensors without their config"):
    # reassemble a loadable PEFT adapter_config.json from JUST the two
    # registry responses above (metadata's hparams + the downloaded tensors),
    # nothing else — proving the reassembly the merge path needs is possible
    # with data the registry already stores, no registry change required.
    reassembled_config = {
        "peft_type": "LORA",
        "r": active_meta["hparams"]["rank"],
        "lora_alpha": active_meta["hparams"]["lora_alpha"],
        "target_modules": active_meta["hparams"]["target_modules"],
        "base_model_name_or_path": BASE_MODEL,
        "inference_mode": True,
        "task_type": "CAUSAL_LM",
    }
    assert reassembled_config["r"] == RANK
    assert set(reassembled_config["target_modules"]) == set(TARGET_MODULES)

    # ---- Seam C2: Edge -> Registry, PROMOTE reachable -----------------------
    promote_resp = registry_client.post(
        f"/adapters/{name}/promote",
        json=_eval_body(name, version=1, edit_similarity=0.62, pass_at_1=0.30),
    )
    assert promote_resp.status_code == 200, promote_resp.text
    assert promote_resp.json()["action"] == "promote"

    # ---- a second round that regresses -> ROLLBACK, same real seam ---------
    save2 = registry_client.post(
        f"/adapters/{name}/versions",
        files={"file": ("adapter.safetensors", payload_bytes, "application/octet-stream")},
        data={"meta": json.dumps(meta)},
    )
    assert save2.status_code == 201, save2.text
    assert save2.json()["ref"]["version"] == 2

    rollback_resp = registry_client.post(
        f"/adapters/{name}/promote",
        json=_eval_body(name, version=2, edit_similarity=0.40, pass_at_1=0.20),
    )
    assert rollback_resp.status_code == 200, rollback_resp.text
    decision = rollback_resp.json()
    assert decision["action"] == "rollback"
    assert decision["active_version_after"] == 1

    # the active pointer actually moved back
    assert registry_client.get(f"/adapters/{name}/active").json()["ref"]["version"] == 1

    # the audit trail carries both decisions, in order, non-destructively
    trail = registry_client.get(f"/adapters/{name}/promotions").json()["decisions"]
    assert [d["action"] for d in trail] == ["promote", "rollback"]
