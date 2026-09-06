"""HTTP service tests (Integration Sprint B2): the FastAPI app in
cluster.server, exercised end to end with FastAPI's TestClient.

Proves the loop the sprint doc calls Seam A + the Cluster-side half of the
loop back out: POST /uploads (one call per client) -> POST /aggregate
(calls the EXISTING aggregate_svd, not reimplemented) -> GET
/adapters/cluster/active (the aggregated adapter, in PEFT-loadable key form).

Deliberately small, fast synthetic adapters here (few layers, small dims) —
this file is about proving the *wiring* is correct (validation, HTTP status
codes, round bookkeeping, response shape). The realistic 24-layer/rank-16/
fp32 proof lives in test_real_adapter_pipeline.py.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from cluster.adapter_format import (
    DEFAULT_ALPHA,
    TARGET_MODULES,
    LoRAAdapter,
    random_adapter,
)
from cluster.schemas.messages import TensorPayload
from cluster.server import _state, app


def _upload_payload(adapter: LoRAAdapter, client_id: str, round_id: int = 0,
                     num_examples: int = 100, peft_style: bool = False) -> dict:
    if peft_style:
        sd = adapter.to_peft_state_dict()
    else:
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


def _reset_state():
    _state.uploads.clear()
    _state.round_id = 0
    _state.active = None
    _state.last_manifest = None


def test_healthz():
    _reset_state()
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "cluster"
    assert body["pending_uploads"] == 0
    assert body["has_active_adapter"] is False


def test_upload_rejects_wrong_tensor_count():
    """Defect 2 regression: a malformed upload must fail cleanly (422), not
    silently accept a partial adapter."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(8, 8, rank=4, num_layers=2, seed=1)
    payload = _upload_payload(adapter, "bad-client")
    payload["tensors"] = payload["tensors"][:-1]  # drop one tensor
    resp = client.post("/uploads", json=payload)
    assert resp.status_code == 422


def test_aggregate_without_uploads_returns_400():
    _reset_state()
    client = TestClient(app)
    resp = client.post("/aggregate", json={})
    assert resp.status_code == 400


def test_active_before_any_aggregate_returns_404():
    _reset_state()
    client = TestClient(app)
    resp = client.get("/adapters/cluster/active")
    assert resp.status_code == 404


def test_full_upload_aggregate_download_loop_short_key_form():
    """Seam A end to end using the short internal key convention."""
    _reset_state()
    client = TestClient(app)

    adapters = [random_adapter(16, 16, rank=4, num_layers=3, seed=s) for s in (10, 11, 12)]
    for i, adapter in enumerate(adapters):
        # give every client's lora_B some non-zero signal so aggregation is
        # not just summing zeros (random_adapter's default B is all-zero)
        rng = np.random.default_rng(i)
        for layer in adapter.layer_indices:
            for name in adapter.target_modules:
                adapter.modules[layer][name]["lora_B"] = rng.normal(
                    size=adapter.modules[layer][name]["lora_B"].shape
                ).astype(np.float32)
        payload = _upload_payload(adapter, f"client-{i}", round_id=0, num_examples=50 * (i + 1))
        resp = client.post("/uploads", json=payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["client_id"] == f"client-{i}"
        assert body["tensors_received"] == 2 * len(TARGET_MODULES) * 3

    health = client.get("/healthz").json()
    assert health["pending_uploads"] == 3

    agg_resp = client.post(
        "/aggregate", json={"aggregation": "svd", "include_manifest": True}
    )
    assert agg_resp.status_code == 200, agg_resp.text
    broadcast = agg_resp.json()
    assert broadcast["num_clients"] == 3
    assert broadcast["aggregation"] == "svd"
    assert broadcast["rank"] == 4
    assert broadcast["num_layers"] == 3
    assert broadcast["peft_config"]["r"] == 4
    assert broadcast["peft_config"]["target_modules"] == list(TARGET_MODULES)
    # tensor names must be the FULL PEFT convention, not the short form
    names = [t["name"] for t in broadcast["tensors"]]
    assert any("self_attn" in n and n.startswith("base_model.model.model.layers.") for n in names)
    assert len(names) == 2 * len(TARGET_MODULES) * 3

    # buffer cleared, round advanced
    health = client.get("/healthz").json()
    assert health["pending_uploads"] == 0
    assert health["round_id"] == 1
    assert health["has_active_adapter"] is True

    manifest = client.get("/aggregate/manifest").json()
    assert manifest["num_clients"] == 3
    assert "mean" in manifest["svd_reconstruction_error"]
    assert manifest["svd_reconstruction_error"]["mean"] >= 0.0

    active = client.get("/adapters/cluster/active").json()
    assert active == broadcast


def test_upload_accepts_real_peft_key_convention_over_http():
    """The other half of Defect 3: an upload using the FULL PEFT key form
    (as a real Edge adapter would ship) must be accepted over the wire, not
    just by calling LoRAAdapter.from_state_dict() directly in a unit test."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(12, 12, rank=4, num_layers=2, alpha=DEFAULT_ALPHA, seed=20)
    payload = _upload_payload(adapter, "peft-client", peft_style=True)
    resp = client.post("/uploads", json=payload)
    assert resp.status_code == 201, resp.text
    assert resp.json()["tensors_received"] == 2 * len(TARGET_MODULES) * 2


def test_aggregate_naive_branch():
    """Review coverage gap: the naive aggregation branch in /aggregate was
    never exercised by any test (only the svd path was)."""
    _reset_state()
    client = TestClient(app)
    adapters = [random_adapter(8, 8, rank=4, num_layers=2, seed=s) for s in (30, 31)]
    for i, adapter in enumerate(adapters):
        rng = np.random.default_rng(i)
        for layer in adapter.layer_indices:
            for name in adapter.target_modules:
                adapter.modules[layer][name]["lora_B"] = rng.normal(
                    size=adapter.modules[layer][name]["lora_B"].shape
                ).astype(np.float32)
        payload = _upload_payload(adapter, f"naive-client-{i}", num_examples=10)
        resp = client.post("/uploads", json=payload)
        assert resp.status_code == 201, resp.text

    agg_resp = client.post("/aggregate", json={"aggregation": "naive"})
    assert agg_resp.status_code == 200, agg_resp.text
    body = agg_resp.json()
    assert body["aggregation"] == "naive"
    assert body["num_clients"] == 2


def test_aggregate_unknown_method_returns_422():
    """Review coverage gap: the unknown-aggregation-method else branch in
    /aggregate was never exercised by any test."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=40)
    resp = client.post("/uploads", json=_upload_payload(adapter, "c0", num_examples=10))
    assert resp.status_code == 201, resp.text

    agg_resp = client.post("/aggregate", json={"aggregation": "bogus-method"})
    assert agg_resp.status_code == 422
    assert "bogus-method" in agg_resp.json()["detail"]


def test_upload_rejects_adapter_format_error_returns_422():
    """Review coverage gap: AdapterFormatError -> 422 in /uploads was never
    exercised. Distinct from test_upload_rejects_wrong_tensor_count, which
    only triggers AdapterUpload's own pydantic-level tensor-COUNT check —
    this payload has the structurally-correct tensor count (so it passes
    pydantic validation) but an unrecognized module name, so it is
    LoRAAdapter.from_state_dict() itself that rejects it."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=41)
    payload = _upload_payload(adapter, "bad-key-client", num_examples=10)
    # Rename one tensor to reference a module that isn't in target_modules —
    # same tensor count, still unique names, so the pydantic structural
    # check passes and the request reaches from_state_dict().
    payload["tensors"][0]["name"] = payload["tensors"][0]["name"].replace(
        "q_proj", "bogus_proj"
    )
    resp = client.post("/uploads", json=payload)
    assert resp.status_code == 422, resp.text
    assert "unrecognized" in resp.json()["detail"] or "bogus_proj" in resp.json()["detail"]


def test_manifest_before_any_aggregate_returns_404():
    """Review coverage gap: /aggregate/manifest's 404-before-any-run path
    was never exercised (only /adapters/cluster/active's 404 was)."""
    _reset_state()
    client = TestClient(app)
    resp = client.get("/aggregate/manifest")
    assert resp.status_code == 404


def test_aggregate_manifest_omits_reconstruction_error_by_default():
    """Fix verification: the expensive SVD-reconstruction-error diagnostic
    is opt-in (AggregateRequest.include_manifest) and is NOT computed unless
    explicitly requested."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=42)
    resp = client.post("/uploads", json=_upload_payload(adapter, "c0", num_examples=10))
    assert resp.status_code == 201, resp.text

    agg_resp = client.post("/aggregate", json={"aggregation": "svd"})
    assert agg_resp.status_code == 200, agg_resp.text

    manifest = client.get("/aggregate/manifest").json()
    assert "svd_reconstruction_error" not in manifest
    assert manifest["num_clients"] == 1


def test_upload_rejects_mismatched_round_id_with_409():
    """Fix verification: a client uploading for a stale/wrong round is
    rejected (409) instead of being silently folded into whichever round is
    currently buffering."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=43)
    payload = _upload_payload(adapter, "stale-client", round_id=7, num_examples=10)
    resp = client.post("/uploads", json=payload)
    assert resp.status_code == 409, resp.text
    assert "round_id" in resp.json()["detail"]
    # and it must NOT have been buffered
    health = client.get("/healthz").json()
    assert health["pending_uploads"] == 0


def test_upload_rejects_garbled_dtype_with_422_not_500():
    """Fix verification: TensorPayload._check_size used to raise a bare
    TypeError on a garbled dtype string, which pydantic v2 does not convert
    into a ValidationError, crashing POST /uploads with an unhandled 500.
    It must now come back as a clean 422."""
    _reset_state()
    client = TestClient(app)
    adapter = random_adapter(8, 8, rank=4, num_layers=1, seed=44)
    payload = _upload_payload(adapter, "garbled-dtype-client", num_examples=10)
    payload["tensors"][0]["dtype"] = "bogus"
    resp = client.post("/uploads", json=payload)
    assert resp.status_code == 422, resp.text
    assert resp.status_code != 500
