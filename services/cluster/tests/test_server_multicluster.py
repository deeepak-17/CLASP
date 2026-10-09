"""Week 8 (HTTP): per-cluster endpoints, membership isolation, straggler policy."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster import server
from cluster.adapter_format import LoRAAdapter
from cluster.aggregation import aggregate_svd
from cluster.schemas.messages import TensorPayload
from tests.conftest import trained_adapter
from tests.http_helpers import create_cluster, reset_server_state, restore_server_state

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _body(client_id, adapter, round_id=0, n=10):
    return {
        "client_id": client_id,
        "round_id": round_id,
        "rank": adapter.rank,
        "target_modules": list(adapter.target_modules),
        "alpha": adapter.alpha,
        "num_layers": adapter.num_layers,
        "num_examples": n,
        "tensors": [
            TensorPayload.from_numpy(k, v).model_dump() for k, v in adapter.to_state_dict().items()
        ],
    }


def _decode(js):
    return LoRAAdapter.from_state_dict(
        {t["name"]: TensorPayload(**t).to_numpy() for t in js["tensors"]},
        rank=js["rank"], alpha=js["alpha"], target_modules=tuple(js["target_modules"]),
        num_layers=js["num_layers"],
    )


def test_clusters_aggregate_independently_and_keep_separate_round_counters():
    a = {c: trained_adapter(i) for i, c in enumerate(("a1", "a2"))}
    b = {c: trained_adapter(10 + i) for i, c in enumerate(("b1", "b2"))}
    create_cluster(client, "team-a", "team-b")
    for cid, ad in a.items():
        assert client.post("/clusters/team-a/uploads", json=_body(cid, ad)).status_code == 201
    for cid, ad in b.items():
        assert client.post("/clusters/team-b/uploads", json=_body(cid, ad)).status_code == 201
    ra = client.post("/clusters/team-a/aggregate")
    assert ra.status_code == 200 and ra.json()["cluster_id"] == "team-a"
    # team-b is untouched by team-a's aggregation
    listing = client.get("/clusters").json()["clusters"]
    assert listing["team-a"]["round_id"] == 1 and listing["team-a"]["pending_uploads"] == 0
    assert listing["team-b"]["round_id"] == 0 and listing["team-b"]["pending_uploads"] == 2
    rb = client.post("/clusters/team-b/aggregate")
    got_a, got_b = _decode(ra.json()), _decode(rb.json())
    ref_a = aggregate_svd(iter(a.values()), [10.0, 10.0], rank=16)
    ref_b = aggregate_svd(iter(b.values()), [10.0, 10.0], rank=16)
    np.testing.assert_allclose(got_a.delta_w("q_proj"), ref_a.delta_w("q_proj"), atol=1e-5)
    np.testing.assert_allclose(got_b.delta_w("q_proj"), ref_b.delta_w("q_proj"), atol=1e-5)
    assert not np.allclose(got_a.delta_w("q_proj"), got_b.delta_w("q_proj"), atol=1e-4)
    act = client.get("/clusters/team-a/adapters/active").json()
    assert act["cluster_id"] == "team-a" and client.get("/clusters/team-a/aggregate/manifest").json()["cluster_id"] == "team-a"


def test_member_of_another_cluster_cannot_upload_403_and_nothing_is_buffered():
    client.put("/clusters/team-a/members", json={"client_ids": ["alice"]})
    client.put("/clusters/team-b/members", json={"client_ids": ["bob"]})
    r = client.post("/clusters/team-b/uploads", json=_body("alice", trained_adapter(1)))
    assert r.status_code == 403 and "team-a" in r.json()["detail"]
    assert client.get("/clusters").json()["clusters"]["team-b"]["pending_uploads"] == 0
    # alice may still upload to her own cluster; the legacy endpoint is also guarded
    assert client.post("/clusters/team-a/uploads", json=_body("alice", trained_adapter(1))).status_code == 201
    assert client.post("/uploads", json=_body("alice", trained_adapter(1))).status_code == 403


def test_moving_a_client_discards_its_buffered_upload_in_the_old_cluster():
    client.put("/clusters/team-a/members", json={"client_ids": ["alice", "carol"]})
    for cid in ("alice", "carol"):
        client.post("/clusters/team-a/uploads", json=_body(cid, trained_adapter(len(cid))))
    r = client.put("/clusters/team-b/members", json={"client_ids": ["alice"]})
    assert r.json()["members"] == ["alice"]
    out = client.post("/clusters/team-a/aggregate").json()
    manifest = client.get("/clusters/team-a/aggregate/manifest").json()
    assert manifest["source_clients"] == ["carol"] and out["num_clients"] == 1


def test_cluster_listing_names_who_uploaded_and_who_was_aggregated():
    client.put("/clusters/team-a/members", json={"client_ids": ["alice", "bob", "carol"]})
    entry = client.get("/clusters").json()["clusters"]["team-a"]
    assert entry["uploaded_clients"] == [] and entry["aggregated_clients"] == []
    assert entry["aggregated_round_id"] is None
    for cid in ("bob", "alice"):
        client.post("/clusters/team-a/uploads", json=_body(cid, trained_adapter(len(cid))))
    entry = client.get("/clusters").json()["clusters"]["team-a"]
    assert entry["uploaded_clients"] == ["alice", "bob"] and entry["pending_uploads"] == 2
    assert entry["members"] == ["alice", "bob", "carol"]
    client.post("/clusters/team-a/aggregate")
    entry = client.get("/clusters").json()["clusters"]["team-a"]
    assert entry["uploaded_clients"] == []
    assert entry["aggregated_clients"] == ["alice", "bob"] and entry["aggregated_round_id"] == 0


def test_unknown_cluster_and_bad_ids():
    assert client.get("/clusters/nope/adapters/active").status_code == 404
    assert client.post("/clusters/nope/aggregate").status_code == 404
    assert client.put("/clusters/bad%20id/members", json={"client_ids": []}).status_code == 422


def test_legacy_endpoints_still_target_cluster_default():
    client.post("/uploads", json=_body("x", trained_adapter(1)))
    out = client.post("/aggregate").json()
    assert out["cluster_id"] == "cluster-default"
    assert "cluster-default" in client.get("/healthz").json()["clusters"]


def test_http_straggler_timeout_skips_late_upload_and_reports_it():
    ticks = iter([0.0, 1.0, 100.0])
    state = server._cluster("team-s", create=True)
    state.clock = lambda: next(ticks)
    ads = [trained_adapter(s) for s in (1, 2, 3)]
    for cid, ad, n in zip(("fast1", "fast2", "late"), ads, (10, 30, 500)):
        assert client.post("/clusters/team-s/uploads", json=_body(cid, ad, n=n)).status_code == 201
    r = client.post("/clusters/team-s/aggregate", json={"straggler_timeout_s": 10.0})
    assert r.status_code == 200 and r.json()["num_clients"] == 2
    manifest = client.get("/clusters/team-s/aggregate/manifest").json()
    assert manifest["skipped_clients"] == {"late": "timeout"}
    ref = aggregate_svd(iter(ads[:2]), [10.0, 30.0], rank=16)  # sample-weighted survivors
    np.testing.assert_allclose(_decode(r.json()).delta_w("k_proj"), ref.delta_w("k_proj"), atol=1e-5)


def test_http_quorum_failure_409_keeps_buffer_for_retry():
    ticks = iter([0.0, 100.0])
    state = server._cluster("team-q", create=True)
    state.clock = lambda: next(ticks)
    for cid, s in (("on-time", 1), ("late", 2)):
        client.post("/clusters/team-q/uploads", json=_body(cid, trained_adapter(s)))
    r = client.post(
        "/clusters/team-q/aggregate", json={"straggler_timeout_s": 10.0, "min_clients": 2}
    )
    assert r.status_code == 409 and "quorum not met" in r.json()["detail"]
    assert client.get("/clusters").json()["clusters"]["team-q"]["pending_uploads"] == 2
    assert state.round_id == 0 and state.active is None


def test_http_invalid_policy_422():
    client.post("/uploads", json=_body("x", trained_adapter(1)))
    assert client.post("/aggregate", json={"weighting": "bogus"}).status_code == 422
    assert client.post("/aggregate", json={"aggregation": "bogus"}).status_code == 422


def test_http_upload_with_nan_tensor_is_rejected_422():
    body = _body("x", trained_adapter(1))
    bad_name = "layers.0.q_proj.lora_B.weight"
    sd = trained_adapter(1).to_state_dict()
    sd[bad_name] = sd[bad_name].copy()
    sd[bad_name][0, 0] = np.nan  # encode the NaN straight into the wire payload
    body["tensors"] = [TensorPayload.from_numpy(k, v).model_dump() for k, v in sd.items()]
    r = client.post("/uploads", json=body)
    assert r.status_code == 422 and "non-finite" in r.json()["detail"]
