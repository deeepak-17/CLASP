"""Week 10 (HTTP): ``POST /recluster`` — dry run, apply, fallback, isolation."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster import server
from cluster.redistribution import adapter_from_broadcast
from tests.conftest import planted_directions, trained_adapter
from tests.http_helpers import reset_server_state, restore_server_state, upload_body

client = TestClient(server.app)
WEB, SCI = "web", "sci"


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _setup_two_clusters(wrong_label=True, same_direction=False, noise=0.01):
    """web: w0 w1 w2 (+ s2 mislabelled), sci: s0 s1; one aggregated round each."""
    dirs = planted_directions(2)
    group = {"w0": 0, "w1": 0, "w2": 0, "s0": 1, "s1": 1, "s2": 1}
    if same_direction:
        group = dict.fromkeys(group, 0)
    members = {WEB: ["w0", "w1", "w2"], SCI: ["s0", "s1"]}
    members[WEB if wrong_label else SCI].append("s2")
    for cid, ids in members.items():
        assert client.put(f"/clusters/{cid}/members", json={"client_ids": ids}).status_code == 200
    for cid, ids in members.items():
        for c in ids:
            ad = trained_adapter(100 + sorted(group).index(c), direction=dirs[group[c]], noise=noise)
            assert client.post(f"/clusters/{cid}/uploads", json=upload_body(c, ad)).status_code == 201
        assert client.post(f"/clusters/{cid}/aggregate").status_code == 200
    return members


def _membership():
    listing = client.get("/clusters").json()["clusters"]
    return {cid: v["members"] for cid, v in listing.items() if v["members"]}


def test_dry_run_proposes_the_fix_and_changes_nothing():
    _setup_two_clusters()
    before = _membership()
    r = client.post("/recluster")
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["applied"] is False and js["method"] == "dynamic"
    assert js["moved"] == {"s2": [WEB, SCI]}
    assert js["assignment"]["s2"] == SCI
    assert js["clients_clustered"] == ["s0", "s1", "s2", "w0", "w1", "w2"]
    assert js["comparison"]["agreement"] < 1.0  # differs from the (wrong) warm start
    assert _membership() == before
    # a dry run does not consume the evidence: asking again gives the same answer
    assert client.post("/recluster").json() == js


def test_apply_moves_the_client_and_consumes_the_evidence():
    _setup_two_clusters()
    # s2 has a buffered round-1 upload in its old cluster; moving it must drop it
    ad = trained_adapter(7)
    assert client.post(f"/clusters/{WEB}/uploads", json=upload_body("s2", ad, round_id=1)).status_code == 201
    assert client.get("/clusters").json()["clusters"][WEB]["pending_uploads"] == 1

    js = client.post("/recluster", json={"apply": True}).json()
    assert js["applied"] is True and js["moved"] == {"s2": [WEB, SCI]}
    members = _membership()
    assert members[SCI] == ["s0", "s1", "s2"] and members[WEB] == ["w0", "w1", "w2"]
    assert client.get("/clusters").json()["clusters"][WEB]["pending_uploads"] == 0

    again = client.post("/recluster", json={"apply": True})
    assert again.status_code == 409 and "no completed aggregation round" in again.json()["detail"]


def test_isolation_survives_the_move():
    _setup_two_clusters()
    client.post("/recluster", json={"apply": True})
    ad = trained_adapter(8)
    # the old cluster now refuses s2 ...
    r = client.post(f"/clusters/{WEB}/uploads", json=upload_body("s2", ad, round_id=1))
    assert r.status_code == 403
    # ... a stale adapter (still trained from web's) is refused by the new one ...
    r = client.post(
        f"/clusters/{SCI}/uploads",
        json=upload_body("s2", ad, round_id=1, base_cluster_id=WEB),
    )
    assert r.status_code == 409 and "stale_adapter" in r.json()["detail"]
    # ... and one trained from the new cluster's adapter is accepted
    r = client.post(
        f"/clusters/{SCI}/uploads",
        json=upload_body("s2", ad, round_id=1, base_cluster_id=SCI),
    )
    assert r.status_code == 201


def test_response_never_carries_adapter_tensors_or_the_matrix_by_default():
    _setup_two_clusters()
    js = client.post("/recluster").json()
    assert "tensors" not in js and "similarity" not in js
    assert "tensors" not in str(js)


def test_include_similarity_returns_a_symmetric_unit_diagonal_matrix():
    _setup_two_clusters()
    js = client.post("/recluster", json={"include_similarity": True}).json()
    sim = np.array(js["similarity"])
    assert sim.shape == (6, 6)
    np.testing.assert_allclose(sim, sim.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)


def test_weak_separation_falls_back_to_static_and_never_applies():
    _setup_two_clusters(wrong_label=False, same_direction=True, noise=0.001)
    before = _membership()
    js = client.post("/recluster", json={"apply": True}).json()
    assert js["method"] == "static_fallback" and js["applied"] is False
    assert "weak separation" in js["reason"] and js["moved"] == {}
    assert _membership() == before
    # evidence is kept after a fallback so a re-try with other settings works
    assert client.post("/recluster", json={"min_separation": 0.0}).status_code == 200


def test_needs_two_clusters_and_an_aggregated_round():
    r = client.post("/recluster")
    assert r.status_code == 409 and ">= 2 clusters" in r.json()["detail"]
    client.put(f"/clusters/{WEB}/members", json={"client_ids": ["w0"]})
    client.put(f"/clusters/{SCI}/members", json={"client_ids": ["s0"]})
    r = client.post("/recluster")
    assert r.status_code == 409 and "no completed aggregation round" in r.json()["detail"]


def test_clients_without_membership_are_reported_and_ignored():
    _setup_two_clusters(wrong_label=False)
    # legacy single-cluster flow: an unassigned client uploads to cluster-default
    assert client.post("/uploads", json=upload_body("loner", trained_adapter(3))).status_code == 201
    assert client.post("/aggregate").status_code == 200
    js = client.post("/recluster").json()
    assert js["ignored"] == {"loner": "not assigned to any cluster"}
    assert "loner" not in js["assignment"]


def test_invalid_parameters_are_rejected():
    assert client.post("/recluster", json={"min_separation": -1}).status_code == 422
    assert client.post("/recluster", json={"min_cluster_size": 0}).status_code == 422


def test_second_round_evidence_uses_the_previous_cluster_adapter_as_start():
    _setup_two_clusters(wrong_label=False)
    st = server._clusters[WEB]
    first_active = adapter_from_broadcast(st.active)
    assert all(start is None for _, start in st.last_round.values())  # round 0: no start yet
    for c in ("w0", "w1", "w2"):
        ad = trained_adapter(500 + ord(c[1]))
        client.post(f"/clusters/{WEB}/uploads", json=upload_body(c, ad, round_id=1))
    assert client.post(f"/clusters/{WEB}/aggregate").status_code == 200
    for _, start in st.last_round.values():
        assert start is not None
        np.testing.assert_allclose(
            start.delta_w("q_proj"), first_active.delta_w("q_proj"), atol=1e-6
        )
