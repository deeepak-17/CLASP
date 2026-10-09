"""Both HTTP route sets over one state, and D7 (epsilon) end to end.

The integration surface (``POST /uploads``, ``POST /aggregate`` with the cluster in
the body, ``/adapters/{id}/active|download|manifest|publish``) and the
cluster-addressed surface (``/clusters/{id}/...``) must be two views of the same
per-cluster state, with the same locking and isolation rules.
"""

from __future__ import annotations

import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from cluster import server
from cluster.aggregation import aggregate_svd_lowrank
from cluster.schemas.messages import AdapterUpload
from cluster.server import cluster_epsilon
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    reset_server_state,
    restore_server_state,
    upload_body,
)

client = TestClient(server.app)
JOIN_S = 60


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _up(cluster, who, seed, *, route="/uploads", **extra):
    body = upload_body(who, trained_adapter(seed), cluster_id=cluster, **extra)
    return client.post(route, json=body)


class _Registry:
    """Stands in for the registry process: records what ``/publish`` sends."""

    def __init__(self, monkeypatch, status=201):
        self.posts: list[dict] = []
        outer = self

        class _Resp:
            status_code = status
            text = "stub"

            def json(self):
                return {"version": len(outer.posts)}

        class _Shim:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, files=None, data=None, **kw):
                outer.posts.append({"url": url, "meta": json.loads(data["meta"]), "files": files})
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Shim)


# ---- the two route sets are one state ---------------------------------------


def test_upload_by_body_aggregate_by_path_read_by_adapters_route():
    for who, seed in (("a1", 1), ("a2", 2)):
        assert _up("cluster-web", who, seed).status_code == 201
    health = client.get("/healthz").json()["clusters"]["cluster-web"]
    assert health["pending_uploads"] == 2 and health["pending_clients"] == ["a1", "a2"]

    agg = client.post("/clusters/cluster-web/aggregate")
    assert agg.status_code == 200 and agg.json()["source_clients"] == ["a1", "a2"]
    for route in ("/adapters/cluster-web/active", "/clusters/cluster-web/adapters/active"):
        got = client.get(route)
        assert got.status_code == 200 and got.json()["cluster_id"] == "cluster-web"
    assert client.get("/adapters/cluster-web/download").status_code == 200
    for route in ("/adapters/cluster-web/manifest", "/clusters/cluster-web/aggregate/manifest"):
        assert client.get(route).json()["source_clients"] == ["a1", "a2"]
    assert client.get("/clusters").json()["clusters"]["cluster-web"]["round_id"] == 1


def test_upload_by_path_aggregate_by_body_cluster_id():
    create_cluster(client, "team-a")
    assert _up("team-a", "c1", 1, route="/clusters/team-a/uploads").status_code == 201
    agg = client.post("/aggregate", json={"cluster_id": "team-a", "include_manifest": True})
    assert agg.status_code == 200 and agg.json()["cluster_id"] == "team-a"
    assert client.get("/adapters/team-a/active").status_code == 200


def test_the_integration_surface_still_creates_a_cluster_on_its_first_upload():
    # demo_ui and the four-seam test rely on this; /clusters/{id}/uploads does not
    assert "cluster-web" not in server._clusters
    assert _up("cluster-web", "c1", 1).status_code == 201
    assert "cluster-web" in server._clusters
    assert _up("typo-clsuter", "c1", 1, route="/clusters/typo-clsuter/uploads").status_code == 404


def test_aggregate_on_a_cluster_that_does_not_exist_is_a_404_and_creates_nothing():
    assert client.post("/aggregate", json={"cluster_id": "nope"}).status_code == 404
    assert "nope" not in server._clusters
    assert client.get("/adapters/nope/active").status_code == 404
    assert client.get("/adapters/nope/manifest").status_code == 404
    assert "nope" not in server._clusters


def test_default_cluster_aliases_still_work():
    assert client.post("/uploads", json=upload_body("c1", trained_adapter(1))).status_code == 201
    assert client.post("/aggregate").status_code == 200
    assert client.get("/adapters/cluster/active").json()["cluster_id"] == server.DEFAULT_CLUSTER_ID
    assert client.get("/aggregate/manifest").status_code == 200


def test_body_cluster_id_must_agree_with_the_path():
    create_cluster(client, "team-a", "team-b")
    r = _up("team-b", "c1", 1, route="/clusters/team-a/uploads")
    assert r.status_code == 422 and "does not match the path" in r.json()["detail"]
    assert _up("team-a", "c1", 1, route="/clusters/team-a/uploads").status_code == 201
    r = client.post("/clusters/team-a/aggregate", json={"cluster_id": "team-b"})
    assert r.status_code == 422
    # leaving cluster_id out of the body is fine: the path names the cluster
    assert client.post("/clusters/team-a/aggregate", json={}).status_code == 200


def test_clusters_created_by_uploads_are_capped(monkeypatch):
    monkeypatch.setattr(server, "MAX_CLUSTERS", 3)  # default cluster + 2 more
    assert _up("c-one", "x", 1).status_code == 201
    assert _up("c-two", "x", 1).status_code == 201
    r = _up("c-three", "x", 1)
    assert r.status_code == 429 and "c-three" not in server._clusters
    assert client.put("/clusters/c-four/members", json={"client_ids": []}).status_code == 429
    assert _up("c-one", "y", 2).status_code == 201  # existing clusters are unaffected


def test_membership_isolation_applies_to_the_body_routed_upload_too():
    create_cluster(client, "team-a", "team-b", members={"team-a": ["alice"]})
    r = _up("team-b", "alice", 1)
    assert r.status_code == 403 and "assigned to cluster" in r.json()["detail"]


def test_a_retained_aggregate_is_an_ablation_not_a_completed_round():
    class Sink:
        def __init__(self):
            self.n = 0

        def publish(self, ref, broadcast):
            self.n += 1

    sink = Sink()
    server.configure_snapshot_sink(sink)
    for who, seed in (("a1", 1), ("a2", 2)):
        assert _up("cluster-web", who, seed).status_code == 201
    r = client.post("/aggregate", json={"cluster_id": "cluster-web", "retain_uploads": True})
    assert r.status_code == 200
    st = server._clusters["cluster-web"]
    assert st.round_id == 0 and st.uploads.keys() == {"a1", "a2"}  # buffer + round kept
    assert st.last_round == {} and sink.n == 0  # not evidence, not published
    assert client.get("/adapters/cluster-web/manifest").json()["retained_uploads"] is True
    assert client.post("/aggregate", json={"cluster_id": "cluster-web"}).status_code == 200
    assert st.round_id == 1 and sink.n == 1 and set(st.last_round) == {"a1", "a2"}


def test_a_rank_the_matrices_cannot_have_is_a_422_on_both_paths():
    assert _up("cluster-web", "c1", 1).status_code == 201
    for lowrank in (True, False):
        r = client.post(
            "/aggregate", json={"cluster_id": "cluster-web", "rank": 999, "exact_lowrank": lowrank}
        )
        assert r.status_code == 422 and "cannot aggregate" in r.json()["detail"]
    assert server._clusters["cluster-web"].uploads.keys() == {"c1"}  # buffer kept
    assert client.post("/aggregate", json={"cluster_id": "cluster-web"}).status_code == 200


def test_lowrank_refuses_adapters_with_different_scaling():
    a, b = trained_adapter(1), trained_adapter(2)
    b.alpha = a.alpha * 2
    with pytest.raises(ValueError, match="share alpha"):
        aggregate_svd_lowrank([a, b], [1.0, 1.0])


# ---- D7: epsilon ------------------------------------------------------------


def test_cluster_epsilon_is_the_max_and_unknown_if_any_client_did_not_report():
    assert cluster_epsilon([2.0, 5.0, 3.5]) == 5.0
    assert cluster_epsilon([4.0]) == 4.0
    assert cluster_epsilon([2.0, None]) is None  # one non-DP client: no finite epsilon
    assert cluster_epsilon([None, None]) is None
    assert cluster_epsilon([]) is None


def test_upload_epsilon_is_optional_and_validated():
    ok = upload_body("c1", trained_adapter(1))
    assert AdapterUpload(**ok).epsilon is None
    assert AdapterUpload(**{**ok, "epsilon": 0.0}).epsilon == 0.0
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            AdapterUpload(**{**ok, "epsilon": bad})
    assert _up("cluster-web", "c1", 1, epsilon=-0.5).status_code == 422


def _edge_privacy(epsilon, **over):
    """The ``privacy`` block exactly as ``edge.wire.privacy_block`` builds it
    (``asdict(contracts.PrivacySpec(...))``): all four keys, always present."""
    return {
        "epsilon": epsilon,
        "delta": 1e-05,
        "noise_multiplier": None,
        "max_grad_norm": None,
        **over,
    }


def test_the_privacy_block_the_edge_sends_sets_the_upload_epsilon():
    ok = upload_body("c1", trained_adapter(1))
    up = AdapterUpload(**{**ok, "privacy": _edge_privacy(7.99)})
    assert up.epsilon == 7.99 and up.privacy is not None and up.privacy.delta == 1e-05
    # DP off (epsilon None in the block) stays "not reported", it is not 0
    assert AdapterUpload(**{**ok, "privacy": _edge_privacy(None)}).epsilon is None
    assert AdapterUpload(**ok).privacy is None


def test_epsilon_and_privacy_epsilon_must_agree_when_both_are_given():
    ok = upload_body("c1", trained_adapter(1))
    both = {**ok, "epsilon": 5.0, "privacy": _edge_privacy(5.0)}
    assert AdapterUpload(**both).epsilon == 5.0
    with pytest.raises(ValueError, match="disagree"):
        AdapterUpload(**{**ok, "epsilon": 5.0, "privacy": _edge_privacy(6.0)})
    # DP off in the block does not contradict a top-level epsilon: nothing to compare
    assert AdapterUpload(**{**ok, "epsilon": 5.0, "privacy": _edge_privacy(None)}).epsilon == 5.0
    assert _up("cluster-web", "c1", 1, epsilon=5.0, privacy=_edge_privacy(6.0)).status_code == 422


def test_the_privacy_block_is_validated_and_tolerates_unknown_keys():
    ok = upload_body("c1", trained_adapter(1))
    for bad in (
        _edge_privacy(-1.0),
        _edge_privacy(float("inf")),
        _edge_privacy(1.0, delta=1.5),
        _edge_privacy(1.0, noise_multiplier=-0.1),
        _edge_privacy("lots"),
    ):
        with pytest.raises(ValueError):
            AdapterUpload(**{**ok, "privacy": bad})
    # a later contracts version may add a field: ignored, not a reason to refuse the upload
    future = {**_edge_privacy(2.0), "accountant": "rdp"}
    assert AdapterUpload(**{**ok, "privacy": future}).epsilon == 2.0


def test_epsilon_sent_in_the_edge_privacy_shape_reaches_the_registry(monkeypatch):
    """The seam that was broken: the Edge sends ``privacy.epsilon``, not a
    top-level ``epsilon``. Cluster epsilon = max over the clients, published as
    ``privacy.epsilon``; one client on the old top-level field mixes in fine."""
    registry = _Registry(monkeypatch)
    for who, seed, kw in (
        ("a1", 1, {"privacy": _edge_privacy(2.0)}),
        ("a2", 2, {"privacy": _edge_privacy(5.0)}),
        ("a3", 3, {"epsilon": 3.5}),
    ):
        assert _up("cluster-web", who, seed, **kw).status_code == 201
    agg = client.post("/aggregate", json={"cluster_id": "cluster-web"})
    assert agg.status_code == 200 and agg.json()["epsilon"] == 5.0
    manifest = client.get("/adapters/cluster-web/manifest").json()
    assert manifest["epsilon"] == 5.0 and "epsilon_not_reported_by" not in manifest
    assert client.post("/adapters/cluster-web/publish").status_code == 201
    assert registry.posts[0]["meta"]["privacy"] == {"epsilon": 5.0}


def test_a_dp_off_edge_client_still_makes_the_cluster_epsilon_unknown(monkeypatch):
    registry = _Registry(monkeypatch)
    assert _up("cluster-web", "dp", 1, privacy=_edge_privacy(2.0)).status_code == 201
    assert _up("cluster-web", "plain", 2, privacy=_edge_privacy(None)).status_code == 201
    assert client.post("/aggregate", json={"cluster_id": "cluster-web"}).json()["epsilon"] is None
    manifest = client.get("/adapters/cluster-web/manifest").json()
    assert manifest["epsilon_not_reported_by"] == ["plain"]
    assert client.post("/adapters/cluster-web/publish").status_code == 201
    assert "privacy" not in registry.posts[0]["meta"]


def test_aggregate_records_the_max_epsilon_and_publishes_it(monkeypatch):
    registry = _Registry(monkeypatch)
    for who, seed, eps in (("a1", 1, 2.0), ("a2", 2, 5.0), ("a3", 3, 3.5)):
        assert _up("cluster-web", who, seed, epsilon=eps).status_code == 201
    agg = client.post("/aggregate", json={"cluster_id": "cluster-web"})
    assert agg.status_code == 200 and agg.json()["epsilon"] == 5.0
    manifest = client.get("/adapters/cluster-web/manifest").json()
    assert manifest["epsilon"] == 5.0 and "epsilon_not_reported_by" not in manifest

    pub = client.post("/adapters/cluster-web/publish")
    assert pub.status_code == 201
    (post,) = registry.posts
    assert post["meta"]["privacy"] == {"epsilon": 5.0}
    assert post["meta"]["source_clients"] == ["a1", "a2", "a3"]
    assert post["url"].endswith("/adapters/cluster-web/versions")


def test_epsilon_is_the_max_over_the_clients_actually_aggregated(monkeypatch):
    # a straggler's epsilon must not count: it is not in the aggregate
    ticks = iter([0.0, 1.0, 50.0])
    monkeypatch.setattr(server._clusters[server.DEFAULT_CLUSTER_ID], "clock", lambda: next(ticks))
    for who, seed, eps in (("a1", 1, 2.0), ("a2", 2, 3.0), ("slow", 3, 9.0)):
        assert client.post(
            "/uploads", json=upload_body(who, trained_adapter(seed), epsilon=eps)
        ).status_code == 201
    agg = client.post("/aggregate", json={"straggler_timeout_s": 10})
    assert agg.status_code == 200
    assert agg.json()["source_clients"] == ["a1", "a2"] and agg.json()["epsilon"] == 3.0
    assert client.get("/aggregate/manifest").json()["skipped_clients"].keys() == {"slow"}


def test_one_client_without_epsilon_makes_the_cluster_epsilon_unknown(monkeypatch):
    registry = _Registry(monkeypatch)
    assert _up("cluster-web", "dp", 1, epsilon=2.0).status_code == 201
    assert _up("cluster-web", "plain", 2).status_code == 201
    agg = client.post("/aggregate", json={"cluster_id": "cluster-web"})
    assert agg.json()["epsilon"] is None
    manifest = client.get("/adapters/cluster-web/manifest").json()
    assert manifest["epsilon"] is None and manifest["epsilon_not_reported_by"] == ["plain"]
    assert client.post("/adapters/cluster-web/publish").status_code == 201
    assert "privacy" not in registry.posts[0]["meta"]  # registry default = epsilon None


# ---- publish_to_registry --------------------------------------------------


def test_publish_errors_are_reported_not_swallowed(monkeypatch):
    assert client.post("/adapters/nope/publish").status_code == 404
    assert _up("cluster-web", "c1", 1).status_code == 201
    assert client.post("/adapters/cluster-web/publish").status_code == 404  # nothing aggregated
    client.post("/aggregate", json={"cluster_id": "cluster-web"})
    _Registry(monkeypatch, status=500)
    r = client.post("/adapters/cluster-web/publish")
    assert r.status_code == 502 and "registry rejected" in r.json()["detail"]


def test_publish_during_an_aggregate_waits_and_sends_the_new_round(monkeypatch):
    registry = _Registry(monkeypatch)
    real = server.aggregate_svd_lowrank
    started, release = threading.Event(), threading.Event()

    def slow(adapters, weights, **kw):
        started.set()
        assert release.wait(JOIN_S)
        return real(adapters, weights, **kw)

    monkeypatch.setattr(server, "aggregate_svd_lowrank", slow)
    assert _up("cluster-web", "c1", 1).status_code == 201

    out: dict[str, object] = {}
    t_agg = threading.Thread(
        target=lambda: TestClient(server.app).post("/aggregate", json={"cluster_id": "cluster-web"}),
        daemon=True,
    )
    t_pub = threading.Thread(
        target=lambda: out.update(r=TestClient(server.app).post("/adapters/cluster-web/publish")),
        daemon=True,
    )
    t_agg.start()
    assert started.wait(JOIN_S)
    t_pub.start()
    t_pub.join(0.5)
    # without the lock this would already have answered 404 (no active adapter yet)
    assert t_pub.is_alive() and "r" not in out

    release.set()
    t_agg.join(JOIN_S)
    t_pub.join(JOIN_S)
    assert not t_agg.is_alive() and not t_pub.is_alive()
    assert out["r"].status_code == 201
    assert registry.posts[0]["meta"]["source_clients"] == ["c1"]
    assert registry.posts[0]["meta"]["round"] == 0
