"""D7 at the cluster (U3): epsilon at most 8 over a client's whole run of 2 rounds.

The edge keeps the ledger and sends the run's cumulative epsilon with every
differentially private upload. The cluster checks it again: an upload that does
not report it, or reports more than the budget or more rounds than a run has,
is refused and never buffered. The cluster's own epsilon for a round is the
largest cumulative epsilon among the clients it aggregated: each client's
accountant already composed its rounds, so nothing is added up here.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cluster import server
from cluster.schemas.messages import AdapterUpload
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    dp,
    promote,
    reset_server_state,
    restore_server_state,
    upload_body,
)

client = TestClient(server.app)
ALPHA = 0.5


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    create_cluster(client, "web", members={"web": ["a", "b"]})
    yield
    restore_server_state(saved)


def _post(who, seed, **extra):
    return client.post("/clusters/web/uploads", json=upload_body(who, trained_adapter(seed), **extra))


def _buffered():
    return sorted(server._clusters["web"].uploads)


def _residual(who, seed, active, **extra):
    return client.post("/clusters/web/uploads", json=upload_body(
        who, trained_adapter(seed), round_id=active["round_id"] + 1,
        base_sha256=active["sha256"], composition_alpha=ALPHA, **extra))


# ---- what an upload must report ------------------------------------------------
def test_the_schema_reads_the_run_accounting_the_edge_sends():
    body = upload_body("a", trained_adapter(1), privacy={
        "epsilon": 3.9, "delta": 1e-5, "noise_multiplier": 0.47, "max_grad_norm": 1.0,
        "epsilon_cumulative": 7.8, "rounds_accounted": 2, "sample_rate": 0.00625, "steps": 160})
    privacy = AdapterUpload(**body).privacy
    assert privacy.epsilon_cumulative == 7.8 and privacy.rounds_accounted == 2
    assert privacy.sample_rate == 0.00625 and privacy.steps == 160


def test_dp_upload_without_cumulative_is_422_and_not_buffered():
    """Review Focus 2: an edge that has not pulled this change."""
    for old_shape in ({"privacy": {"epsilon": 7.99, "delta": 1e-5, "noise_multiplier": None,
                                   "max_grad_norm": None}},
                      {"epsilon": 7.99}):
        r = _post("a", 1, **old_shape)
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert detail["error"] == "privacy_budget_unreported"
        assert "update the edge" in detail["hint"] and "cumulative epsilon" in detail["hint"]
    assert _buffered() == []


def test_upload_over_eight_is_422_and_not_buffered():
    r = _post("a", 1, privacy=dp(4.5, cumulative=8.4, rounds=2))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "privacy_budget_exceeded"
    assert "8.4" in r.json()["detail"]["hint"] and "8" in r.json()["detail"]["hint"]
    assert _buffered() == []
    # exactly the budget is allowed
    assert _post("a", 1, privacy=dp(4.0, cumulative=8.0, rounds=2)).status_code == 201


def test_upload_claiming_a_third_round_is_422():
    r = _post("a", 1, privacy=dp(1.0, cumulative=3.0, rounds=3))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "privacy_budget_exceeded"
    assert "3" in r.json()["detail"]["hint"] and "2 rounds" in r.json()["detail"]["hint"]
    assert _buffered() == []


def test_a_cumulative_below_the_round_is_422():
    r = _post("a", 1, privacy={"epsilon": 4.0, "epsilon_cumulative": 3.0, "rounds_accounted": 1})
    assert r.status_code == 422
    assert _buffered() == []


def test_a_refused_upload_does_not_replace_the_one_already_buffered():
    assert _post("a", 1, privacy=dp(3.9)).status_code == 201
    assert _post("a", 2, privacy=dp(4.5, cumulative=8.4, rounds=2)).status_code == 422
    stored = server._clusters["web"].uploads["a"]
    assert stored.epsilon == 3.9 and stored.epsilon_cumulative == 3.9


# ---- the cluster's epsilon ------------------------------------------------------
def test_cluster_epsilon_is_max_of_client_cumulative_not_a_sum():
    """Two rounds at 3.9 then 7.8 cumulative: the cluster reports 7.8, never 11.7."""
    for who, seed in (("a", 1), ("b", 2)):
        assert _post(who, seed, privacy=dp(3.9)).status_code == 201
    first = client.post("/clusters/web/aggregate", json={}).json()
    assert first["epsilon"] == 3.9
    promote(client, "web", first)

    for who, seed, cumulative in (("a", 3, 7.8), ("b", 4, 7.1)):
        r = _residual(who, seed, first, privacy=dp(3.9, cumulative=cumulative, rounds=2))
        assert r.status_code == 201, r.text
    second = client.post("/clusters/web/aggregate", json={}).json()
    assert second["epsilon"] == 7.8
    manifest = client.get("/clusters/web/aggregate/manifest").json()
    assert manifest["epsilon"] == 7.8 and manifest["epsilon_this_round"] == 3.9
    assert manifest["epsilon_composition"] == (
        "per-client accountant composition over the run; cluster epsilon is the max over clients")
    assert manifest["client_privacy"] == [
        {"client_id": "a", "epsilon_round": 3.9, "epsilon_cumulative": 7.8, "rounds_accounted": 2},
        {"client_id": "b", "epsilon_round": 3.9, "epsilon_cumulative": 7.1, "rounds_accounted": 2},
    ]
    assert second["client_privacy"] == manifest["client_privacy"]


def test_a_client_that_sat_out_round_two_still_counts_through_the_base():
    """The merged adapter contains every round under it: a client with a larger
    spend in round 1 that did not send in round 2 keeps the cluster's epsilon up."""
    for who, seed, eps in (("a", 1, 2.0), ("b", 2, 6.0)):
        assert _post(who, seed, privacy=dp(eps)).status_code == 201
    first = client.post("/clusters/web/aggregate", json={}).json()
    assert first["epsilon"] == 6.0
    promote(client, "web", first)
    assert _residual("a", 3, first, privacy=dp(2.0, cumulative=3.1, rounds=2)).status_code == 201
    second = client.post("/clusters/web/aggregate", json={"min_clients": 1}).json()
    assert second["epsilon"] == 6.0


def test_non_dp_upload_still_makes_cluster_epsilon_none():
    assert _post("a", 1, privacy=dp(3.9)).status_code == 201
    assert _post("b", 2).status_code == 201  # trained without DP: nothing to report
    merged = client.post("/clusters/web/aggregate", json={}).json()
    assert merged["epsilon"] is None
    manifest = client.get("/clusters/web/aggregate/manifest").json()
    assert manifest["epsilon_not_reported_by"] == ["b"]
    assert manifest["client_privacy"] == [
        {"client_id": "a", "epsilon_round": 3.9, "epsilon_cumulative": 3.9, "rounds_accounted": 1},
        {"client_id": "b", "epsilon_round": None, "epsilon_cumulative": None,
         "rounds_accounted": None},
    ]


# ---- what the registry is told ---------------------------------------------------
class _Registry:
    """Stands in for the registry's HTTP endpoint: records what publish sends."""

    def __init__(self, monkeypatch):
        import json

        import httpx

        self.posts = []
        outer = self

        class _Resp:
            status_code = 201
            text = ""

            def json(self):
                return {"name": "web", "version": 1}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, files=None, data=None, **_):
                outer.posts.append({"url": url, "meta": json.loads(data["meta"])})
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)


def test_publish_meta_carries_client_privacy(monkeypatch):
    registry = _Registry(monkeypatch)
    for who, seed, eps in (("a", 1, 3.9), ("b", 2, 3.2)):
        assert _post(who, seed, privacy=dp(eps)).status_code == 201
    assert client.post("/clusters/web/aggregate", json={}).status_code == 200
    assert client.post("/adapters/web/publish").status_code == 201
    meta = registry.posts[0]["meta"]
    assert meta["privacy"] == {"epsilon": 3.9, "epsilon_cumulative": 3.9}
    assert meta["client_privacy"] == [
        {"client_id": "a", "epsilon_round": 3.9, "epsilon_cumulative": 3.9, "rounds_accounted": 1},
        {"client_id": "b", "epsilon_round": 3.2, "epsilon_cumulative": 3.2, "rounds_accounted": 1},
    ]


def test_publish_without_dp_sends_no_privacy_but_names_the_clients(monkeypatch):
    registry = _Registry(monkeypatch)
    for who, seed in (("a", 1), ("b", 2)):
        assert _post(who, seed).status_code == 201
    assert client.post("/clusters/web/aggregate", json={}).status_code == 200
    assert client.post("/adapters/web/publish").status_code == 201
    meta = registry.posts[0]["meta"]
    assert "privacy" not in meta
    assert [row["client_id"] for row in meta["client_privacy"]] == ["a", "b"]
    assert all(row["epsilon_cumulative"] is None for row in meta["client_privacy"])
