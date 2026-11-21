"""Decision U11: an aggregate is a candidate until it is promoted.

``POST .../aggregate`` stores a candidate and leaves ``active`` (what edges pull
and train on) alone; ``promote`` makes it active, ``reject`` drops it. Upload
helpers come from ``test_fedprox_server`` and ``http_helpers``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cluster import server
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    promote,
    reset_server_state,
    restore_server_state,
    upload_body,
)
from tests.test_fedprox_server import _based, _round0

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _decide(action, sha, cid="web"):
    return client.post(f"/clusters/{cid}/adapters/candidate/{action}", json={"sha256": sha})


def test_aggregate_creates_a_candidate_and_active_is_unchanged():
    cand = _round0(promoted=False)
    assert client.get("/clusters/web/adapters/candidate").json()["sha256"] == cand["sha256"]
    assert client.get("/clusters/web/adapters/active").status_code == 404
    assert server._clusters["web"].active is None
    # a second round on top of a promoted adapter leaves the promoted one served
    promote(client, "web", cand)
    for c in ("a", "b"):
        assert client.post("/clusters/web/uploads",
                           json=_based(c, trained_adapter(20), cand)).status_code == 201
    cand2 = client.post("/clusters/web/aggregate", json={}).json()
    assert cand2["sha256"] != cand["sha256"]
    assert client.get("/clusters/web/adapters/active").json()["sha256"] == cand["sha256"]
    assert client.get("/adapters/web/download").headers["X-CLASP-Round"] == str(cand["round_id"])
    assert client.get("/clusters/web/aggregate/manifest").json()["sha256"] == cand2["sha256"]
    assert server._clusters["web"].last_round  # re-clustering evidence still set


def test_active_is_404_until_the_first_candidate_is_promoted():
    cand = _round0(promoted=False)
    assert client.get("/clusters/web/adapters/active").status_code == 404
    assert client.get("/adapters/web/active").status_code == 404
    assert client.get("/adapters/web/download").status_code == 404
    assert client.get("/clusters").json()["clusters"]["web"]["has_active_adapter"] is False
    promote(client, "web", cand)
    assert client.get("/clusters/web/adapters/active").json()["sha256"] == cand["sha256"]


def test_promote_makes_the_candidate_active_and_clears_it():
    cand = _round0(promoted=False)
    r = _decide("promote", cand["sha256"])
    assert r.status_code == 200, r.text
    assert r.json() == {"active_sha256": cand["sha256"], "round_id": cand["round_id"],
                         "dropped_uploads": []}
    assert client.get("/clusters/web/adapters/active").json()["sha256"] == cand["sha256"]
    assert client.get("/clusters/web/adapters/candidate").status_code == 404
    assert server._clusters["web"].candidate is None


def test_reject_drops_the_candidate_and_keeps_active():
    cand = _round0(promoted=False)
    promote(client, "web", cand)
    for c in ("a", "b"):
        client.post("/clusters/web/uploads", json=_based(c, trained_adapter(21), cand))
    cand2 = client.post("/clusters/web/aggregate", json={}).json()
    r = _decide("reject", cand2["sha256"])
    assert r.status_code == 200, r.text
    assert r.json() == {"active_sha256": cand["sha256"]}
    assert client.get("/clusters/web/adapters/candidate").status_code == 404
    assert client.get("/clusters/web/adapters/active").json()["sha256"] == cand["sha256"]
    # a rejected round keeps its number: the next aggregate takes the next one
    assert server._clusters["web"].round_id == cand2["round_id"] + 1
    for c in ("a", "b"):
        r = client.post("/clusters/web/uploads", json=_based(
            c, trained_adapter(22), cand, round_id=server._clusters["web"].round_id))
        assert r.status_code == 201, r.text
    assert client.post("/clusters/web/aggregate", json={}).json()["round_id"] == cand2["round_id"] + 1


def test_promote_or_reject_with_the_wrong_sha_is_409():
    cand = _round0(promoted=False)
    for action in ("promote", "reject"):
        r = _decide(action, "0" * 64)
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["sha256"] == cand["sha256"]
    assert server._clusters["web"].candidate is not None
    assert server._clusters["web"].active is None
    promote(client, "web", cand)
    # nothing pending any more: both are 409 again
    assert _decide("promote", cand["sha256"]).status_code == 409
    assert _decide("reject", cand["sha256"]).status_code == 409


def test_second_aggregate_while_a_candidate_is_pending_is_409_and_keeps_uploads():
    cand = _round0(promoted=False)
    rnd = server._clusters["web"].round_id
    assert client.post("/clusters/web/uploads",
                       json=upload_body("a", trained_adapter(30), round_id=rnd)).status_code == 201
    r = client.post("/clusters/web/aggregate", json={})
    assert r.status_code == 409
    assert r.json()["detail"] == {"error": "candidate_pending", "sha256": cand["sha256"]}
    assert set(server._clusters["web"].uploads) == {"a"}
    assert server._clusters["web"].round_id == rnd


def test_round_two_uploads_must_declare_the_promoted_adapter_not_the_candidate():
    promoted = _round0()
    for c in ("a", "b"):
        client.post("/clusters/web/uploads", json=_based(c, trained_adapter(40), promoted))
    cand = client.post("/clusters/web/aggregate", json={}).json()
    rnd = server._clusters["web"].round_id
    # trained on the candidate: refused, the edges are still serving `promoted`
    bad = client.post("/clusters/web/uploads", json=_based(
        "a", trained_adapter(41), cand, round_id=rnd))
    assert bad.status_code == 409 and "stale_adapter" in bad.text
    # trained on the promoted adapter: accepted and buffered for the next round
    ok = client.post("/clusters/web/uploads", json=_based(
        "a", trained_adapter(41), promoted, round_id=rnd))
    assert ok.status_code == 201, ok.text
    assert set(server._clusters["web"].uploads) == {"a"}


def test_after_a_first_round_reject_uploads_without_a_base_are_accepted_again():
    cand = _round0(promoted=False)
    assert _decide("reject", cand["sha256"]).status_code == 200
    assert server._clusters["web"].active is None
    rnd = server._clusters["web"].round_id
    r = client.post("/clusters/web/uploads",
                    json=upload_body("a", trained_adapter(50), round_id=rnd))
    assert r.status_code == 201, r.text
    # and one declaring a base is still refused: there is no merged adapter
    r = client.post("/clusters/web/uploads", json=_based(
        "b", trained_adapter(51), cand, round_id=rnd))
    assert r.status_code == 409


def test_publish_sends_the_candidate_with_set_active_false(monkeypatch):
    import httpx

    sent = {}

    class _Resp:
        status_code = 201
        text = "{}"

        def json(self):
            return {"ref": {"name": "cluster-web", "version": 1}}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, files=None, data=None, **kw):
            sent["meta"] = json.loads(data["meta"])
            sent["blob"] = files["file"][1]
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    # nothing aggregated yet: nothing to publish
    create_cluster(client, "web", members={"web": ["a", "b"]})
    assert client.post("/adapters/web/publish", json={}).status_code == 404
    cand = _round0(promoted=False)
    r = client.post("/adapters/web/publish", json={"adapter_name": "cluster-web"})
    assert r.status_code == 201, r.text
    assert sent["meta"]["set_active"] is False
    import hashlib
    assert hashlib.sha256(sent["blob"]).hexdigest() == cand["sha256"]
    # once promoted there is no candidate left to publish
    promote(client, "web", cand)
    assert client.post("/adapters/web/publish", json={}).status_code == 404


def test_clusters_listing_reports_candidate_sha256():
    create_cluster(client, "web", members={"web": ["a", "b"]})
    row = client.get("/clusters").json()["clusters"]["web"]
    assert row["candidate_sha256"] is None and row["active_sha256"] is None
    cand = _round0(promoted=False)
    row = client.get("/clusters").json()["clusters"]["web"]
    assert row["candidate_sha256"] == cand["sha256"] and row["active_sha256"] is None
    promote(client, "web", cand)
    row = client.get("/clusters").json()["clusters"]["web"]
    assert row["candidate_sha256"] is None and row["active_sha256"] == cand["sha256"]


def test_promote_drops_uploads_trained_on_the_replaced_adapter_and_the_next_aggregate_works():
    promoted = _round0()
    for c in ("a", "b"):
        client.post("/clusters/web/uploads", json=_based(c, trained_adapter(60), promoted))
    cand = client.post("/clusters/web/aggregate", json={}).json()
    rnd = server._clusters["web"].round_id
    # edges still serve `promoted` while the candidate is evaluated
    assert client.post("/clusters/web/uploads", json=_based(
        "a", trained_adapter(61), promoted, round_id=rnd)).status_code == 201
    r = _decide("promote", cand["sha256"])
    assert r.status_code == 200, r.text
    assert r.json()["dropped_uploads"] == ["a"]
    assert server._clusters["web"].uploads == {}
    for c in ("a", "b"):
        assert client.post("/clusters/web/uploads", json=_based(
            c, trained_adapter(62), cand, round_id=rnd)).status_code == 201
    assert client.post("/clusters/web/aggregate", json={}).status_code == 200


def test_promoting_the_first_candidate_drops_full_uploads_buffered_while_it_waited():
    cand = _round0(promoted=False)
    rnd = server._clusters["web"].round_id
    # nothing is served yet, so a bare-base (full) adapter is accepted and buffered
    assert client.post("/clusters/web/uploads",
                       json=upload_body("a", trained_adapter(80), round_id=rnd)).status_code == 201
    r = _decide("promote", cand["sha256"])
    assert r.status_code == 200, r.text
    assert r.json()["dropped_uploads"] == ["a"]
    assert server._clusters["web"].uploads == {}
    # round 2 builds on the promoted adapter, as a cumulative round
    for c, seed in (("a", 81), ("b", 82)):
        assert client.post("/clusters/web/uploads", json=_based(
            c, trained_adapter(seed), cand, round_id=rnd)).status_code == 201
    agg = client.post("/clusters/web/aggregate", json={})
    assert agg.status_code == 200, agg.text
    assert agg.json()["update_rule"] == "fedprox_cumulative"
    assert agg.json()["base_sha256"] == cand["sha256"]


def test_reject_keeps_uploads_buffered_during_the_candidate_and_the_next_aggregate_uses_them():
    promoted = _round0()
    for c in ("a", "b"):
        client.post("/clusters/web/uploads", json=_based(c, trained_adapter(70), promoted))
    cand = client.post("/clusters/web/aggregate", json={}).json()
    rnd = server._clusters["web"].round_id
    assert client.post("/clusters/web/uploads", json=_based(
        "a", trained_adapter(71), promoted, round_id=rnd)).status_code == 201
    assert _decide("reject", cand["sha256"]).status_code == 200
    assert set(server._clusters["web"].uploads) == {"a"}
    r = client.post("/clusters/web/aggregate", json={})
    assert r.status_code == 200, r.text
    assert r.json()["source_clients"] == ["a"] and r.json()["round_id"] == rnd
