"""HTTP service: concurrency, unknown-cluster 404, and the quorum denominator.

The handlers are plain ``def`` functions, which FastAPI runs on a thread pool, so
an upload can arrive while another request is aggregating. These tests drive the
real app from real threads (no mocks of the lock): the aggregate is slowed down
at the point where ``aggregate_svd`` runs, which is where real adapters spend
seconds.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from cluster import server
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    reset_server_state,
    restore_server_state,
    upload_body,
)

JOIN_S = 60  # generous: only a deadlock can hit it


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _client() -> TestClient:
    return TestClient(server.app)


def _upload(client, cluster, who, seed, round_id=0):
    return client.post(
        f"/clusters/{cluster}/uploads",
        json=upload_body(who, trained_adapter(seed), round_id=round_id),
    )


# ---- unknown cluster: 404, and nothing is created ---------------------------


def test_upload_to_an_unknown_cluster_is_a_404_and_creates_nothing():
    client = _client()
    r = _upload(client, "web-clsuter", "c1", 1)  # typo'd id
    assert r.status_code == 404 and "unknown cluster" in r.json()["detail"]
    assert "web-clsuter" not in server._clusters
    assert "web-clsuter" not in client.get("/clusters").json()["clusters"]


def test_only_put_members_creates_a_cluster():
    client = _client()
    assert _upload(client, "team-a", "c1", 1).status_code == 404
    assert client.put("/clusters/team-a/members", json={"client_ids": ["c1"]}).status_code == 200
    assert _upload(client, "team-a", "c1", 1).status_code == 201
    assert client.post("/clusters/team-x/aggregate").status_code == 404


def test_an_invalid_cluster_id_is_still_a_422():
    assert _upload(_client(), "bad%20id", "c1", 1).status_code == 422


def test_the_default_cluster_still_takes_uploads_without_registration():
    r = _client().post("/uploads", json=upload_body("c1", trained_adapter(1)))
    assert r.status_code == 201


# ---- min_fraction is measured against the clients the round expected --------


def _members(client, cluster, ids):
    create_cluster(client, cluster, members={cluster: list(ids)})


def test_min_fraction_uses_the_clusters_member_count():
    client = _client()
    _members(client, "team-a", ["c1", "c2", "c3", "c4"])
    for i, who in enumerate(("c1", "c2")):  # 2 of 4 members uploaded
        assert _upload(client, "team-a", who, i).status_code == 201
    r = client.post("/clusters/team-a/aggregate", json={"min_fraction": 0.75})
    assert r.status_code == 409
    assert "3 required" in r.json()["detail"] and "of 4 expected" in r.json()["detail"]
    assert server._clusters["team-a"].uploads.keys() == {"c1", "c2"}  # buffer kept
    assert _upload(client, "team-a", "c3", 3).status_code == 201  # now 3 of 4
    ok = client.post("/clusters/team-a/aggregate", json={"min_fraction": 0.75})
    assert ok.status_code == 200
    manifest = client.get("/clusters/team-a/aggregate/manifest").json()
    assert manifest["expected_clients"] == 4 and manifest["num_clients"] == 3


def test_without_members_min_fraction_falls_back_to_the_arrived_uploads():
    client = _client()
    create_cluster(client, "team-a")  # created, nobody assigned
    for i, who in enumerate(("c1", "c2")):
        assert _upload(client, "team-a", who, i).status_code == 201
    # nothing says how many were expected -> the fraction cannot bite (documented)
    ok = client.post("/clusters/team-a/aggregate", json={"min_fraction": 1.0})
    assert ok.status_code == 200
    assert client.get("/clusters/team-a/aggregate/manifest").json()["expected_clients"] == 2


def test_explicit_expected_clients_overrides_the_member_count():
    client = _client()
    create_cluster(client, "team-a")
    for i, who in enumerate(("c1", "c2")):
        assert _upload(client, "team-a", who, i).status_code == 201
    r = client.post(
        "/clusters/team-a/aggregate", json={"min_fraction": 0.5, "expected_clients": 6}
    )
    assert r.status_code == 409 and "3 required" in r.json()["detail"]
    assert client.post("/clusters/team-a/aggregate", json={"expected_clients": 0}).status_code == 422


def test_expected_never_drops_below_the_uploads_that_arrived():
    client = _client()
    _members(client, "team-a", ["c1"])  # one member, but two clients upload
    for i, who in enumerate(("c1", "stranger")):
        assert _upload(client, "team-a", who, i).status_code == 201
    assert client.post("/clusters/team-a/aggregate", json={"min_fraction": 1.0}).status_code == 200
    assert client.get("/clusters/team-a/aggregate/manifest").json()["expected_clients"] == 2


# ---- an upload during a running aggregate is not accepted-then-lost ---------


class _SlowAggregate:
    """Wraps ``aggregate_svd``: signals when it starts, blocks until released."""

    def __init__(self, real):
        self.real = real
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, adapters, weights, **kw):
        self.started.set()
        assert self.release.wait(JOIN_S), "test never released the aggregate"
        return self.real(adapters, weights, **kw)


def test_an_upload_during_an_aggregate_waits_and_is_then_refused_not_lost(monkeypatch):
    slow = _SlowAggregate(server.aggregate_svd)
    monkeypatch.setattr(server, "aggregate_svd", slow)
    setup = _client()
    create_cluster(setup, "team-a")
    for i, who in enumerate(("c1", "c2")):
        assert _upload(setup, "team-a", who, i).status_code == 201

    agg: dict[str, object] = {}
    late: dict[str, object] = {}
    t_agg = threading.Thread(
        target=lambda: agg.update(r=_client().post("/clusters/team-a/aggregate")), daemon=True
    )
    t_late = threading.Thread(
        target=lambda: late.update(r=_upload(_client(), "team-a", "c3", 9)), daemon=True
    )
    t_agg.start()
    assert slow.started.wait(JOIN_S)  # the aggregate is now inside aggregate_svd
    t_late.start()
    t_late.join(0.5)
    assert t_late.is_alive(), "the late upload must wait for the running aggregate"
    assert "r" not in late  # and it has NOT been answered 201 yet

    slow.release.set()
    t_agg.join(JOIN_S)
    t_late.join(JOIN_S)
    assert not t_agg.is_alive() and not t_late.is_alive()

    assert agg["r"].status_code == 200
    manifest = setup.get("/clusters/team-a/aggregate/manifest").json()
    assert manifest["source_clients"] == ["c1", "c2"]
    # round 0 was consumed while c3 waited: an honest 409, not a lost 201
    assert late["r"].status_code == 409 and "round_id" in late["r"].json()["detail"]
    state = server._clusters["team-a"]
    assert state.uploads == {} and state.round_id == 1
    # c3 simply retries for the new round and is buffered for the next aggregate
    assert _upload(setup, "team-a", "c3", 9, round_id=1).status_code == 201
    assert state.uploads.keys() == {"c3"}


def test_stress_every_accepted_upload_is_aggregated_exactly_once(monkeypatch):
    """Many uploaders race a repeatedly-aggregating thread. Each client keeps
    retrying (409 -> read the round again) until it gets its single 201. Every
    one of those 201s must be counted by exactly one aggregate."""
    real = server.aggregate_svd
    aggregated: list[int] = []

    def counting(adapters, weights, **kw):
        aggregated.append(len(weights))
        return real(adapters, weights, **kw)

    monkeypatch.setattr(server, "aggregate_svd", counting)
    setup = _client()
    create_cluster(setup, "team-a")
    n_clients = 12
    errors: list[BaseException] = []
    accepted = []
    state = server._clusters["team-a"]
    done = threading.Event()

    def uploader(idx: int) -> None:
        try:
            client = _client()
            while True:
                r = _upload(client, "team-a", f"c{idx}", idx, round_id=state.round_id)
                if r.status_code == 201:
                    accepted.append(idx)
                    return
                assert r.status_code == 409, r.text  # only a round race is acceptable
        except BaseException as exc:  # noqa: BLE001 - surfaced by the main thread
            errors.append(exc)

    def aggregator() -> None:
        try:
            client = _client()
            while not done.is_set():
                r = client.post("/clusters/team-a/aggregate")
                assert r.status_code in (200, 400), r.text  # 400 = nothing buffered yet
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    uploaders = [threading.Thread(target=uploader, args=(i,), daemon=True) for i in range(n_clients)]
    agg = threading.Thread(target=aggregator, daemon=True)
    agg.start()
    for t in uploaders:
        t.start()
    for t in uploaders:
        t.join(JOIN_S)
    done.set()
    agg.join(JOIN_S)
    assert not any(t.is_alive() for t in [*uploaders, agg]), "deadlock"
    assert not errors, errors
    if state.uploads:  # drain whatever arrived after the last aggregate
        assert setup.post("/clusters/team-a/aggregate").status_code == 200
    assert sorted(accepted) == list(range(n_clients))
    assert sum(aggregated) == n_clients  # none lost, none aggregated twice


# ---- readers never trip over a concurrent writer ----------------------------


def test_listing_clusters_while_others_are_created_and_moved():
    setup = _client()
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            client = _client()
            for i in range(150):
                r = client.put(f"/clusters/c{i}/members", json={"client_ids": [f"x{i}", "mover"]})
                assert r.status_code == 200, r.text
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            client = _client()
            while not stop.is_set():
                assert client.get("/clusters").status_code == 200
                assert client.get("/healthz").status_code == 200
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, daemon=True)] + [
        threading.Thread(target=reader, daemon=True) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(JOIN_S)
    assert not any(t.is_alive() for t in threads) and not errors, errors
    assert setup.get("/clusters").json()["clusters"]["c149"]["members"] == ["mover", "x149"]


def test_health_and_listing_answer_while_an_aggregate_is_running(monkeypatch):
    slow = _SlowAggregate(server.aggregate_svd)
    monkeypatch.setattr(server, "aggregate_svd", slow)
    setup = _client()
    create_cluster(setup, "team-a")
    assert _upload(setup, "team-a", "c1", 1).status_code == 201
    t = threading.Thread(
        target=lambda: _client().post("/clusters/team-a/aggregate"), daemon=True
    )
    t.start()
    try:
        assert slow.started.wait(JOIN_S)
        # a liveness probe must not queue behind a multi-second aggregate
        assert setup.get("/healthz").status_code == 200
        assert "team-a" in setup.get("/clusters").json()["clusters"]
        assert setup.get("/clusters/team-a/adapters/active").status_code == 404  # none yet
    finally:
        slow.release.set()
        t.join(JOIN_S)
    assert not t.is_alive()


def test_a_client_moved_during_an_aggregate_does_not_leak_into_its_old_cluster():
    """set_members drops the moved client's buffered upload under the old
    cluster's lock, so it waits for a running aggregate instead of racing it."""
    client = _client()
    create_cluster(client, "team-a", "team-b", members={"team-a": ["c1"]})
    assert _upload(client, "team-a", "c1", 1).status_code == 201
    assert client.put("/clusters/team-b/members", json={"client_ids": ["c1"]}).status_code == 200
    assert server._clusters["team-a"].uploads == {}  # dropped on the move
    assert _upload(client, "team-a", "c1", 1).status_code == 403  # and cannot come back
