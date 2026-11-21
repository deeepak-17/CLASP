"""GET /sync/pending and POST /sync/{id}/done: the cluster step of a decision is retried until it lands."""
from __future__ import annotations

from .test_composite import make_adapter
from .test_composite_api import _save
from .test_promote_api import _promote_body


def _candidate(client, name="cluster-web", *, kind="cluster", cluster_id="web"):
    _save(client, name, make_adapter(1), kind=kind, cluster_id=cluster_id, set_active=False)


def _decide(client, name="cluster-web", *, good=True, version=1, kind="cluster"):
    body = _promote_body(name, version, good=good)
    body["eval"]["adapter"].update(kind=kind, cluster_id="web")
    r = client.post(f"/adapters/{name}/promote", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _pending(client):
    r = client.get("/sync/pending")
    assert r.status_code == 200
    return r.json()["pending"]


def test_nothing_is_pending_on_a_fresh_registry(client):
    assert _pending(client) == []


def test_a_promote_of_a_cluster_adapter_creates_one_pending_row(client):
    _candidate(client)
    sha = client.get("/adapters/cluster-web/versions/1").json()["sha256"]
    assert _decide(client)["action"] == "promote"
    (row,) = _pending(client)
    assert row["adapter"] == "cluster-web" and row["version"] == 1
    assert row["cluster_id"] == "web" and row["sha256"] == sha
    assert row["action"] == "promote" and row["attempts"] == 0 and isinstance(row["id"], int)


def test_a_reject_creates_a_pending_reject_row(client):
    _candidate(client)
    assert _decide(client, good=False)["action"] == "reject"
    (row,) = _pending(client)
    assert row["action"] == "reject" and row["adapter"] == "cluster-web"


def test_an_adapter_without_a_cluster_id_creates_no_row(client):
    _save(client, "flask", make_adapter(1), kind="client", set_active=False)
    assert _decide(client, "flask", kind="client")["action"] == "promote"
    assert _pending(client) == []


def test_a_client_adapter_with_a_cluster_id_creates_no_row(client):
    # client adapters routinely carry their cluster's id (demo seed: flask -> web);
    # only a cluster adapter's decision is a cluster step
    _save(client, "flask", make_adapter(1), kind="client", cluster_id="web", set_active=False)
    decided = _decide(client, "flask", kind="client")
    assert decided["action"] == "promote" and decided["cluster_sync_id"] is None
    _save(client, "flask", make_adapter(2), kind="client", cluster_id="web", set_active=False)
    assert _decide(client, "flask", kind="client", version=2, good=False)["cluster_sync_id"] is None
    assert _pending(client) == []


def test_done_ok_empties_the_list(client):
    _candidate(client)
    _decide(client)
    (row,) = _pending(client)
    r = client.post(f"/sync/{row['id']}/done", json={"ok": True, "error": None})
    assert r.status_code == 200
    assert _pending(client) == []
    # idempotent: telling it again is not an error
    assert client.post(f"/sync/{row['id']}/done", json={"ok": True}).status_code == 200


def test_done_not_ok_keeps_the_row_and_counts_the_attempt(client):
    _candidate(client)
    _decide(client)
    (row,) = _pending(client)
    r = client.post(f"/sync/{row['id']}/done", json={"ok": False, "error": "cluster unreachable"})
    assert r.status_code == 200
    (row,) = _pending(client)
    assert row["attempts"] == 1 and row["last_error"] == "cluster unreachable"
    client.post(f"/sync/{row['id']}/done", json={"ok": False, "error": "again"})
    assert _pending(client)[0]["attempts"] == 2


def test_done_for_an_unknown_row_is_404(client):
    assert client.post("/sync/999/done", json={"ok": True}).status_code == 404


def test_done_body_is_validated(client):
    assert client.post("/sync/1/done", json={"error": "x"}).status_code == 422
    assert client.post("/sync/1/done", json={"ok": "yes"}).status_code == 422
    assert client.post("/sync/1/done", json={"ok": False, "error": 5}).status_code == 422


def test_promote_answers_with_the_id_of_its_sync_row(client):
    _candidate(client)
    decision = _decide(client)
    (row,) = _pending(client)
    assert decision["cluster_sync_id"] == row["id"]


def test_promote_without_a_cluster_id_answers_null(client):
    _save(client, "flask", make_adapter(1), kind="client", set_active=False)
    assert _decide(client, "flask", kind="client")["cluster_sync_id"] is None


def test_a_pending_row_names_the_later_done_row_that_supersedes_it(client):
    _candidate(client)
    first = _decide(client)["cluster_sync_id"]
    _save(client, "cluster-web", make_adapter(2), kind="cluster", cluster_id="web", set_active=False)
    second = _decide(client, version=2)["cluster_sync_id"]
    assert [r["superseded_by"] for r in _pending(client)] == [None, None]
    client.post(f"/sync/{second}/done", json={"ok": True})
    (row,) = _pending(client)
    assert row["id"] == first and row["superseded_by"] == second


def _done(client, sync_id):
    assert client.post(f"/sync/{sync_id}/done", json={"ok": True}).status_code == 200


def _promoted_cluster_versions(client, n=2):
    """n cluster versions, each published as a candidate, promoted, and synced."""
    for v in range(1, n + 1):
        _save(client, "cluster-web", make_adapter(v), kind="cluster", cluster_id="web",
              set_active=False)
        _done(client, _decide(client, version=v)["cluster_sync_id"])


def test_every_row_names_the_version_active_after_its_decision(client):
    _candidate(client)
    _decide(client)
    (row,) = _pending(client)
    assert row["active_version_after"] == 1


def test_a_restore_of_a_cluster_adapter_creates_a_pending_restore_row(client):
    """The cluster serves what the edges train on: a registry restore that it
    never hears about would leave it serving the version restored away from."""
    _promoted_cluster_versions(client)
    sha1 = client.get("/adapters/cluster-web/versions/1").json()["sha256"]
    r = client.post("/adapters/cluster-web/restore", json={"reason": "drill"})
    assert r.status_code == 200, r.text
    (row,) = _pending(client)
    assert r.json()["cluster_sync_id"] == row["id"]
    assert row["action"] == "restore" and row["cluster_id"] == "web"
    assert row["version"] == 2 and row["active_version_after"] == 1 and row["sha256"] == sha1


def test_a_restore_of_a_client_adapter_creates_no_row(client):
    for v in (1, 2):
        _save(client, "flask", make_adapter(v), kind="client", cluster_id="web", set_active=False)
        _decide(client, "flask", kind="client", version=v)
    r = client.post("/adapters/flask/restore", json={"reason": "drill"})
    assert r.status_code == 200, r.text
    assert r.json()["cluster_sync_id"] is None and _pending(client) == []
