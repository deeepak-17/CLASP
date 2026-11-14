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
