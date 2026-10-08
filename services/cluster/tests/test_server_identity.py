"""P2-side client-identity hook (stub provider; real identity comes from P3 mTLS)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cluster import server
from tests.conftest import trained_adapter
from tests.http_helpers import reset_server_state, restore_server_state, upload_body

client = TestClient(server.app)
HEADER = "x-test-client-cn"  # stand-in for "CN of the verified client certificate"


def _stub_provider(request):
    return request.headers.get(HEADER)


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _post(cid, claimed, headers=None, cluster="team-a"):
    return client.post(
        f"/clusters/{cluster}/uploads",
        json=upload_body(claimed, trained_adapter(1)),
        headers=headers or {},
    )


def test_default_is_unchanged_self_asserted_identity():
    assert _post("c1", "c1").status_code == 201
    assert _post("c2", "someone-else", {HEADER: "ignored"}).status_code == 201  # no provider


def test_matching_authenticated_identity_is_accepted():
    server.configure_identity(_stub_provider, require=True)
    assert _post("c1", "c1", {HEADER: "c1"}).status_code == 201


def test_identity_spoofing_is_refused():
    server.configure_identity(_stub_provider)
    r = _post("c1", "victim", {HEADER: "attacker"})
    assert r.status_code == 403 and "may not upload as" in r.json()["detail"]
    assert server._clusters["team-a"].uploads == {}


def test_missing_identity_is_refused_only_when_required():
    server.configure_identity(_stub_provider, require=False)
    assert _post("c1", "c1").status_code == 201
    server.configure_identity(_stub_provider, require=True)
    r = _post("c2", "c2")
    assert r.status_code == 401


def test_a_failing_provider_fails_closed():
    def boom(_request):
        raise RuntimeError("cert store unavailable")

    server.configure_identity(boom)
    assert _post("c1", "c1", {HEADER: "c1"}).status_code == 401
    assert server._clusters["team-a"].uploads == {}


def test_require_without_a_provider_is_a_noop():
    server.configure_identity(None, require=True)
    assert _post("c1", "c1").status_code == 201


def test_members_cannot_read_another_clusters_adapter_but_unassigned_callers_can():
    for cid, ids in (("team-a", ["a1"]), ("team-b", ["b1"])):
        client.put(f"/clusters/{cid}/members", json={"client_ids": ids})
    for cid, who in (("team-a", "a1"), ("team-b", "b1")):
        client.post(f"/clusters/{cid}/uploads", json=upload_body(who, trained_adapter(2)))
        client.post(f"/clusters/{cid}/aggregate")
    server.configure_identity(_stub_provider)

    own = client.get("/clusters/team-a/adapters/active", headers={HEADER: "a1"})
    assert own.status_code == 200
    other = client.get("/clusters/team-b/adapters/active", headers={HEADER: "a1"})
    assert other.status_code == 403 and "may not read" in other.json()["detail"]
    # a registry / evaluation service has no membership: not restricted
    svc = client.get("/clusters/team-b/adapters/active", headers={HEADER: "evaluation-svc"})
    assert svc.status_code == 200
    # no identity at all (provider returns None, require off): not restricted either
    assert client.get("/clusters/team-b/adapters/active").status_code == 200
