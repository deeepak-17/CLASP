"""Who may call which cluster route (roles from certificate names, U5).

The routes that drive a round or change what is served are for the panel
(certificate ``demo-ui``, role ``svc:evaluation``) and an administrator.
Uploads keep their own rule (the certificate name must equal ``client_id``) and
reading the active adapter keeps the membership rule. Without a certificate
name (plain HTTP) every route is open, as before.
"""
from __future__ import annotations

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

client = TestClient(server.app)
HEADER = "x-test-client-cn"
NAMES = {"svc:cluster": "cluster", "svc:evaluation": "demo-ui", "admin": "clasp-admin",
         "edge": "client-flask", "unknown": "clasp-dev-client"}
OPERATORS = {"svc:evaluation", "admin"}
SHA = "0" * 64

#: The operator routes: (method, route template, a concrete path, a body).
RESTRICTED = [
    ("POST", "/aggregate", "/aggregate", {}),
    ("POST", "/adapters/{cluster_id}/publish", "/adapters/web/publish", {}),
    ("PUT", "/clusters/{cluster_id}/members", "/clusters/web/members",
     {"client_ids": ["client-flask"]}),
    ("POST", "/clusters/{cluster_id}/aggregate", "/clusters/web/aggregate", {}),
    ("GET", "/clusters/{cluster_id}/adapters/candidate", "/clusters/web/adapters/candidate", None),
    ("POST", "/clusters/{cluster_id}/adapters/candidate/promote",
     "/clusters/web/adapters/candidate/promote", {"sha256": SHA}),
    ("POST", "/clusters/{cluster_id}/adapters/candidate/reject",
     "/clusters/web/adapters/candidate/reject", {"sha256": SHA}),
    ("POST", "/clusters/{cluster_id}/adapters/active/restore",
     "/clusters/web/adapters/active/restore", {}),
    ("POST", "/recluster", "/recluster", {}),
]
#: Routes with their own rules (or none): never refused for the caller's role.
UNRESTRICTED = {
    ("GET", "/healthz"), ("POST", "/uploads"), ("GET", "/adapters/cluster/active"),
    ("GET", "/adapters/{cluster_id}/active"), ("GET", "/adapters/{cluster_id}/download"),
    ("GET", "/aggregate/manifest"), ("GET", "/adapters/{cluster_id}/manifest"),
    ("GET", "/clusters"), ("POST", "/clusters/{cluster_id}/uploads"),
    ("GET", "/clusters/{cluster_id}/adapters/active"),
    ("GET", "/clusters/{cluster_id}/aggregate/manifest"),
}
CASES = [(method, path, body, role, role in OPERATORS)
         for method, _, path, body in RESTRICTED for role in sorted(NAMES)]


def _provider(request):
    return request.headers.get(HEADER)


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    create_cluster(client, "web")
    yield
    restore_server_state(saved)


def _call(method, path, body, name=None):
    headers = {HEADER: name} if name is not None else {}
    return client.request(method, path, json=body, headers=headers)


def test_the_policy_names_every_route():
    built_in = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    served = {(m, r.path) for r in server.app.routes if getattr(r, "methods", None)
              for m in r.methods - {"HEAD", "OPTIONS"} if r.path not in built_in}
    assert served == UNRESTRICTED | {(method, template) for method, template, _, _ in RESTRICTED}


@pytest.mark.parametrize("method, path, body, role, allowed", CASES)
def test_operator_routes_admit_exactly_the_operator_roles(method, path, body, role, allowed):
    server.configure_identity(_provider)
    status = _call(method, path, body, NAMES[role]).status_code
    if allowed:
        assert status != 403, f"{role} should be allowed on {method} {path}, got {status}"
        assert status < 500
    else:
        assert status == 403, f"{role} should be refused on {method} {path}, got {status}"


@pytest.mark.parametrize("method, path, body", [(m, p, b) for m, _, p, b in RESTRICTED])
def test_with_no_identity_provider_every_route_behaves_as_before(method, path, body):
    assert _call(method, path, body, "client-flask").status_code != 403  # header ignored


@pytest.mark.parametrize("method, path, body", [(m, p, b) for m, _, p, b in RESTRICTED])
def test_a_provider_that_finds_no_certificate_leaves_the_route_open(method, path, body):
    """Plain HTTP behind a provider: no name, no role to check (the fall-back)."""
    server.configure_identity(_provider)
    assert _call(method, path, body).status_code != 403


def test_an_edge_still_uploads_as_itself_and_reads_its_cluster():
    server.configure_identity(_provider, require=True)
    assert client.put("/clusters/web/members", json={"client_ids": ["client-flask"]},
                      headers={HEADER: "demo-ui"}).status_code == 200
    up = client.post("/clusters/web/uploads", json=upload_body("client-flask", trained_adapter(1)),
                     headers={HEADER: "client-flask"})
    assert up.status_code == 201, up.text
    assert client.get("/clusters", headers={HEADER: "client-flask"}).status_code == 200
    # ... and may not merge the round, read the candidate, or change who is a member
    for method, _, path, body in RESTRICTED:
        assert _call(method, path, body, "client-flask").status_code == 403, path
    assert sorted(server._clusters["web"].uploads) == ["client-flask"]


def test_a_refused_aggregate_keeps_the_buffer_and_names_the_caller():
    server.configure_identity(_provider)
    client.post("/clusters/web/uploads", json=upload_body("client-flask", trained_adapter(1)),
                headers={HEADER: "client-flask"})
    r = _call("POST", "/clusters/web/aggregate", {}, "cluster")
    assert r.status_code == 403
    assert "'cluster'" in r.json()["detail"] and "svc:evaluation" in r.json()["detail"]
    assert server._clusters["web"].uploads and server._clusters["web"].candidate is None


def test_role_checks_do_not_depend_on_the_upload_binding_switch(monkeypatch):
    """CLASP_CLUSTER_BIND_IDENTITY=0 installs no provider; the verified peer
    name in the scope still decides the operator routes."""
    from security.mtls.peer import SCOPE_KEY
    from starlette.requests import Request

    def request(name):
        return Request({"type": "http", "headers": [], SCOPE_KEY: name})

    server.configure_identity(None)
    assert server._role_identity(request("client-flask")) == "client-flask"
    assert server._role_identity(Request({"type": "http", "headers": []})) is None
