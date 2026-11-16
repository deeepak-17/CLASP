"""Who may call which registry route (roles from certificate names, U5).

Services identify each other by the name in their mTLS certificate; the name
maps to a role (``security.auth.roles``) and each route names the roles allowed
to call it. Without TLS there is no name and every route is open, as before.
A certificate name is simulated here with a header through
``registry.app.configure_identity``; ``test_serve`` checks the real handshake.
"""
from __future__ import annotations

import ssl

import httpx
import pytest

from .test_serve import _cert, _key, _tls_env, _write

HEADER = "x-test-cn"
NAMES = {"svc:cluster": "cluster", "svc:evaluation": "demo-ui", "admin": "clasp-admin",
         "edge": "client-flask", "unknown": "clasp-dev-client"}
ALL = set(NAMES)
READERS = {"svc:cluster", "svc:evaluation", "admin"}

#: The whole policy: (method, route template, a concrete path, roles allowed).
POLICY = [
    ("GET", "/healthz", "/healthz", ALL),
    ("GET", "/adapters", "/adapters", READERS),
    ("GET", "/adapters/{name}/versions", "/adapters/x/versions", READERS),
    ("GET", "/adapters/{name}/versions/{version}", "/adapters/x/versions/1", READERS),
    ("GET", "/adapters/{name}/versions/{version}/file", "/adapters/x/versions/1/file", READERS),
    ("GET", "/adapters/{name}/active", "/adapters/x/active", READERS),
    ("GET", "/adapters/{name}/lineage", "/adapters/x/lineage", READERS),
    ("GET", "/adapters/{name}/promotions", "/adapters/x/promotions", READERS),
    ("GET", "/privacy/ledger", "/privacy/ledger", READERS),
    ("GET", "/sync/pending", "/sync/pending", READERS),
    ("POST", "/adapters/{name}/versions", "/adapters/x/versions", {"svc:cluster", "admin"}),
    ("POST", "/adapters/{name}/promote", "/adapters/x/promote", {"svc:evaluation", "admin"}),
    ("POST", "/adapters/{name}/compose", "/adapters/x/compose", {"svc:evaluation", "admin"}),
    ("POST", "/sync/{sync_id}/done", "/sync/1/done", {"svc:evaluation", "admin"}),
    ("POST", "/adapters/{name}/restore", "/adapters/x/restore", {"svc:evaluation", "admin"}),
    ("POST", "/adapters/{name}/gc", "/adapters/x/gc", {"admin"}),
    ("POST", "/gc", "/gc", {"admin"}),
]
CASES = [(method, path, role, role in allowed)
         for method, _, path, allowed in POLICY for role in sorted(ALL)]


@pytest.fixture
def api(client):
    import registry.app as appmod

    appmod.configure_identity(lambda request: request.headers.get(HEADER))
    yield client
    appmod.configure_identity(None)


def _call(api, method, path, name=None):
    headers = {HEADER: name} if name is not None else {}
    if method == "GET":
        return api.get(path, headers=headers)
    return api.post(path, json={}, headers=headers)


def test_the_policy_names_every_route():
    """A new route must be added to the policy before it ships."""
    import registry.app as appmod

    built_in = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    served = {(m, r.path) for r in appmod.app.routes if getattr(r, "methods", None)
              for m in r.methods - {"HEAD", "OPTIONS"} if r.path not in built_in}
    assert served == {(method, template) for method, template, _, _ in POLICY}


@pytest.mark.parametrize("method, path, role, allowed", CASES)
def test_each_route_admits_exactly_its_roles(api, method, path, role, allowed):
    status = _call(api, method, path, NAMES[role]).status_code
    if allowed:
        assert status != 403, f"{role} should be allowed on {method} {path}, got {status}"
        assert status < 500
    else:
        assert status == 403, f"{role} should be refused on {method} {path}, got {status}"


def test_an_edge_gets_403_from_every_route_except_healthz(api):
    for method, _, path, _ in POLICY:
        status = _call(api, method, path, "client-numpy").status_code
        assert (status == 403) == (path != "/healthz"), (method, path, status)


@pytest.mark.parametrize("method, path", [(m, p) for m, _, p, _ in POLICY])
def test_with_no_certificate_name_every_route_behaves_as_before(api, method, path):
    """The fall-back: plain HTTP has no certificate, so no role check applies."""
    assert _call(api, method, path).status_code != 403


def test_with_no_identity_provider_installed_nothing_is_refused(client):
    for method, _, path, _ in POLICY:
        assert _call(client, method, path, "client-flask").status_code != 403  # header ignored


def test_a_refusal_names_the_caller_and_does_nothing(api):
    r = api.post("/gc", json={}, headers={HEADER: "demo-ui"})
    assert r.status_code == 403
    detail = r.json()["detail"]
    assert "demo-ui" in detail and "svc:evaluation" in detail and "admin" in detail


def test_over_real_mtls_the_certificate_name_decides(running, tmp_path):
    """End to end on a socket: the verified certificate's CN is the identity."""
    ca_key = _key()
    ca = _cert("clasp-test-ca", ca_key, "clasp-test-ca", ca_key, ca=True)
    srv_key = _key()
    srv = _cert("registry", srv_key, "clasp-test-ca", ca_key, san="localhost")
    files = {"ca": _write(tmp_path / "ca2.pem", cert=ca),
             "server_cert": _write(tmp_path / "server2.pem", cert=srv),
             "server_key": _write(tmp_path / "server2.key", key=srv_key)}

    def context(name):
        key = _key()
        cert = _cert(name, key, "clasp-test-ca", ca_key)
        _write(tmp_path / f"{name}.pem", cert=cert)
        _write(tmp_path / f"{name}.key", key=key)
        ctx = ssl.create_default_context(cafile=str(files["ca"]))
        ctx.load_cert_chain(str(tmp_path / f"{name}.pem"), str(tmp_path / f"{name}.key"))
        return ctx

    port = running(_tls_env(files))
    base = f"https://localhost:{port}"
    assert httpx.get(f"{base}/adapters", verify=context("demo-ui"), timeout=5).status_code == 200
    assert httpx.get(f"{base}/adapters", verify=context("cluster"), timeout=5).status_code == 200
    refused = httpx.get(f"{base}/adapters", verify=context("client-flask"), timeout=5)
    assert refused.status_code == 403 and "client-flask" in refused.json()["detail"]
    assert httpx.get(f"{base}/healthz", verify=context("client-flask"), timeout=5).status_code == 200
    assert httpx.post(f"{base}/gc", json={}, verify=context("demo-ui"), timeout=5).status_code == 403
    assert httpx.post(f"{base}/gc", json={}, verify=context("clasp-admin"),
                      timeout=5).status_code != 403
