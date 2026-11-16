"""security.auth.roles: a service's role is read off its verified certificate name."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from security.auth.roles import (
    DEFAULT_ROLES,
    ROLE_ADMIN,
    ROLE_CLUSTER,
    ROLE_EDGE,
    ROLE_EVALUATION,
    ROLE_OPEN,
    ROLE_TABLE_ENV,
    ROLE_UNKNOWN,
    load_role_table,
    require_roles,
    role_of,
)
from security.mtls.peer import SCOPE_KEY


def test_the_role_names_and_the_default_table():
    assert (ROLE_CLUSTER, ROLE_EVALUATION, ROLE_ADMIN, ROLE_EDGE, ROLE_OPEN) == (
        "svc:cluster", "svc:evaluation", "admin", "edge", "open")
    assert DEFAULT_ROLES == {"cluster": ROLE_CLUSTER, "demo-ui": ROLE_EVALUATION,
                             "clasp-admin": ROLE_ADMIN}


@pytest.mark.parametrize("name, role", [
    (None, ROLE_OPEN),                    # no TLS: the fall-back
    ("cluster", ROLE_CLUSTER),
    ("demo-ui", ROLE_EVALUATION),
    ("clasp-admin", ROLE_ADMIN),
    ("client-flask", ROLE_EDGE),          # every edge certificate is named for its client
    ("client-scikit-learn", ROLE_EDGE),
    ("clasp-dev-client", ROLE_UNKNOWN),   # a CA-signed certificate with no role
    ("registry", ROLE_UNKNOWN),
    ("", ROLE_UNKNOWN),                   # a certificate with an empty name is not "no TLS"
    ("client-", ROLE_UNKNOWN),
])
def test_role_of(name, role):
    assert role_of(name) == role


def test_a_table_entry_wins_over_the_client_prefix():
    assert role_of("client-ops", table={"client-ops": ROLE_ADMIN}) == ROLE_ADMIN
    assert role_of("cluster", table={}) == ROLE_UNKNOWN


def test_the_table_can_come_from_a_json_file(tmp_path, monkeypatch):
    path = tmp_path / "roles.json"
    path.write_text(json.dumps({"panel-2": "svc:evaluation"}), encoding="utf-8")
    assert load_role_table({ROLE_TABLE_ENV: str(path)}) == {"panel-2": "svc:evaluation"}
    assert load_role_table({}) == DEFAULT_ROLES
    monkeypatch.setenv(ROLE_TABLE_ENV, str(path))
    assert load_role_table() == {"panel-2": "svc:evaluation"}


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"x": 3}', '{"x": "root"}'])
def test_a_bad_table_file_is_an_error_not_an_open_door(tmp_path, content):
    path = tmp_path / "roles.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="roles.json"):
        load_role_table({ROLE_TABLE_ENV: str(path)})
    with pytest.raises(ValueError):
        load_role_table({ROLE_TABLE_ENV: str(tmp_path / "missing.json")})


def _app(identity=None):
    app = FastAPI()
    kwargs = {"identity": identity} if identity is not None else {}

    @app.get("/ops", dependencies=[Depends(require_roles(ROLE_EVALUATION, ROLE_ADMIN, **kwargs))])
    def ops():
        return {"ok": True}

    @app.get("/who")
    def who(role: str = Depends(require_roles(ROLE_EVALUATION, ROLE_ADMIN, ROLE_CLUSTER, **kwargs))):
        return {"role": role}

    return TestClient(app)


def _header(request):
    return request.headers.get("x-test-cn")


def test_the_dependency_admits_allowed_roles_and_refuses_the_rest():
    api = _app(_header)
    assert api.get("/ops", headers={"x-test-cn": "demo-ui"}).status_code == 200
    assert api.get("/ops", headers={"x-test-cn": "clasp-admin"}).status_code == 200
    for name in ("cluster", "client-flask", "clasp-dev-client"):
        r = api.get("/ops", headers={"x-test-cn": name})
        assert r.status_code == 403
        # says who was refused and which roles may call, never anything secret
        assert name in r.json()["detail"] and "svc:evaluation" in r.json()["detail"]
    assert api.get("/who", headers={"x-test-cn": "cluster"}).json() == {"role": "svc:cluster"}


def test_without_an_identity_the_route_is_open():
    """The fall-back (U6): no TLS, so no certificate name, so nothing to check."""
    api = _app(_header)
    assert api.get("/ops").status_code == 200
    assert api.get("/who").json() == {"role": "open"}


def test_by_default_the_identity_is_the_verified_peer_name_in_the_scope():
    dependency = require_roles(ROLE_ADMIN)

    def request(name):
        return SimpleNamespace(scope={SCOPE_KEY: name} if name is not None else {}, headers={})

    assert dependency(request("clasp-admin")) == ROLE_ADMIN
    assert dependency(request(None)) == ROLE_OPEN
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refused:
        dependency(request("client-flask"))
    assert refused.value.status_code == 403


def test_the_dependency_reads_the_role_table_env(tmp_path, monkeypatch):
    path = tmp_path / "roles.json"
    path.write_text(json.dumps({"night-shift": "admin"}), encoding="utf-8")
    monkeypatch.setenv(ROLE_TABLE_ENV, str(path))
    api = _app(_header)
    assert api.get("/ops", headers={"x-test-cn": "night-shift"}).status_code == 200
    assert api.get("/ops", headers={"x-test-cn": "demo-ui"}).status_code == 403  # table replaced


def test_require_roles_needs_at_least_one_role():
    with pytest.raises(ValueError):
        require_roles()
    with pytest.raises(ValueError):
        require_roles("superuser")
