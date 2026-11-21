"""The registry's audit trail: an access log table and an audit file (U4).

Every request but ``/healthz`` is a row in ``access_log`` (who, with which
role, what, the answer). Every refused authorisation, every accepted mutating
call and every recorded privacy row is a line in the audit file
(``CLASP_AUDIT_LOG``, default ``<data root>/audit.jsonl``). Neither ever holds
a secret, a token or a signature.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from security.auth.hmac_signer import sign_evaluation
from sqlalchemy import inspect

from registry import db

from .test_composite import make_adapter
from .test_composite_api import _save
from .test_promote_api import _promote_body

HEADER = "x-test-cn"
NAME = "cluster-web"
SECRET = "5e" * 32


@pytest.fixture
def api(client):
    import registry.app as appmod

    appmod.configure_identity(lambda request: request.headers.get(HEADER))
    yield client
    appmod.configure_identity(None)


def _as(name):
    return {HEADER: name}


def _audit_lines(tmp_path) -> list[dict]:
    from registry import audit

    audit.close_audit()  # flush
    path = tmp_path / "registry_api" / "audit.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] \
        if path.exists() else []


def _access(api, **params):
    r = api.get("/access-log", params=params, headers=_as("clasp-admin"))
    assert r.status_code == 200, r.text
    return r.json()["rows"]


def _candidate(api):
    _save(api, NAME, make_adapter(1), kind="cluster", cluster_id="web")
    return _save(api, NAME, make_adapter(2), kind="cluster", cluster_id="web",
                 set_active=False)["sha256"]


def _body(sha, secret=None):
    body = _promote_body(NAME, 2, good=True)
    body["eval"]["adapter"].update(kind="cluster", cluster_id="web")
    body["candidate_sha256"] = sha
    if secret:
        body["eval_signature"] = sign_evaluation(
            adapter=NAME, version=2, candidate_sha256=sha, eval_result=body["eval"],
            baseline_guard=body["baseline_guard"], secret=secret)
    return body


# ---- schema ------------------------------------------------------------------------
def test_the_access_log_table_and_its_index(tmp_path):
    engine = db.make_engine(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    db.create_all(engine)
    insp = inspect(engine)
    assert {c["name"] for c in insp.get_columns("access_log")} == {
        "id", "at", "identity", "role", "method", "path", "adapter", "version", "status"}
    assert any(i["column_names"] == ["identity", "at"] for i in insp.get_indexes("access_log"))


def test_the_repository_adds_and_lists_rows(store):
    for i, who in enumerate(("cluster", "demo-ui", "cluster")):
        store.repo.add_access({"at": f"2026-11-16T00:00:0{i}+00:00", "identity": who,
                               "role": "svc:x", "method": "GET", "path": "/adapters",
                               "adapter": None, "version": None, "status": 200})
    rows = store.repo.access_log()
    assert [r["identity"] for r in rows] == ["cluster", "demo-ui", "cluster"]  # newest first
    assert [r["at"][-7] for r in rows] == ["2", "1", "0"]
    assert len(store.repo.access_log(limit=2)) == 2
    assert len(store.repo.access_log(identity="demo-ui")) == 1


# ---- what gets written ----------------------------------------------------------------
def test_a_promote_writes_one_audit_line_and_one_access_row_with_the_callers_role(api, tmp_path):
    sha = _candidate(api)
    before_lines = len(_audit_lines(tmp_path))
    r = api.post(f"/adapters/{NAME}/promote", json=_body(sha), headers=_as("demo-ui"))
    assert r.status_code == 200
    lines = _audit_lines(tmp_path)[before_lines:]
    assert len(lines) == 1
    line = lines[0]
    assert line["event"] == "ADAPTER_VALIDATED" and line["actor"] == "demo-ui" and line["success"]
    assert line["metadata"] == {"service": "registry", "role": "svc:evaluation", "method": "POST",
                                "path": f"/adapters/{NAME}/promote", "status": 200}
    rows = [x for x in _access(api) if x["path"].endswith("/promote")]
    assert len(rows) == 1
    assert (rows[0]["identity"], rows[0]["role"], rows[0]["method"], rows[0]["status"]) == (
        "demo-ui", "svc:evaluation", "POST", 200)
    assert rows[0]["adapter"] == NAME and rows[0]["at"]


def test_a_refused_call_is_an_auth_failure_line_and_a_403_row(api, tmp_path):
    r = api.post("/gc", json={}, headers=_as("client-flask"))
    assert r.status_code == 403
    lines = [x for x in _audit_lines(tmp_path) if x["event"] == "AUTH_FAILURE"]
    assert len(lines) == 1 and lines[0]["actor"] == "client-flask" and lines[0]["success"] is False
    assert "POST /gc -> 403" in lines[0]["detail"] and "edge" in lines[0]["detail"]
    rows = _access(api, identity="client-flask")
    assert [(x["path"], x["role"], x["status"]) for x in rows] == [("/gc", "edge", 403)]


def test_reads_are_rows_but_not_audit_lines_and_healthz_is_neither(api, tmp_path):
    api.get("/healthz", headers=_as("cluster"))
    api.get("/adapters", headers=_as("cluster"))
    assert _audit_lines(tmp_path) == []
    rows = _access(api, identity="cluster")
    assert [x["path"] for x in rows] == ["/adapters"]


def test_without_a_certificate_name_the_row_says_open(client, tmp_path):
    """The fall-back: allowed, and on the record as unauthenticated."""
    assert client.get("/adapters").status_code == 200
    import registry.app as appmod

    rows = appmod.get_store().repo.access_log()
    assert (rows[0]["identity"], rows[0]["role"], rows[0]["path"]) == (None, "open", "/adapters")


def test_recorded_privacy_rows_are_audit_lines(api, tmp_path):
    _save(api, NAME, make_adapter(1), kind="cluster", cluster_id="web", set_active=False,
          client_privacy=[
              {"client_id": "client-flask", "epsilon_round": 3.9, "epsilon_cumulative": 7.8,
               "rounds_accounted": 2},
              {"client_id": "client-plain", "epsilon_round": None, "epsilon_cumulative": None,
               "rounds_accounted": None}])
    lines = [x for x in _audit_lines(tmp_path) if x["resource"] == "privacy_budget"]
    assert len(lines) == 1
    assert lines[0]["event"] == "PRIVACY_BUDGET_RECORDED" and lines[0]["actor"] == "client-flask"
    assert lines[0]["metadata"] == {"epsilon_spent": 7.8, "epsilon_target": 8.0}


# ---- who may read it ---------------------------------------------------------------------
def test_the_access_log_is_for_admins(api):
    for name in ("cluster", "demo-ui", "client-flask", "clasp-dev-client"):
        assert api.get("/access-log", headers=_as(name)).status_code == 403, name
    assert api.get("/access-log", headers=_as("clasp-admin")).status_code == 200


def test_limit_and_identity_filter_and_bad_limits(api):
    for _ in range(5):
        api.get("/adapters", headers=_as("cluster"))
    api.get("/adapters", headers=_as("demo-ui"))
    assert len(_access(api, limit=3)) == 3
    assert {x["identity"] for x in _access(api, identity="demo-ui")} == {"demo-ui"}
    assert len(_access(api)) <= 200
    for bad in (0, -1, 100000):
        assert api.get("/access-log", params={"limit": bad},
                       headers=_as("clasp-admin")).status_code == 422


# ---- what never gets written -----------------------------------------------------------------
def test_no_audit_line_or_access_row_contains_a_secret_token_or_signature(api, tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET", SECRET)
    sha = _candidate(api)
    body = _body(sha, secret=SECRET)
    token = "eyJhbGciOiJIUzI1NiJ9.e30.not-a-real-token"
    headers = {**_as("demo-ui"), "Authorization": f"Bearer {token}"}
    forged = {**body, "eval_signature": "ab" * 32}
    assert api.post(f"/adapters/{NAME}/promote", json=forged, headers=headers).status_code == 401
    assert api.post(f"/adapters/{NAME}/promote", json=body, headers=headers).status_code == 200
    audit_text = (tmp_path / "registry_api" / "audit.jsonl")
    from registry import audit

    audit.close_audit()
    text = audit_text.read_text(encoding="utf-8") + json.dumps(_access(api))
    for secret in (SECRET, token, body["eval_signature"], "ab" * 32):
        assert secret not in text
    assert "AUTH_FAILURE" in text  # the 401 itself is on the record


def test_the_audit_file_can_be_placed_with_an_env_variable(api, tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "registry-audit.jsonl"
    monkeypatch.setenv("CLASP_AUDIT_LOG", str(target))
    api.post("/gc", json={}, headers=_as("client-flask"))
    from registry import audit

    audit.close_audit()
    assert "AUTH_FAILURE" in Path(target).read_text(encoding="utf-8")


def test_the_access_log_keeps_only_its_newest_rows(tmp_path):
    """Every request is a row; a panel polls several times a minute, so the
    table is capped: past the cap the oldest rows go."""
    from registry import db
    from registry.repository import SqlMetadataRepository

    engine = db.make_engine(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    db.create_all(engine)
    repo = SqlMetadataRepository(engine, access_log_max_rows=5, access_log_prune_every=1)
    for i in range(12):
        repo.add_access({"at": f"t{i:02d}", "identity": None, "role": "open", "method": "GET",
                         "path": f"/p{i}", "status": 200})
    assert [r["path"] for r in repo.access_log()] == [f"/p{i}" for i in range(11, 6, -1)]
    engine.dispose()


def test_the_default_cap_is_generous():
    from registry.repository import ACCESS_LOG_MAX_ROWS, ACCESS_LOG_PRUNE_EVERY

    assert ACCESS_LOG_MAX_ROWS >= 100_000 and 1 <= ACCESS_LOG_PRUNE_EVERY <= 10_000
