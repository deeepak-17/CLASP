"""What the services need from the middleware and the audit logger (Task 8.6).

* the rate limiter is keyed by the caller's verified certificate name when
  there is one (else the remote address), skips exempt paths, and can be
  reconfigured in place;
* the audit logger can write to a file only, and be closed;
* ``AccessAuditMiddleware`` records every request and writes an audit line for
  each refused authorisation and each accepted mutating call.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from security.audit import AccessAuditMiddleware, AuditLogger
from security.middleware import (
    RateLimitConfig,
    RateLimiter,
    RateLimitMiddleware,
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)


def _name(request):
    return request.headers.get("x-test-cn")


def _app(limiter=None, **kw):
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/thing")
    def thing():
        return {"ok": True}

    if limiter is not None:
        app.add_middleware(RateLimitMiddleware, limiter=limiter, **kw)
    return TestClient(app)


# ---- rate limiter ------------------------------------------------------------------
def test_the_limiter_is_keyed_by_identity_when_there_is_one(monkeypatch):
    from security.middleware import rate_limiter

    # a burst: no time passes, so nothing is refilled between the requests
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 1000.0)
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=600, burst_size=60))
    api = _app(limiter, key_func=_name, exempt_paths=("/healthz",))
    codes = [api.get("/thing", headers={"x-test-cn": "cluster"}).status_code for _ in range(61)]
    assert codes[:60] == [200] * 60 and codes[60] == 429
    # a different identity, same address: its own bucket
    assert api.get("/thing", headers={"x-test-cn": "demo-ui"}).status_code == 200
    # no identity: the remote address is the key, also its own bucket
    assert api.get("/thing").status_code == 200


def test_a_429_says_when_to_come_back():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=1))
    api = _app(limiter, key_func=_name)
    assert api.get("/thing").status_code == 200
    refused = api.get("/thing")
    assert refused.status_code == 429 and int(refused.headers["retry-after"]) >= 1


def test_exempt_paths_are_never_limited_and_cost_nothing():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=2))
    api = _app(limiter, key_func=_name, exempt_paths=("/healthz",))
    assert all(api.get("/healthz").status_code == 200 for _ in range(50))
    assert api.get("/thing").status_code == 200 and api.get("/thing").status_code == 200
    assert api.get("/thing").status_code == 429
    assert api.get("/healthz").status_code == 200


def test_without_a_key_func_the_remote_address_is_the_key_as_before():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=1))
    api = _app(limiter)
    assert api.get("/thing", headers={"x-test-cn": "a"}).status_code == 200
    assert api.get("/thing", headers={"x-test-cn": "b"}).status_code == 429  # header ignored


def test_reconfigure_replaces_the_limits_and_forgets_the_buckets():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=1))
    assert limiter.check("a") and not limiter.check("a")
    limiter.reconfigure(RateLimitConfig(requests_per_minute=600, burst_size=60))
    assert all(limiter.check("a") for _ in range(60)) and not limiter.check("a")
    limiter.reconfigure(RateLimitConfig(enabled=False))
    assert all(limiter.check("a") for _ in range(500))
    api = _app(limiter, key_func=_name)
    assert all(api.get("/thing").status_code == 200 for _ in range(100))


# ---- audit logger -------------------------------------------------------------------
def test_an_audit_logger_can_write_to_a_file_only_and_be_closed(tmp_path, capsys):
    path = tmp_path / "deep" / "audit.jsonl"
    audit = AuditLogger(log_file=path, stream=False)
    audit.log_auth_failure("client-flask", "mtls-role", "POST /gc -> 403")
    audit.close()
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and lines[0]["event"] == "AUTH_FAILURE" and lines[0]["success"] is False
    assert capsys.readouterr().err == ""          # nothing on the console
    path.unlink()                                  # the file handle was released (Windows)


def test_two_file_loggers_do_not_write_into_each_others_files(tmp_path):
    one = AuditLogger(log_file=tmp_path / "one.jsonl", stream=False)
    two = AuditLogger(log_file=tmp_path / "two.jsonl", stream=False)
    one.log_auth_failure("a", "m", "r")
    two.log_auth_failure("b", "m", "r")
    one.close(), two.close()
    assert '"actor": "a"' in (tmp_path / "one.jsonl").read_text(encoding="utf-8")
    assert '"actor": "b"' in (tmp_path / "two.jsonl").read_text(encoding="utf-8")
    assert '"actor": "b"' not in (tmp_path / "one.jsonl").read_text(encoding="utf-8")


# ---- access audit middleware ---------------------------------------------------------
def _audited(tmp_path):
    app = FastAPI()
    rows = []
    audit = AuditLogger(log_file=tmp_path / "audit.jsonl", stream=False)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/adapters/{name}/versions/{version}")
    def read(name: str, version: int):
        return {"name": name}

    @app.post("/adapters/{name}/promote")
    def promote(name: str):
        return {"action": "promote"}

    @app.post("/gc")
    def gc():
        raise HTTPException(403, "admins only")

    @app.post("/bad")
    def bad():
        raise HTTPException(422, "no")

    app.add_middleware(
        AccessAuditMiddleware, audit=lambda: audit, identity=_name, service="registry",
        record=rows.append, exempt_paths=("/healthz",))
    return TestClient(app), rows, audit, tmp_path / "audit.jsonl"


def _lines(path, audit):
    audit.close()
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_every_request_but_health_is_recorded_with_the_callers_role(tmp_path):
    api, rows, audit, path = _audited(tmp_path)
    api.get("/healthz")
    api.get("/adapters/cluster-web/versions/2", headers={"x-test-cn": "demo-ui"})
    api.post("/adapters/cluster-web/promote", headers={"x-test-cn": "demo-ui"})
    api.get("/adapters/x/versions/1")                         # plain HTTP: the fall-back
    assert [(r["identity"], r["role"], r["method"], r["status"]) for r in rows] == [
        ("demo-ui", "svc:evaluation", "GET", 200),
        ("demo-ui", "svc:evaluation", "POST", 200),
        (None, "open", "GET", 200)]
    assert rows[0]["adapter"] == "cluster-web" and rows[0]["version"] == 2
    assert rows[1]["adapter"] == "cluster-web" and rows[1]["version"] is None
    assert rows[0]["path"] == "/adapters/cluster-web/versions/2" and rows[0]["at"]
    audit.close()


def test_an_accepted_mutating_call_and_a_refusal_each_write_one_audit_line(tmp_path):
    api, rows, audit, path = _audited(tmp_path)
    api.post("/adapters/cluster-web/promote", headers={"x-test-cn": "demo-ui"})
    api.post("/gc", headers={"x-test-cn": "client-flask"})
    api.get("/adapters/x/versions/1", headers={"x-test-cn": "demo-ui"})   # a read: no audit line
    api.post("/bad", headers={"x-test-cn": "demo-ui"})                    # a 422: not accepted
    lines = _lines(path, audit)
    assert [(x["event"], x["actor"], x["success"]) for x in lines] == [
        ("ADAPTER_VALIDATED", "demo-ui", True), ("AUTH_FAILURE", "client-flask", False)]
    assert lines[0]["metadata"] == {"service": "registry", "role": "svc:evaluation",
                                    "method": "POST", "path": "/adapters/cluster-web/promote",
                                    "status": 200}
    assert "POST /gc -> 403" in lines[1]["detail"]
    assert len(rows) == 4


def test_a_recorder_or_audit_failure_never_fails_the_request(tmp_path):
    app = FastAPI()

    @app.post("/x")
    def x():
        return {"ok": True}

    def boom(_row):
        raise RuntimeError("disk full")

    app.add_middleware(AccessAuditMiddleware, audit=lambda: None, identity=_name,
                       service="registry", record=boom)
    assert TestClient(app).post("/x").status_code == 200


def test_the_three_middlewares_stack_and_every_response_carries_the_headers():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=1))
    app = FastAPI()

    @app.post("/thing")
    def thing():
        return {"ok": True}

    # added innermost first: a request meets headers, then the size check, then the limiter
    app.add_middleware(RateLimitMiddleware, limiter=limiter, key_func=_name)
    app.add_middleware(RequestValidationMiddleware, max_body_bytes=10)
    app.add_middleware(SecurityHeadersMiddleware)
    api = TestClient(app)
    ok = api.post("/thing", content=b"0123456789")
    too_big = api.post("/thing", content=b"0123456789a")
    limited = api.post("/thing", content=b"x")
    assert (ok.status_code, too_big.status_code, limited.status_code) == (200, 413, 429)
    for response in (ok, too_big, limited):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"


# ---- pages with logins, and services without a data directory -------------------------
def test_a_page_can_say_who_the_person_is_and_a_login_is_a_token_issued_line(tmp_path):
    from security.audit import ServiceAudit

    app = FastAPI()
    rows = []
    trail = ServiceAudit("demo-ui", default_path=lambda: tmp_path / "panel.jsonl")

    @app.post("/api/login")
    def login():
        return {"token": "t"}

    @app.post("/api/live/register")
    def register():
        return {"registered": {}}

    app.add_middleware(
        AccessAuditMiddleware, audit=trail.logger, identity=lambda r: r.headers.get("x-user"),
        role=lambda r, name: "operator" if name else "open", service="demo-ui",
        record=rows.append)
    api = TestClient(app)
    api.post("/api/login", headers={"x-user": "operator"})
    api.post("/api/live/register", headers={"x-user": "operator"})
    trail.close()
    lines = [json.loads(x) for x in (tmp_path / "panel.jsonl").read_text("utf-8").splitlines()]
    assert [x["event"] for x in lines] == ["TOKEN_ISSUED", "ADAPTER_VALIDATED"]
    assert lines[1]["metadata"]["role"] == "operator" and rows[0]["role"] == "operator"


def test_service_audit_follows_the_env_variable_and_reopens_when_it_changes(tmp_path, monkeypatch):
    from security.audit import ServiceAudit

    trail = ServiceAudit("cluster")
    monkeypatch.setenv("CLASP_AUDIT_LOG", str(tmp_path / "a.jsonl"))
    trail.logger().log_auth_failure("x", "m", "first")
    monkeypatch.setenv("CLASP_AUDIT_LOG", str(tmp_path / "b.jsonl"))
    trail.logger().log_auth_failure("x", "m", "second")
    trail.close()
    assert "first" in (tmp_path / "a.jsonl").read_text("utf-8")
    assert "second" in (tmp_path / "b.jsonl").read_text("utf-8")
    assert "second" not in (tmp_path / "a.jsonl").read_text("utf-8")


def test_service_audit_with_no_file_configured_writes_to_the_console(monkeypatch, capsys):
    from security.audit import ServiceAudit

    monkeypatch.delenv("CLASP_AUDIT_LOG", raising=False)
    trail = ServiceAudit("cluster-console-test")
    trail.logger().log_auth_failure("x", "m", "to the console")
    trail.close()
    assert "to the console" in capsys.readouterr().err


def test_a_dropped_logger_never_lends_its_file_to_a_later_one(tmp_path):
    """Private loggers must not be named by object address: Python reuses
    addresses, and a new logger would inherit an unclosed one's file handle."""
    names = set()
    for i in range(200):
        audit = AuditLogger(log_file=tmp_path / f"{i % 2}.jsonl", stream=False)
        names.add(audit.logger.name)
        del audit                     # dropped without close(), as a crashed caller would
    assert len(names) == 200

    stale = AuditLogger(log_file=tmp_path / "old.jsonl", stream=False)
    del stale
    fresh = AuditLogger(log_file=tmp_path / "new.jsonl", stream=False)
    fresh.log_auth_failure("x", "m", "goes to the new file")
    fresh.close()
    assert "goes to the new file" in (tmp_path / "new.jsonl").read_text(encoding="utf-8")
    assert "goes to the new file" not in (tmp_path / "old.jsonl").read_text(encoding="utf-8")
