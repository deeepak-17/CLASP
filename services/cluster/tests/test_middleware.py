"""Security headers, a body limit, a rate limit and an audit trail on the cluster (U4)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from security.middleware import RateLimitConfig, rate_limit_from_env

from cluster import server
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    dp,
    reset_server_state,
    restore_server_state,
    upload_body,
)

client = TestClient(server.app)
HEADER = "x-test-client-cn"
MIB = 1024 * 1024


def _provider(request):
    return request.headers.get(HEADER)


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    saved = reset_server_state()
    monkeypatch.setenv("CLASP_AUDIT_LOG", str(tmp_path / "cluster-audit.jsonl"))
    create_cluster(client, "web", members={"web": ["client-flask", "client-requests"]})
    server.configure_identity(_provider)
    yield tmp_path / "cluster-audit.jsonl"
    server.audit_trail.close()
    server.rate_limiter.reconfigure(RateLimitConfig(enabled=False))
    restore_server_state(saved)


@pytest.fixture
def limited(monkeypatch):
    from security.middleware import rate_limiter

    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 1000.0)
    server.rate_limiter.reconfigure(rate_limit_from_env({}))


def _lines(path) -> list[dict]:
    server.audit_trail.close()
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] \
        if path.exists() else []


def _upload(who="client-flask", seed=1, **extra):
    return client.post("/clusters/web/uploads", json=upload_body(who, trained_adapter(seed), **extra),
                       headers={HEADER: who})


def test_every_response_carries_the_security_headers():
    for response in (client.get("/healthz"), client.get("/clusters"),
                     client.post("/clusters/web/aggregate", json={}, headers={HEADER: "cluster"}),
                     client.get("/clusters/nope/adapters/active")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["content-security-policy"] == "default-src 'self'"


def test_the_body_limit_is_512_mebibytes_and_one_byte_more_is_413():
    assert server.MAX_BODY_BYTES == 512 * MIB
    over = client.post("/clusters/web/uploads", content=b"{}", headers={
        "content-length": str(server.MAX_BODY_BYTES + 1), "content-type": "application/json"})
    assert over.status_code == 413
    at_limit = client.post("/clusters/web/uploads", content=b"{}", headers={
        "content-length": str(server.MAX_BODY_BYTES), "content-type": "application/json"})
    assert at_limit.status_code != 413
    assert server._clusters["web"].uploads == {}


def test_a_real_size_upload_fits_the_body_limit_many_times_over():
    """192 fp32 tensors of a rank-16 adapter on a 2048-wide model, base64 in JSON."""
    raw = 192 * 16 * 2048 * 4
    as_json = raw * 4 // 3 + 192 * 200          # base64, plus each tensor's name/shape/dtype
    assert as_json < server.MAX_BODY_BYTES / 10


def test_request_61_in_a_burst_is_429_and_another_identity_is_not(limited):
    codes = [client.get("/clusters", headers={HEADER: "client-flask"}).status_code
             for _ in range(61)]
    assert codes[:60] == [200] * 60 and codes[60] == 429
    assert client.get("/clusters", headers={HEADER: "client-requests"}).status_code == 200
    assert client.get("/clusters", headers={HEADER: "client-flask"}).headers["retry-after"]


def test_healthz_is_never_rate_limited(limited):
    assert all(client.get("/healthz", headers={HEADER: "client-flask"}).status_code == 200
               for _ in range(200))
    assert client.get("/clusters", headers={HEADER: "client-flask"}).status_code == 200


def test_an_accepted_upload_and_a_merge_are_audit_lines(_clean):
    assert _upload("client-flask", 1).status_code == 201
    assert _upload("client-requests", 2).status_code == 201
    assert client.post("/clusters/web/aggregate", json={},
                       headers={HEADER: "demo-ui"}).status_code == 200
    lines = _lines(_clean)
    assert [(x["event"], x["actor"]) for x in lines if x["resource"] == "adapter"] == [
        ("ADAPTER_VALIDATED", "anonymous"),          # the PUT .../members in the fixture
        ("ADAPTER_UPLOADED", "client-flask"), ("ADAPTER_UPLOADED", "client-requests"),
        ("AGGREGATION_COMPLETED", "demo-ui")]
    merged = lines[-1]["metadata"]
    assert merged == {"service": "cluster", "role": "svc:evaluation", "method": "POST",
                      "path": "/clusters/web/aggregate", "status": 200}


def test_a_refused_authorisation_is_an_audit_line(_clean):
    assert client.post("/clusters/web/aggregate", json={},
                       headers={HEADER: "client-flask"}).status_code == 403
    spoof = client.post("/clusters/web/uploads",
                        json=upload_body("client-requests", trained_adapter(1)),
                        headers={HEADER: "client-flask"})
    assert spoof.status_code == 403
    failures = [x for x in _lines(_clean) if x["event"] == "AUTH_FAILURE"]
    assert len(failures) == 2 and all(x["actor"] == "client-flask" for x in failures)
    assert "POST /clusters/web/aggregate -> 403" in failures[0]["detail"]


def test_a_budget_refusal_is_a_privacy_audit_line(_clean):
    assert _upload(privacy=dp(4.5, cumulative=8.4, rounds=2)).status_code == 422
    assert _upload(privacy={"epsilon": 7.9}).status_code == 422
    privacy = [x for x in _lines(_clean) if x["resource"] == "privacy_budget"]
    assert [(x["event"], x["actor"], x["success"]) for x in privacy] == [
        ("PRIVACY_BUDGET_EXHAUSTED", "client-flask", False),
        ("PRIVACY_BUDGET_EXHAUSTED", "client-flask", False)]
    assert privacy[0]["metadata"] == {"error": "privacy_budget_exceeded", "epsilon_cumulative": 8.4,
                                      "rounds_accounted": 2, "budget": 8.0}
    assert privacy[1]["metadata"]["error"] == "privacy_budget_unreported"


def test_a_signature_refusal_is_on_the_record_without_the_key_or_signature(_clean, tmp_path,
                                                                           monkeypatch):
    key = "a1" * 32
    secrets = tmp_path / "upload_hmac.json"
    secrets.write_text(json.dumps({"client-flask": key}), encoding="utf-8")
    monkeypatch.setenv("CLASP_UPLOAD_HMAC_SECRETS", str(secrets))
    body = upload_body("client-flask", trained_adapter(1))
    body["signature"] = "de" * 32
    r = client.post("/clusters/web/uploads", json=body, headers={HEADER: "client-flask"})
    assert r.status_code == 401
    server.audit_trail.close()
    text = _clean.read_text(encoding="utf-8")
    assert "AUTH_FAILURE" in text and "-> 401" in text
    assert key not in text and "de" * 32 not in text


def test_reads_and_health_are_not_audit_lines(_clean):
    before = len(_lines(_clean))
    client.get("/healthz")
    client.get("/clusters", headers={HEADER: "client-flask"})
    assert len(_lines(_clean)) == before
