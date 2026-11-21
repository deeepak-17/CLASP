"""Security headers, a body limit and a rate limit in front of every registry route (U4)."""
from __future__ import annotations

import pytest
from security.middleware import RateLimitConfig, rate_limit_from_env

HEADER = "x-test-cn"
MIB = 1024 * 1024


@pytest.fixture
def api(client):
    import registry.app as appmod

    appmod.configure_identity(lambda request: request.headers.get(HEADER))
    yield client
    appmod.configure_identity(None)
    appmod.rate_limiter.reconfigure(RateLimitConfig(enabled=False))


@pytest.fixture
def limited(api, monkeypatch):
    """The real limits (600 a minute, bursts of 60) with the clock stopped."""
    import registry.app as appmod
    from security.middleware import rate_limiter

    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 1000.0)
    appmod.rate_limiter.reconfigure(rate_limit_from_env({}))
    return api


def test_every_response_carries_the_security_headers(api):
    for response in (api.get("/healthz"), api.get("/adapters"), api.get("/adapters/nope/active"),
                     api.post("/gc", json={}, headers={HEADER: "client-flask"})):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["content-security-policy"] == "default-src 'self'"
        assert "max-age" in response.headers["strict-transport-security"]
        assert response.headers["cache-control"] == "no-store"


def test_the_body_limit_is_the_upload_cap_plus_one_mebibyte():
    import registry.app as appmod

    assert appmod.MAX_BODY_BYTES == appmod.DEFAULT_MAX_UPLOAD + MIB == 1024 * MIB + MIB


def test_a_body_one_byte_over_the_limit_is_413(api):
    import registry.app as appmod

    over = api.post("/adapters/x/versions", content=b"x",
                    headers={"content-length": str(appmod.MAX_BODY_BYTES + 1)})
    assert over.status_code == 413
    at_limit = api.post("/gc", content=b"{}", headers={
        "content-length": str(appmod.MAX_BODY_BYTES), "content-type": "application/json"})
    assert at_limit.status_code != 413
    assert api.get("/adapters").json()["adapters"] == []


def test_the_default_rate_limit_is_600_a_minute_in_bursts_of_60():
    config = rate_limit_from_env({})
    assert (config.requests_per_minute, config.burst_size, config.enabled) == (600, 60, True)
    custom = rate_limit_from_env({"CLASP_RATE_LIMIT_PER_MIN": "1200"})
    assert (custom.requests_per_minute, custom.burst_size) == (1200, 60)
    assert rate_limit_from_env({"CLASP_RATE_LIMIT_PER_MIN": "0"}).enabled is False
    for bad in ("fast", "-5", "1.5"):
        with pytest.raises(ValueError, match="CLASP_RATE_LIMIT_PER_MIN"):
            rate_limit_from_env({"CLASP_RATE_LIMIT_PER_MIN": bad})


def test_request_61_in_a_burst_from_one_identity_is_429_and_another_identity_is_not(limited):
    codes = [limited.get("/adapters", headers={HEADER: "cluster"}).status_code for _ in range(61)]
    assert codes[:60] == [200] * 60 and codes[60] == 429
    assert limited.get("/adapters", headers={HEADER: "demo-ui"}).status_code == 200
    refused = limited.get("/adapters", headers={HEADER: "cluster"})
    assert refused.status_code == 429 and int(refused.headers["retry-after"]) >= 1
    assert refused.headers["x-content-type-options"] == "nosniff"   # headers on a 429 too


def test_healthz_is_never_rate_limited(limited):
    assert all(limited.get("/healthz", headers={HEADER: "cluster"}).status_code == 200
               for _ in range(200))
    # ... and did not use up the identity's own allowance
    assert limited.get("/adapters", headers={HEADER: "cluster"}).status_code == 200


def test_without_a_certificate_name_the_remote_address_is_limited(limited):
    codes = [limited.get("/adapters").status_code for _ in range(61)]
    assert codes[60] == 429
    assert limited.get("/adapters", headers={HEADER: "demo-ui"}).status_code == 200


def test_a_rate_limited_call_reaches_no_route(limited):
    for _ in range(60):
        limited.get("/adapters", headers={HEADER: "clasp-admin"})
    assert limited.post("/gc", json={}, headers={HEADER: "clasp-admin"}).status_code == 429
