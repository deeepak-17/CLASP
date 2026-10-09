"""Tests for security.middleware — rate limiting, request validation, security headers."""
from __future__ import annotations

import pytest

# Middleware tests need fastapi/starlette
fastapi = pytest.importorskip("fastapi", reason="fastapi required for middleware tests")

from security.middleware.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    TokenBucket,
)
from security.middleware.request_validator import (
    ValidationError,
    sanitize_client_id,
    sanitize_path,
    validate_round_id,
    validate_tensor_payload,
)
from security.middleware.security_headers import SecurityHeadersConfig


# ── Token Bucket ─────────────────────────────────────────────────────────────


class TestTokenBucket:
    def test_consume_within_capacity(self) -> None:
        bucket = TokenBucket(capacity=5, fill_rate=1.0)
        assert bucket.consume(3) is True

    def test_consume_exhausts_tokens(self) -> None:
        bucket = TokenBucket(capacity=2, fill_rate=0.0)
        assert bucket.consume(1) is True
        assert bucket.consume(1) is True
        assert bucket.consume(1) is False


# ── Rate Limiter ─────────────────────────────────────────────────────────────


class TestRateLimiter:
    def test_allows_initial_burst(self) -> None:
        cfg = RateLimitConfig(burst_size=3, requests_per_minute=60)
        limiter = RateLimiter(cfg)
        for _ in range(3):
            assert limiter.check("c1") is True

    def test_rejects_after_burst(self) -> None:
        cfg = RateLimitConfig(burst_size=1, requests_per_minute=0)
        limiter = RateLimiter(cfg)
        assert limiter.check("c1") is True
        assert limiter.check("c1") is False

    def test_disabled_always_allows(self) -> None:
        cfg = RateLimitConfig(enabled=False, burst_size=0)
        limiter = RateLimiter(cfg)
        assert limiter.check("c1") is True

    def test_reset_restores_tokens(self) -> None:
        cfg = RateLimitConfig(burst_size=1, requests_per_minute=0)
        limiter = RateLimiter(cfg)
        limiter.check("c1")
        limiter.check("c1")
        limiter.reset("c1")
        assert limiter.check("c1") is True

    def test_get_stats(self) -> None:
        cfg = RateLimitConfig(burst_size=5, requests_per_minute=60)
        limiter = RateLimiter(cfg)
        stats = limiter.get_stats("c1")
        assert "remaining" in stats
        assert "next_refill_seconds" in stats

    def test_stale_bucket_eviction(self) -> None:
        """Buckets older than TTL should be evicted."""
        cfg = RateLimitConfig(burst_size=5, bucket_ttl_seconds=0.0)
        limiter = RateLimiter(cfg)
        limiter.check("old-client")
        # Next call for a different client triggers eviction
        limiter.check("new-client")
        with limiter.lock:
            assert "old-client" not in limiter.buckets


# ── Input Validation ─────────────────────────────────────────────────────────


class TestSanitizeClientId:
    def test_valid_id(self) -> None:
        assert sanitize_client_id("edge-0") == "edge-0"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            sanitize_client_id("")

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            sanitize_client_id("a" * 65)

    def test_special_chars_raises(self) -> None:
        with pytest.raises(ValidationError):
            sanitize_client_id("edge;drop table")


class TestSanitizePath:
    def test_valid_path(self) -> None:
        assert sanitize_path("adapters/lora.bin") == "adapters/lora.bin"

    def test_traversal_raises(self) -> None:
        with pytest.raises(ValidationError):
            sanitize_path("../../etc/passwd")

    def test_absolute_raises(self) -> None:
        with pytest.raises(ValidationError):
            sanitize_path("/etc/passwd")


class TestValidateTensorPayload:
    def test_valid_payload(self) -> None:
        tensors = [{"name": "w", "dtype": "f32", "shape": [2, 3], "data_b64": "AAAA"}]
        validate_tensor_payload(tensors)  # should not raise

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_tensor_payload([{"name": "w"}])

    def test_too_many_tensors_raises(self) -> None:
        tensors = [{"name": "w", "dtype": "f", "shape": [], "data_b64": "A"}] * 1001
        with pytest.raises(ValidationError):
            validate_tensor_payload(tensors)

    def test_not_a_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_tensor_payload("not a list")


class TestValidateRoundId:
    def test_valid(self) -> None:
        validate_round_id(0)
        validate_round_id(42)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_round_id(-1)

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_round_id("one")


# ── Security Headers Config ──────────────────────────────────────────────────


class TestSecurityHeadersConfig:
    def test_defaults(self) -> None:
        cfg = SecurityHeadersConfig()
        assert cfg.x_content_type_options == "nosniff"
        assert cfg.x_frame_options == "DENY"
