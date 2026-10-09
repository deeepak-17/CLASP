"""Tests for security.auth — API keys, HMAC signing, and JWT tokens."""
from __future__ import annotations

import time

import pytest

from security.auth.api_key import (
    APIKeyStore,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from security.auth.hmac_signer import (
    sign_adapter_upload,
    sign_payload,
    verify_adapter_upload,
    verify_signature,
)

# JWT tests require PyJWT
jwt_mod = pytest.importorskip("jwt", reason="PyJWT required for JWT tests")

from security.auth.jwt_handler import (  # noqa: E402
    JWTConfig,
    TokenError,
    create_token,
    verify_token,
)


# ── API Key ──────────────────────────────────────────────────────────────────


class TestGenerateAPIKey:
    def test_has_prefix(self) -> None:
        key = generate_api_key(prefix="test")
        assert key.startswith("test_")

    def test_default_prefix(self) -> None:
        key = generate_api_key()
        assert key.startswith("clasp_")

    def test_length(self) -> None:
        key = generate_api_key()
        # prefix "clasp" + "_" + 32 random chars
        assert len(key) == len("clasp_") + 32

    def test_uniqueness(self) -> None:
        keys = {generate_api_key() for _ in range(50)}
        assert len(keys) == 50


class TestHashAndVerifyAPIKey:
    def test_hash_is_hex(self) -> None:
        h = hash_api_key("test-key")
        assert len(h) == 64  # SHA-256 hex
        int(h, 16)  # should not raise

    def test_verify_correct_key(self) -> None:
        key = "my-secret-key"
        h = hash_api_key(key)
        assert verify_api_key(key, h) is True

    def test_verify_wrong_key(self) -> None:
        h = hash_api_key("correct")
        assert verify_api_key("wrong", h) is False


class TestAPIKeyStore:
    def test_register_and_authenticate(self) -> None:
        store = APIKeyStore()
        raw = store.register("svc-a")
        assert store.authenticate("svc-a", raw) is True

    def test_wrong_key_fails(self) -> None:
        store = APIKeyStore()
        store.register("svc-a")
        assert store.authenticate("svc-a", "wrong") is False

    def test_unknown_service_fails(self) -> None:
        store = APIKeyStore()
        assert store.authenticate("no-such-svc", "key") is False

    def test_revoke(self) -> None:
        store = APIKeyStore()
        raw = store.register("svc-a")
        store.revoke("svc-a")
        assert store.authenticate("svc-a", raw) is False

    def test_list_services(self) -> None:
        store = APIKeyStore()
        store.register("a")
        store.register("b")
        assert set(store.list_services()) == {"a", "b"}


# ── HMAC Signing ─────────────────────────────────────────────────────────────


class TestSignPayload:
    def test_sign_returns_hex(self) -> None:
        sig = sign_payload(b"hello", "secret")
        assert len(sig) == 64
        int(sig, 16)

    def test_verify_valid(self) -> None:
        sig = sign_payload(b"data", "key")
        assert verify_signature(b"data", sig, "key") is True

    def test_verify_tampered(self) -> None:
        sig = sign_payload(b"data", "key")
        assert verify_signature(b"tampered", sig, "key") is False

    def test_verify_wrong_secret(self) -> None:
        sig = sign_payload(b"data", "key")
        assert verify_signature(b"data", sig, "wrong-key") is False


class TestSignAdapterUpload:
    def test_round_trip(self) -> None:
        tensors = {"layer.weight": b"\x01\x02", "layer.bias": b"\x03"}
        meta = {"round": 1}
        sig = sign_adapter_upload(tensors, meta, "s")
        assert verify_adapter_upload(tensors, meta, sig, "s") is True

    def test_tampered_tensor_fails(self) -> None:
        tensors = {"w": b"\x01\x02"}
        sig = sign_adapter_upload(tensors, {}, "s")
        tensors["w"] = b"\xff\xff"
        assert verify_adapter_upload(tensors, {}, sig, "s") is False

    def test_length_prefix_prevents_boundary_shift(self) -> None:
        """Signature must differ when bytes move from name to data."""
        sig_a = sign_adapter_upload({"ab": b"cd"}, {}, "s")
        sig_b = sign_adapter_upload({"a": b"bcd"}, {}, "s")
        assert sig_a != sig_b


# ── JWT ──────────────────────────────────────────────────────────────────────


class TestJWT:
    CFG = JWTConfig(secret_key="test-secret-key-1234", token_expiry_minutes=60)

    def test_create_and_verify(self) -> None:
        token = create_token("edge-0", self.CFG)
        payload = verify_token(token, self.CFG)
        assert payload["sub"] == "edge-0"
        assert payload["iss"] == "clasp-security"

    def test_custom_claims(self) -> None:
        token = create_token("edge-0", self.CFG, claims={"role": "client"})
        payload = verify_token(token, self.CFG)
        assert payload["role"] == "client"

    def test_wrong_secret_raises(self) -> None:
        token = create_token("edge-0", self.CFG)
        bad_cfg = JWTConfig(secret_key="wrong")
        with pytest.raises(TokenError):
            verify_token(token, bad_cfg)

    def test_expired_token_raises(self) -> None:
        cfg = JWTConfig(secret_key="test", token_expiry_minutes=0)
        token = create_token("edge-0", cfg)
        time.sleep(1)  # let it expire
        with pytest.raises(TokenError, match="expired"):
            verify_token(token, cfg)
