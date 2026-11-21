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
    sign_upload_body,
    upload_signing_parts,
    verify_adapter_upload,
    verify_signature,
    verify_upload_body,
)

# JWT tests require PyJWT
jwt_mod = pytest.importorskip("jwt", reason="PyJWT required for JWT tests")

from security.auth.jwt_handler import (  # noqa: E402 — after the importorskip guard
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
    LAYOUT = {"layer.weight": ("float32", (2,)), "layer.bias": ("uint8", (1,))}

    def test_round_trip(self) -> None:
        tensors = {"layer.weight": b"\x01\x02", "layer.bias": b"\x03"}
        meta = {"round": 1}
        sig = sign_adapter_upload(tensors, meta, "s", layouts=self.LAYOUT)
        assert verify_adapter_upload(tensors, meta, sig, "s", layouts=self.LAYOUT) is True

    def test_tampered_tensor_fails(self) -> None:
        layouts = {"w": ("uint8", (2,))}
        tensors = {"w": b"\x01\x02"}
        sig = sign_adapter_upload(tensors, {}, "s", layouts=layouts)
        tensors["w"] = b"\xff\xff"
        assert verify_adapter_upload(tensors, {}, sig, "s", layouts=layouts) is False

    def test_length_prefix_prevents_boundary_shift(self) -> None:
        """Signature must differ when bytes move from name to data."""
        layout = ("uint8", (2,))
        sig_a = sign_adapter_upload({"ab": b"cd"}, {}, "s", layouts={"ab": layout})
        sig_b = sign_adapter_upload({"a": b"bcd"}, {}, "s", layouts={"a": layout})
        assert sig_a != sig_b

    @pytest.mark.parametrize("layout", [("float32", (4, 2)), ("int32", (2, 4)), ("float32", (8,))])
    def test_the_signature_covers_dtype_and_shape(self, layout) -> None:
        """The same bytes read with another dtype or shape are another tensor."""
        tensors = {"w": bytes(32)}
        sig = sign_adapter_upload(tensors, {}, "s", layouts={"w": ("float32", (2, 4))})
        assert verify_adapter_upload(tensors, {}, sig, "s", layouts={"w": layout}) is False

    def test_every_tensor_needs_a_layout(self) -> None:
        with pytest.raises(ValueError, match="layout"):
            sign_adapter_upload({"w": b"\x00"}, {}, "s", layouts={})


def _upload_body() -> dict:
    import base64

    return {"client_id": "client-flask", "round_id": 0, "num_examples": 12,
            "tensors": [{"name": "layers.0.q_proj.lora_A.weight", "dtype": "float32",
                         "shape": [2, 4], "data_b64": base64.b64encode(bytes(32)).decode()}]}


class TestUploadBody:
    """The one recipe the edge signs with and the cluster verifies with."""

    def test_round_trip_survives_json(self) -> None:
        import json

        body = _upload_body()
        sig = sign_upload_body(body, "k")
        carried = json.loads(json.dumps({**body, "signature": sig, "timestamp": "t"}))
        assert verify_upload_body(carried, carried["signature"], "k") is True

    @pytest.mark.parametrize("field,value", [("shape", [4, 2]), ("dtype", "int32"),
                                             ("data_b64", "AA==")])
    def test_a_changed_tensor_entry_fails(self, field, value) -> None:
        body = _upload_body()
        sig = sign_upload_body(body, "k")
        body["tensors"][0][field] = value
        assert verify_upload_body(body, sig, "k") is False

    def test_a_changed_sample_count_fails(self) -> None:
        body = _upload_body()
        sig = sign_upload_body(body, "k")
        assert verify_upload_body({**body, "num_examples": 13}, sig, "k") is False

    @pytest.mark.parametrize("tensors", [None, [{"name": "w"}], [{"name": "w", "dtype": "f",
                                         "shape": ["x"], "data_b64": ""}], "nope"])
    def test_a_malformed_body_does_not_verify(self, tensors) -> None:
        body = {**_upload_body(), "tensors": tensors}
        assert verify_upload_body(body, "00" * 32, "k") is False

    def test_parts_leave_out_the_unsigned_keys(self) -> None:
        tensors, layouts, metadata = upload_signing_parts(
            {**_upload_body(), "signature": "x", "timestamp": "y"})
        assert set(metadata) == {"client_id", "round_id", "num_examples"}
        name = "layers.0.q_proj.lora_A.weight"
        assert tensors == {name: bytes(32)} and layouts == {name: ("float32", (2, 4))}


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
