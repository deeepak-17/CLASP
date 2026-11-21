from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SecretConfigError(ValueError):
    """A secret is configured but cannot be used (unreadable or empty)."""


def secret_from_env(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """A shared secret from ``<name>``, else from the file ``<name>_FILE`` names (C3).

    ``None`` when neither is set: the check that needs it is then skipped with
    a warning (the fall-back, C11). A variable that is set but empty, or a file
    that is missing or empty, is a :class:`SecretConfigError`: a secret that
    was meant to be there must not quietly turn the check off. The message
    names the variable and never the value.
    """
    env = os.environ if env is None else env
    file_var = f"{name}_FILE"
    if name in env:
        value = env[name].strip()
        if not value:
            raise SecretConfigError(f"{name} is set but empty")
        return value
    if env.get(file_var):
        try:
            value = Path(env[file_var]).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SecretConfigError(
                f"{file_var} names a file that cannot be read ({type(exc).__name__})") from exc
        if not value:
            raise SecretConfigError(f"{file_var} names an empty file")
        return value
    return None


def sign_payload(payload: bytes, secret: str) -> str:
    """
    Compute HMAC-SHA256 hex digest of a given payload.
    
    Args:
        payload: The raw bytes payload to sign.
        secret: The secret key used for signing.
        
    Returns:
        The hex-encoded HMAC-SHA256 signature string.
    """
    secret_bytes = secret.encode("utf-8")
    mac = hmac.new(secret_bytes, msg=payload, digestmod=hashlib.sha256)
    return mac.hexdigest()


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify HMAC signature using constant-time comparison.
    
    Args:
        payload: The raw bytes payload to verify.
        signature: The expected hex-encoded signature.
        secret: The secret key used for signing.
        
    Returns:
        True if the signature is valid, False otherwise.
    """
    expected_signature = sign_payload(payload, secret)
    return hmac.compare_digest(expected_signature, signature)


#: A tensor's layout: its dtype name and its shape. The same bytes read with
#: another layout are another tensor, so the layout is signed with the bytes.
Layout = tuple[str, tuple[int, ...]]

#: Upload body keys the signature does not cover: the tensors (signed entry by
#: entry, as bytes plus layout), the signature itself, and a timestamp a relay
#: may add. Shared by the edge that signs and the cluster that verifies.
UNSIGNED_UPLOAD_KEYS = ("tensors", "signature", "timestamp")


def _prefixed(data: bytes) -> bytes:
    return struct.pack("<I", len(data)) + data


def _layout_bytes(name: str, layout: Layout) -> bytes:
    dtype, shape = layout
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"tensor {name!r}: dtype must be a non-empty string")
    if not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape):
        raise ValueError(f"tensor {name!r}: shape must be non-negative integers")
    dims = b"".join(struct.pack("<Q", d) for d in shape)
    return _prefixed(dtype.encode("utf-8")) + struct.pack("<I", len(shape)) + dims


def _upload_message(tensors: Mapping[str, bytes], metadata: Mapping[str, Any],
                    layouts: Mapping[str, Layout]) -> bytes:
    """Length-prefixed canonical encoding, so no byte can move between fields.

    Per tensor, by name: ``name || dtype || ndim:4LE || dims:8LE each || data``
    (strings and data each prefixed with their 4-byte LE length); then the
    metadata as sorted-key JSON, also length-prefixed.
    """
    missing = sorted(set(tensors) - set(layouts))
    if missing:
        raise ValueError(f"no layout (dtype, shape) for tensor(s) {missing}")
    parts = [_prefixed(name.encode("utf-8")) + _layout_bytes(name, layouts[name])
             + _prefixed(tensors[name]) for name in sorted(tensors)]
    parts.append(_prefixed(json.dumps(metadata, sort_keys=True).encode("utf-8")))
    return b"".join(parts)


def sign_adapter_upload(tensors: Mapping[str, bytes], metadata: Mapping[str, Any], secret: str,
                        *, layouts: Mapping[str, Layout]) -> str:
    """HMAC-SHA256 over an adapter upload: every tensor's name, dtype, shape and
    raw bytes, and the rest of the body (``metadata``). Hex digest.

    ``layouts`` maps each tensor name to ``(dtype, shape)``; a tensor without
    one is a ``ValueError``.
    """
    return sign_payload(_upload_message(tensors, metadata, layouts), secret)


def verify_adapter_upload(tensors: Mapping[str, bytes], metadata: Mapping[str, Any],
                          signature: str, secret: str, *, layouts: Mapping[str, Layout]) -> bool:
    """Whether ``signature`` is :func:`sign_adapter_upload`'s over exactly these values."""
    expected_signature = sign_adapter_upload(tensors, metadata, secret, layouts=layouts)
    return hmac.compare_digest(expected_signature, signature)


def upload_signing_parts(
        body: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Layout], dict[str, Any]]:
    """``(tensors, layouts, metadata)`` of an upload body, as the signature reads it.

    ``body["tensors"]`` is a list of ``{"name", "dtype", "shape", "data_b64"}``;
    metadata is every other key except :data:`UNSIGNED_UPLOAD_KEYS`. A body
    that is not shaped like that raises ``ValueError`` (or ``KeyError`` /
    ``TypeError`` for a missing or mistyped field).
    """
    entries = body["tensors"]
    if not isinstance(entries, list):
        raise ValueError("tensors must be a list")
    tensors: dict[str, bytes] = {}
    layouts: dict[str, Layout] = {}
    for entry in entries:
        name = str(entry["name"])
        tensors[name] = base64.b64decode(entry["data_b64"], validate=True)
        layouts[name] = (entry["dtype"], tuple(entry["shape"]))
    metadata = {k: v for k, v in body.items() if k not in UNSIGNED_UPLOAD_KEYS}
    return tensors, layouts, metadata


def sign_upload_body(body: Mapping[str, Any], secret: str) -> str:
    """The edge's signature over an upload body (see :func:`upload_signing_parts`)."""
    tensors, layouts, metadata = upload_signing_parts(body)
    # through JSON once, so tuples and floats are what the receiver will read
    return sign_adapter_upload(tensors, json.loads(json.dumps(metadata)), secret, layouts=layouts)


def verify_upload_body(body: Mapping[str, Any], signature: Any, secret: str) -> bool:
    """Whether ``signature`` is the edge's over this body; ``False`` for a
    missing signature or a body that cannot be read as an upload."""
    if not isinstance(signature, str) or not signature:
        return False
    try:
        tensors, layouts, metadata = upload_signing_parts(body)
        return verify_adapter_upload(tensors, metadata, signature, secret, layouts=layouts)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return False


def evaluation_message(adapter: str, version: int, candidate_sha256: str,
                       eval_result: Mapping[str, Any], baseline_guard: Any) -> bytes:
    """The bytes an evaluation signature covers: which adapter version (by
    number and by the sha256 of its weights) was measured, and every number
    measured. Canonical JSON, so key order and transport do not matter."""
    return json.dumps(
        {"adapter": adapter, "version": version, "candidate_sha256": candidate_sha256,
         "eval": eval_result, "baseline_guard": baseline_guard},
        sort_keys=True, separators=(",", ":")).encode()


def sign_evaluation(*, adapter: str, version: int, candidate_sha256: str,
                    eval_result: Mapping[str, Any], baseline_guard: Any, secret: str) -> str:
    """Sign an evaluation result for the registry's ``/promote`` (U4)."""
    return sign_payload(
        evaluation_message(adapter, version, candidate_sha256, eval_result, baseline_guard), secret)


def verify_evaluation(adapter: str, version: int, candidate_sha256: str,
                      eval_result: Mapping[str, Any], baseline_guard: Any,
                      signature: Any, secret: str) -> bool:
    """Whether ``signature`` is the evaluator's over exactly these values."""
    if not isinstance(signature, str) or not signature:
        return False
    try:
        message = evaluation_message(adapter, version, candidate_sha256, eval_result, baseline_guard)
    except (TypeError, ValueError):
        return False
    return verify_signature(message, signature, secret)
