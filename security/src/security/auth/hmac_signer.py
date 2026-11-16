from __future__ import annotations

import hashlib
import hmac
import json
import os
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


def sign_adapter_upload(tensors: dict[str, bytes], metadata: dict[str, Any], secret: str) -> str:
    """
    Sign adapter upload using a length-prefixed canonical encoding.

    Each tensor entry is encoded as ``len(name):4LE || name || len(data):4LE || data``
    to prevent bytes from shifting between the name and data fields without
    invalidating the signature.

    Args:
        tensors: Dictionary mapping tensor names to their byte data.
        metadata: Dictionary of metadata associated with the upload.
        secret: The secret key used for signing.

    Returns:
        The hex-encoded HMAC-SHA256 signature string.
    """
    import struct

    payload_parts: list[bytes] = []
    # Sort tensor names to ensure consistent ordering
    for name in sorted(tensors.keys()):
        name_bytes = name.encode("utf-8")
        data = tensors[name]
        # Length-prefix each component so name/data boundaries are unambiguous
        payload_parts.append(struct.pack("<I", len(name_bytes)))
        payload_parts.append(name_bytes)
        payload_parts.append(struct.pack("<I", len(data)))
        payload_parts.append(data)

    # Append serialized metadata (sorted keys for deterministic JSON)
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    payload_parts.append(struct.pack("<I", len(metadata_bytes)))
    payload_parts.append(metadata_bytes)

    full_payload = b"".join(payload_parts)
    return sign_payload(full_payload, secret)


def verify_adapter_upload(tensors: dict[str, bytes], metadata: dict[str, Any], signature: str, secret: str) -> bool:
    """
    Verify an adapter upload signature.
    
    Args:
        tensors: Dictionary mapping tensor names to their byte data.
        metadata: Dictionary of metadata associated with the upload.
        signature: The provided signature to verify.
        secret: The secret key used for signing.
        
    Returns:
        True if the signature is valid, False otherwise.
    """
    expected_signature = sign_adapter_upload(tensors, metadata, secret)
    return hmac.compare_digest(expected_signature, signature)


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
