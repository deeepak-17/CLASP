from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


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
