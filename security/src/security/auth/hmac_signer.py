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
    Sign adapter upload by concatenating sorted tensor names + data + json metadata.
    
    Args:
        tensors: Dictionary mapping tensor names to their byte data.
        metadata: Dictionary of metadata associated with the upload.
        secret: The secret key used for signing.
        
    Returns:
        The hex-encoded HMAC-SHA256 signature string.
    """
    payload_parts = []
    # Sort tensor names to ensure consistent ordering
    for name in sorted(tensors.keys()):
        payload_parts.append(name.encode("utf-8"))
        payload_parts.append(tensors[name])
        
    # Append serialized metadata (sorted keys for deterministic JSON)
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    payload_parts.append(metadata_bytes)
    
    # Concatenate all parts to form the final payload
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
