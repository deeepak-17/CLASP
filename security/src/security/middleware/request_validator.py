from __future__ import annotations

import re
import os
import logging
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for validation errors."""
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation error for '{field}': {reason}")

def sanitize_client_id(client_id: str) -> str:
    """
    Validate and sanitize client IDs.
    Must be alphanumeric + hyphens only, max 64 chars.
    
    Args:
        client_id: The client identifier to sanitize.
        
    Returns:
        The sanitized client ID.
        
    Raises:
        ValidationError: If the client ID is invalid.
    """
    if not client_id or not isinstance(client_id, str):
        raise ValidationError("client_id", "Must be a non-empty string.")
    
    if len(client_id) > 64:
        raise ValidationError("client_id", "Exceeds maximum length of 64 characters.")
        
    if not re.match(r'^[a-zA-Z0-9\-]+$', client_id):
        raise ValidationError("client_id", "Must contain only alphanumeric characters and hyphens.")
        
    return client_id

def sanitize_path(path: str) -> str:
    """
    Prevent path traversal attacks.
    
    Args:
        path: The path string to sanitize.
        
    Returns:
        The sanitized path.
        
    Raises:
        ValidationError: If '..' or absolute paths are detected.
    """
    if not path or not isinstance(path, str):
        raise ValidationError("path", "Must be a non-empty string.")
        
    if ".." in path:
        raise ValidationError("path", "Path traversal detected.")
        
    if os.path.isabs(path) or path.startswith("/") or path.startswith("\\"):
        raise ValidationError("path", "Absolute paths are not allowed.")
        
    return path

def validate_tensor_payload(
    tensors: list[dict[str, Any]], 
    max_tensors: int = 1000, 
    max_tensor_bytes: int = 100 * 1024 * 1024
) -> None:
    """
    Validate tensor upload payloads.
    
    Args:
        tensors: List of tensor dictionaries.
        max_tensors: Maximum number of allowed tensors.
        max_tensor_bytes: Maximum allowed bytes per tensor.
        
    Raises:
        ValidationError: If the payload is invalid.
    """
    if not isinstance(tensors, list):
        raise ValidationError("tensors", "Payload must be a list of tensors.")
        
    if len(tensors) > max_tensors:
        raise ValidationError("tensors", f"Exceeds maximum allowed tensors ({max_tensors}).")
        
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            raise ValidationError(f"tensors[{i}]", "Must be a dictionary.")
            
        for field in ("name", "dtype", "shape", "data_b64"):
            if field not in tensor:
                raise ValidationError(f"tensors[{i}].{field}", "Missing required field.")
                
        # Basic check for base64 size limit (very rough approximation)
        data = tensor.get("data_b64", "")
        if isinstance(data, str) and (len(data) * 3 / 4) > max_tensor_bytes:
             raise ValidationError(f"tensors[{i}].data_b64", f"Exceeds maximum tensor size ({max_tensor_bytes} bytes).")

def validate_round_id(round_id: int) -> None:
    """
    Ensure round_id is a non-negative integer.
    
    Args:
        round_id: The round ID to validate.
        
    Raises:
        ValidationError: If the round ID is invalid.
    """
    if not isinstance(round_id, int):
        raise ValidationError("round_id", "Must be an integer.")
    
    if round_id < 0:
        raise ValidationError("round_id", "Must be non-negative.")

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to log and reject requests with suspiciously large bodies."""
    
    def __init__(self, app: Any, max_body_bytes: int = 200 * 1024 * 1024):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI app.
            max_body_bytes: Maximum allowed request body size in bytes.
        """
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process the request and check body size.
        """
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_body_bytes:
                    logger.warning(f"Rejected request due to large body size: {length} bytes")
                    return Response("Payload Too Large", status_code=413)
            except ValueError:
                pass
                
        return await call_next(request)
