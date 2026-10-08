from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError


class TokenError(Exception):
    """Custom exception raised for invalid or expired JWT tokens."""
    pass


@dataclass
class JWTConfig:
    """Configuration for JWT creation and verification."""
    secret_key: str
    algorithm: str = "HS256"
    token_expiry_minutes: int = 60
    issuer: str = "clasp-security"


def create_token(subject: str, config: JWTConfig, claims: dict[str, Any] | None = None) -> str:
    """
    Create a signed JWT with subject, issuer, issued-at, expiration, and custom claims.
    
    Args:
        subject: The subject of the token (e.g., user or service ID).
        config: JWT configuration holding the secret and settings.
        claims: Optional dictionary of additional claims to include in the token.
        
    Returns:
        The encoded JWT string.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": subject,
        "iss": config.issuer,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=config.token_expiry_minutes)
    }
    if claims:
        # Avoid overriding standard claims
        for k, v in claims.items():
            if k not in payload:
                payload[k] = v
                
    return jwt.encode(payload, config.secret_key, algorithm=config.algorithm)


def verify_token(token: str, config: JWTConfig) -> dict[str, Any]:
    """
    Verify and decode a JWT.
    
    Args:
        token: The encoded JWT string.
        config: JWT configuration used for verification.
        
    Returns:
        A dictionary containing the decoded token payload.
        
    Raises:
        TokenError: If the token is invalid, expired, or verification fails.
    """
    try:
        decoded = jwt.decode(
            token,
            config.secret_key,
            algorithms=[config.algorithm],
            issuer=config.issuer
        )
        return decoded
    except ExpiredSignatureError as e:
        raise TokenError("Token has expired.") from e
    except InvalidTokenError as e:
        raise TokenError(f"Invalid token: {e}") from e
