from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers."""
    content_security_policy: str = "default-src 'self'"
    x_content_type_options: str = "nosniff"
    x_frame_options: str = "DENY"
    strict_transport_security: str = "max-age=31536000; includeSubDomains"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    cache_control: str = "no-store"
    permissions_policy: str = "geolocation=(), microphone=()"

@dataclass
class CORSConfig:
    """Configuration for CORS."""
    allowed_origins: list[str] = field(default_factory=list)
    allowed_methods: list[str] = field(default_factory=lambda: ["GET"])
    allowed_headers: list[str] = field(default_factory=list)
    allow_credentials: bool = False

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app: Any, config: SecurityHeadersConfig | None = None):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI app.
            config: Security headers configuration.
        """
        super().__init__(app)
        self.config = config or self.default_config()

    @classmethod
    def default_config(cls) -> SecurityHeadersConfig:
        """Get the default security headers configuration."""
        return SecurityHeadersConfig()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and add security headers to the response.
        """
        response = await call_next(request)
        
        headers = {
            "Content-Security-Policy": self.config.content_security_policy,
            "X-Content-Type-Options": self.config.x_content_type_options,
            "X-Frame-Options": self.config.x_frame_options,
            "Strict-Transport-Security": self.config.strict_transport_security,
            "X-XSS-Protection": self.config.x_xss_protection,
            "Referrer-Policy": self.config.referrer_policy,
            "Cache-Control": self.config.cache_control,
            "Permissions-Policy": self.config.permissions_policy
        }
        
        for key, value in headers.items():
            if value:
                response.headers[key] = value
                
        return response

def apply_security_headers(
    app: FastAPI, 
    config: SecurityHeadersConfig | None = None, 
    cors: CORSConfig | None = None
) -> None:
    """
    Apply security headers and CORS middleware to a FastAPI app.
    
    Args:
        app: The FastAPI application.
        config: Security headers configuration.
        cors: CORS configuration.
    """
    app.add_middleware(
        SecurityHeadersMiddleware,
        config=config
    )
    
    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allowed_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allowed_methods,
            allow_headers=cors.allowed_headers,
        )
