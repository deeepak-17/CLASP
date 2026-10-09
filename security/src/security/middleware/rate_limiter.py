from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


@dataclass
class RateLimitConfig:
    """Configuration for the rate limiter."""
    requests_per_minute: int = 60
    burst_size: int = 10
    enabled: bool = True
    max_buckets: int = 10_000
    bucket_ttl_seconds: float = 300.0

class TokenBucket:
    """Internal token bucket implementation."""
    def __init__(self, capacity: int, fill_rate: float):
        """
        Initialize the token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold.
            fill_rate: Number of tokens added per second.
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = float(capacity)
        self.last_fill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume.
            
        Returns:
            bool: True if tokens were consumed, False if insufficient tokens.
        """
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_fill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.last_fill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class RateLimiter:
    """Manages per-client rate limiting."""
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the rate limiter.
        
        Args:
            config: Configuration for the rate limiter.
        """
        self.config = config
        self.fill_rate = config.requests_per_minute / 60.0
        self.buckets: dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def _get_bucket(self, client_id: str) -> TokenBucket:
        with self.lock:
            self._evict_stale_buckets()
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(
                    capacity=self.config.burst_size,
                    fill_rate=self.fill_rate
                )
            return self.buckets[client_id]

    def _evict_stale_buckets(self) -> None:
        """Remove idle buckets that have exceeded the TTL, or oldest when over max_buckets."""
        now = time.monotonic()
        stale = [
            cid for cid, bucket in self.buckets.items()
            if (now - bucket.last_fill) > self.config.bucket_ttl_seconds
        ]
        for cid in stale:
            del self.buckets[cid]
        # If still over capacity, evict the oldest buckets
        if len(self.buckets) >= self.config.max_buckets:
            sorted_ids = sorted(
                self.buckets, key=lambda cid: self.buckets[cid].last_fill
            )
            for cid in sorted_ids[: len(self.buckets) - self.config.max_buckets + 1]:
                del self.buckets[cid]

    def check(self, client_id: str) -> bool:
        """
        Check if a request is allowed for a given client.
        
        Args:
            client_id: Identifier for the client.
            
        Returns:
            bool: True if allowed, False otherwise.
        """
        if not self.config.enabled:
            return True
        bucket = self._get_bucket(client_id)
        return bucket.consume(1)

    def reset(self, client_id: str) -> None:
        """
        Reset a client's bucket.
        
        Args:
            client_id: Identifier for the client.
        """
        with self.lock:
            if client_id in self.buckets:
                del self.buckets[client_id]

    def get_stats(self, client_id: str) -> dict[str, Any]:
        """
        Get statistics for a client's bucket.
        
        Args:
            client_id: Identifier for the client.
            
        Returns:
            dict containing remaining tokens and time to next full refill.
        """
        if client_id not in self.buckets:
            return {"remaining": self.config.burst_size, "next_refill_seconds": 0.0}
        
        bucket = self._get_bucket(client_id)
        with bucket.lock:
            now = time.monotonic()
            elapsed = now - bucket.last_fill
            tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.fill_rate)
            missing = bucket.capacity - tokens
            time_to_fill = missing / bucket.fill_rate if missing > 0 else 0.0
            return {
                "remaining": int(tokens),
                "next_refill_seconds": time_to_fill
            }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app: Any, limiter: RateLimiter):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI app.
            limiter: The rate limiter instance.
        """
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and enforce rate limits.
        """
        if not self.limiter.config.enabled:
            return await call_next(request)

        # Use request.client.host — X-Forwarded-For is client-controlled and
        # trivially spoofed unless a trusted reverse proxy strips it.
        client_id = request.client.host if request.client else "unknown"

        if not self.limiter.check(client_id):
            stats = self.limiter.get_stats(client_id)
            headers = {
                "Retry-After": str(int(stats["next_refill_seconds"] + 1))
            }
            return Response(
                content="Too Many Requests", 
                status_code=429, 
                headers=headers
            )

        return await call_next(request)

def rate_limit_dependency(limiter: RateLimiter) -> Callable[..., Any]:
    """
    Create a FastAPI dependency for rate limiting.
    
    Args:
        limiter: The rate limiter instance.
        
    Returns:
        A dependency callable.
    """
    def dependency(request: Request) -> None:
        # Use request.client.host — X-Forwarded-For is client-controlled and
        # trivially spoofed unless a trusted reverse proxy strips it.
        client_id = request.client.host if request.client else "unknown"
            
        if not limiter.check(client_id):
            stats = limiter.get_stats(client_id)
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests",
                headers={
                    "Retry-After": str(int(stats["next_refill_seconds"] + 1))
                }
            )
    return dependency
