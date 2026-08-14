"""mTLS handshake prototype between two standalone services.

Demonstrates mutual certificate authentication using ``asyncio`` and
Python's ``ssl`` module.  Both server and client present their certs;
connections without valid certs are rejected.

This prototype validates the security layer before integration into
the real Edge→Cluster upload pipeline (FastAPI + httpx).

References
----------
- NIST SP 1800-35 — Implementing a Zero Trust Architecture (2025)
- OWASP GenAI LLM Top 10 (2026)
"""
from __future__ import annotations

import asyncio
import logging
import ssl
from pathlib import Path

from security.tls_config import create_server_ssl_context, create_client_ssl_context

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 0  # OS picks a free port


class SecureServer:
    """Asyncio SSL server with mutual certificate verification.

    After calling :meth:`start`, the server listens on *host:port* and
    echoes back any data it receives (for testing purposes).
    """

    def __init__(
        self,
        server_cert: str | Path,
        server_key: str | Path,
        ca_cert: str | Path,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.ssl_ctx = create_server_ssl_context(server_cert, server_key, ca_cert)
        self.host = host
        self.port = port
        self._server: asyncio.Server | None = None

    async def start(self) -> int:
        """Start the server.  Returns the actual port being listened on."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
            ssl=self.ssl_ctx,
        )
        # Retrieve the actual port (useful when port=0)
        sock = self._server.sockets[0]
        actual_port = sock.getsockname()[1]
        self.port = actual_port
        logger.info("SecureServer listening on %s:%d", self.host, actual_port)
        return actual_port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Echo handler — reads one message, echoes it, then closes."""
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            if data:
                writer.write(data)
                await writer.drain()
        except (ssl.SSLError, asyncio.TimeoutError, ConnectionError) as exc:
            logger.debug("Client connection error: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ssl.SSLError, ConnectionError):
                pass


class SecureClient:
    """Asyncio SSL client with mutual certificate presentation."""

    def __init__(
        self,
        client_cert: str | Path,
        client_key: str | Path,
        ca_cert: str | Path,
        server_hostname: str = DEFAULT_HOST,
    ) -> None:
        self.ssl_ctx = create_client_ssl_context(
            client_cert, client_key, ca_cert, server_hostname
        )
        self.server_hostname = server_hostname

    async def send(
        self,
        host: str,
        port: int,
        message: bytes,
        timeout: float = 5.0,
    ) -> bytes:
        """Connect, send *message*, read echo, return it."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                host,
                port,
                ssl=self.ssl_ctx,
                server_hostname=self.server_hostname,
            ),
            timeout=timeout,
        )
        try:
            writer.write(message)
            await writer.drain()
            response = await asyncio.wait_for(reader.read(4096), timeout=timeout)
            return response
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ssl.SSLError, ConnectionError):
                pass
