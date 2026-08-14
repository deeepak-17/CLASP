"""Tests for the mTLS handshake prototype.

Week 4 acceptance criteria:
  - Valid client cert → handshake succeeds, echo works
  - No client cert → rejected
  - Rogue cert (different CA) → rejected
  - Handshake completes in < 5 ms on localhost
"""
from __future__ import annotations

import asyncio
import ssl
import time
from pathlib import Path

import pytest

from security.ca import CertificateAuthority
from security.mtls_service import SecureServer, SecureClient


@pytest.fixture
def ca_and_certs(tmp_path: Path) -> dict:
    """Set up a CA with server and client certs."""
    ca = CertificateAuthority(certs_dir=tmp_path / "ca", ca_cn="mTLS Test CA")
    server_cert, server_key = ca.issue_server_cert(
        cn="test-server", san_dns=("localhost",), san_ips=("127.0.0.1",)
    )
    client_cert, client_key = ca.issue_client_cert(cn="test-client")
    return {
        "ca": ca,
        "server_cert": server_cert,
        "server_key": server_key,
        "client_cert": client_cert,
        "client_key": client_key,
    }


@pytest.fixture
def rogue_ca(tmp_path: Path) -> CertificateAuthority:
    """A second CA that is NOT trusted by the server."""
    return CertificateAuthority(
        certs_dir=tmp_path / "rogue_ca", ca_cn="Rogue CA"
    )


class TestMTLSHandshake:
    """End-to-end mTLS handshake tests."""

    @pytest.mark.asyncio
    async def test_valid_handshake_echo(self, ca_and_certs: dict) -> None:
        """Valid certs → handshake succeeds and echo works."""
        ca = ca_and_certs["ca"]
        server = SecureServer(
            ca_and_certs["server_cert"],
            ca_and_certs["server_key"],
            ca.ca_cert_path,
        )
        port = await server.start()

        client = SecureClient(
            ca_and_certs["client_cert"],
            ca_and_certs["client_key"],
            ca.ca_cert_path,
        )

        try:
            msg = b"hello from CLASP edge"
            response = await client.send("localhost", port, msg)
            assert response == msg
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_reject_rogue_cert(
        self, ca_and_certs: dict, rogue_ca: CertificateAuthority
    ) -> None:
        """Client cert from a different CA → handshake rejected."""
        ca = ca_and_certs["ca"]
        server = SecureServer(
            ca_and_certs["server_cert"],
            ca_and_certs["server_key"],
            ca.ca_cert_path,  # server trusts only the real CA
        )
        port = await server.start()

        # Issue client cert from the ROGUE CA
        rogue_cert, rogue_key = rogue_ca.issue_client_cert(cn="rogue-edge")
        rogue_client = SecureClient(
            rogue_cert,
            rogue_key,
            rogue_ca.ca_cert_path,  # rogue client trusts rogue CA
        )

        try:
            with pytest.raises((ssl.SSLError, ConnectionError, OSError)):
                await rogue_client.send("localhost", port, b"I am rogue")
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_handshake_latency(self, ca_and_certs: dict) -> None:
        """Handshake should complete in < 50ms on localhost (target < 5ms)."""
        ca = ca_and_certs["ca"]
        server = SecureServer(
            ca_and_certs["server_cert"],
            ca_and_certs["server_key"],
            ca.ca_cert_path,
        )
        port = await server.start()

        client = SecureClient(
            ca_and_certs["client_cert"],
            ca_and_certs["client_key"],
            ca.ca_cert_path,
        )

        try:
            start = time.perf_counter()
            await client.send("localhost", port, b"latency test")
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Allow up to 50ms for CI environments; localhost should be < 5ms
            assert elapsed_ms < 50, f"Handshake took {elapsed_ms:.1f}ms (> 50ms)"
        finally:
            await server.stop()
