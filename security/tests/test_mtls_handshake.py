"""Tests for security.mtls handshake — server + client mutual authentication."""
from __future__ import annotations

import socket
import ssl
import tempfile
import threading
from pathlib import Path

import pytest

from security.mtls.ca import create_ca, save_pem
from security.mtls.certs import generate_client_cert, generate_service_cert, verify_cert
from security.mtls.context import client_ssl_context, server_ssl_context


@pytest.fixture(scope="module")
def pki_dir():
    """Create a full PKI hierarchy in a temp directory for the test session."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # Create CA
        ca_key, ca_cert = create_ca(common_name="Test CA")
        save_pem(ca_key, ca_cert, d, name="ca")

        # Create server cert
        srv_key, srv_cert = generate_service_cert(
            ca_key, ca_cert, "test-server",
            sans=["localhost", "127.0.0.1"],
        )
        save_pem(srv_key, srv_cert, d, name="server")

        # Create valid client cert
        cli_key, cli_cert = generate_client_cert(ca_key, ca_cert, "test-client")
        save_pem(cli_key, cli_cert, d, name="client")

        # Create a second CA (for bad-cert tests)
        rogue_ca_key, rogue_ca_cert = create_ca(common_name="Rogue CA")
        save_pem(rogue_ca_key, rogue_ca_cert, d, name="rogue-ca")

        # Create a client cert signed by the rogue CA
        bad_key, bad_cert = generate_client_cert(
            rogue_ca_key, rogue_ca_cert, "rogue-client",
        )
        save_pem(bad_key, bad_cert, d, name="bad-client")

        yield d


class TestVerifyCert:
    """Certificate validation against the CA."""

    def test_valid_cert_passes(self, pki_dir: Path) -> None:
        from security.mtls.ca import load_pem
        _, ca_cert = load_pem(pki_dir, "ca")
        _, srv_cert = load_pem(pki_dir, "server")
        assert verify_cert(srv_cert, ca_cert) is True

    def test_wrong_ca_fails(self, pki_dir: Path) -> None:
        from security.mtls.ca import load_pem
        _, rogue_ca = load_pem(pki_dir, "rogue-ca")
        _, srv_cert = load_pem(pki_dir, "server")
        assert verify_cert(srv_cert, rogue_ca) is False


class TestMutualHandshake:
    """Full mTLS handshake between a server and a client using SSL contexts."""

    @staticmethod
    def _run_server(srv_ctx: ssl.SSLContext, host: str, port: int,
                    result: dict, ready: threading.Event) -> None:
        """Background thread: accept one connection then close."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(1)
            ready.set()
            try:
                sock.settimeout(5.0)
                conn, _ = sock.accept()
                with srv_ctx.wrap_socket(conn, server_side=True) as tls:
                    tls.sendall(b"CLASP_OK")
                    result["server_ok"] = True
                    result["client_cert_subject"] = (
                        tls.getpeercert().get("subject") if tls.getpeercert() else None
                    )
            except Exception as exc:
                result["server_error"] = str(exc)

    def test_successful_handshake(self, pki_dir: Path) -> None:
        """Server and client with valid certs complete the handshake."""
        # File naming from ca.save_pem: {name}_cert.pem, {name}_key.pem
        srv_ctx = server_ssl_context(
            cert_path=pki_dir / "server_cert.pem",
            key_path=pki_dir / "server_key.pem",
            ca_cert_path=pki_dir / "ca_cert.pem",
        )
        cli_ctx = client_ssl_context(
            cert_path=pki_dir / "client_cert.pem",
            key_path=pki_dir / "client_key.pem",
            ca_cert_path=pki_dir / "ca_cert.pem",
        )

        host = "127.0.0.1"
        result: dict = {}
        ready = threading.Event()

        # Bind to pick port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]

        srv_thread = threading.Thread(
            target=self._run_server, args=(srv_ctx, host, port, result, ready),
            daemon=True,
        )
        srv_thread.start()
        ready.wait(timeout=3.0)

        with socket.create_connection((host, port), timeout=5.0) as sock:
            with cli_ctx.wrap_socket(sock, server_hostname="localhost") as tls:
                data = tls.recv(1024)
                assert data == b"CLASP_OK"
                result["client_ok"] = True

        srv_thread.join(timeout=5.0)
        assert result.get("server_ok") is True
        assert result.get("client_ok") is True

    def test_bad_client_cert_rejected(self, pki_dir: Path) -> None:
        """A client cert signed by a rogue CA should be rejected by the server."""
        srv_ctx = server_ssl_context(
            cert_path=pki_dir / "server_cert.pem",
            key_path=pki_dir / "server_key.pem",
            ca_cert_path=pki_dir / "ca_cert.pem",  # server trusts only the real CA
        )
        # Client uses a cert signed by the rogue CA.
        cli_ctx = client_ssl_context(
            cert_path=pki_dir / "bad-client_cert.pem",
            key_path=pki_dir / "bad-client_key.pem",
            ca_cert_path=pki_dir / "rogue-ca_cert.pem",
        )

        host = "127.0.0.1"
        result: dict = {}
        ready = threading.Event()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]

        srv_thread = threading.Thread(
            target=self._run_server, args=(srv_ctx, host, port, result, ready),
            daemon=True,
        )
        srv_thread.start()
        ready.wait(timeout=3.0)

        with pytest.raises((ssl.SSLError, ssl.SSLCertVerificationError, ConnectionResetError,
                            OSError)):
            with socket.create_connection((host, port), timeout=5.0) as sock:
                with cli_ctx.wrap_socket(sock, server_hostname="localhost") as tls:
                    tls.recv(1024)

        srv_thread.join(timeout=5.0)
        # Server should NOT have marked success.
        assert result.get("server_ok") is not True
