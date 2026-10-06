"""Edge transport: plain HTTP, and mTLS built on P3's security library (D7).

The mTLS tests stand up a real TLS 1.3 server with a certificate from P3's
``security.CertificateAuthority`` and P3's server SSL context, then talk to it
through ``edge.transport.mtls_session``. They skip themselves when the installed
``security`` package does not provide the mTLS API yet (it lives on P3's branch
until it is integrated) — the plain-HTTP behaviour is tested regardless.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from edge import transport


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server's naming
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        ok = payload.get("client_id") == "good"
        body = json.dumps({"received": payload.get("client_id")}).encode()
        self.send_response(201 if ok else 422)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


def _serve(server: HTTPServer):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def plain_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    _serve(server)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_plain_session_needs_no_security_library():
    assert isinstance(transport.session(), requests.Session)


def test_post_upload_returns_the_cluster_reply(plain_server):
    reply = transport.post_upload(requests.Session(), plain_server, {"client_id": "good"})
    assert reply == {"received": "good"}


def test_post_upload_raises_with_the_clusters_message(plain_server):
    with pytest.raises(RuntimeError, match="422"):
        transport.post_upload(requests.Session(), plain_server, {"client_id": "bad"})


# --------------------------------------------------------------------------- #
# mTLS — needs P3's security library
# --------------------------------------------------------------------------- #
@pytest.fixture
def pki(tmp_path):
    security = pytest.importorskip("security")
    if not all(hasattr(security, n) for n in ("CertificateAuthority",
                                              "create_server_ssl_context",
                                              "create_client_ssl_context")):
        pytest.skip("installed security package has no mTLS API yet (P3 branch not integrated)")
    ca = security.CertificateAuthority(certs_dir=tmp_path / "certs")
    server_cert, server_key = ca.issue_server_cert(san_dns=("localhost",), san_ips=("127.0.0.1",))
    client_cert, client_key = ca.issue_client_cert(cn="clasp-edge-test")
    return security, ca.ca_cert_path, (server_cert, server_key), (client_cert, client_key)


@pytest.fixture
def mtls_server(pki):
    security, ca_cert, (server_cert, server_key), _ = pki
    ctx = security.create_server_ssl_context(server_cert, server_key, ca_cert)
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    _serve(server)
    yield f"https://localhost:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_mtls_session_completes_a_mutual_handshake(pki, mtls_server):
    _, ca_cert, _, (client_cert, client_key) = pki
    http = transport.mtls_session(client_cert, client_key, ca_cert)
    r = http.get(f"{mtls_server}/healthz", timeout=10)
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_mtls_upload_goes_over_the_authenticated_channel(pki, mtls_server):
    _, ca_cert, _, (client_cert, client_key) = pki
    http = transport.session({"client_cert": str(client_cert), "client_key": str(client_key),
                              "ca_cert": str(ca_cert)})
    assert transport.post_upload(http, mtls_server, {"client_id": "good"}) == {"received": "good"}


def test_server_refuses_a_client_without_a_certificate(pki, mtls_server):
    """CERT_REQUIRED on the server side: trusting the CA is not enough, the
    edge has to present its own certificate."""
    _, ca_cert, _, _ = pki
    anonymous = requests.Session()
    anonymous.verify = str(ca_cert)
    with pytest.raises(requests.exceptions.RequestException):
        anonymous.get(f"{mtls_server}/healthz", timeout=10)


def test_edge_refuses_a_server_outside_the_clasp_ca(pki, mtls_server, tmp_path):
    """And the other direction: a server certificate from a different CA is
    rejected by the edge before any request is sent."""
    security, _, _, (client_cert, client_key) = pki
    other_ca = security.CertificateAuthority(certs_dir=tmp_path / "other")
    http = transport.mtls_session(client_cert, client_key, other_ca.ca_cert_path)
    with pytest.raises(requests.exceptions.SSLError):
        http.get(f"{mtls_server}/healthz", timeout=10)
