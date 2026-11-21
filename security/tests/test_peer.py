"""security.mtls.peer: the verified peer certificate name, read in one place.

uvicorn verifies a client certificate during the handshake but does not put it
in the ASGI scope. ``PeerCertH11Protocol`` reads the verified certificate's CN
off the socket when the connection opens and adds it to every request's scope;
``peer_identity`` reads it back. The cluster and the registry both serve with
it, so "who is calling" means the same thing in both.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from security.mtls.ca import create_ca, save_pem
from security.mtls.certs import generate_client_cert, generate_service_cert
from security.mtls.context import client_ssl_context
from security.mtls.peer import (
    SCOPE_KEY,
    UNNAMED_PEER,
    PeerCertH11Protocol,
    common_name_from_peercert,
    peer_identity,
    peer_name_for,
)

uvicorn = pytest.importorskip("uvicorn")


def test_common_name_from_peercert_dict_shape():
    peercert = {"subject": ((("countryName", "IN"),), (("commonName", "client-1"),))}
    assert common_name_from_peercert(peercert) == "client-1"
    assert common_name_from_peercert({}) is None and common_name_from_peercert(None) is None
    assert common_name_from_peercert({"subject": ((("countryName", "IN"),),)}) is None


def test_peer_identity_reads_the_scope_key():
    assert SCOPE_KEY == "clasp.peer_common_name"
    assert peer_identity(SimpleNamespace(scope={SCOPE_KEY: "client-flask"})) == "client-flask"
    assert peer_identity(SimpleNamespace(scope={})) is None


def test_a_verified_certificate_without_a_name_is_not_the_no_tls_case():
    """``None`` means "no TLS" and opens every route (U6). A TLS peer whose
    certificate carries no CN must not be read as that: it gets a name that
    maps to no role, so restricted routes refuse it."""
    from security.auth.roles import ROLE_UNKNOWN, role_of

    no_cn = SimpleNamespace(getpeercert=lambda: {"subject": ((("countryName", "IN"),),)})
    named = SimpleNamespace(getpeercert=lambda: {"subject": ((("commonName", "cluster"),),)})
    assert peer_name_for(None) is None
    assert peer_name_for(no_cn) == UNNAMED_PEER
    assert peer_name_for(SimpleNamespace(getpeercert=lambda: None)) == UNNAMED_PEER
    assert peer_name_for(named) == "cluster"
    assert role_of(UNNAMED_PEER) == ROLE_UNKNOWN


def test_the_package_exports_the_helpers():
    from security import mtls

    for name in ("SCOPE_KEY", "PeerCertH11Protocol", "common_name_from_peercert", "peer_identity"):
        assert getattr(mtls, name) is globals()[name]


async def _whoami(scope, receive, send):
    """The smallest ASGI app: answers with the peer name the protocol put in the scope."""
    assert scope["type"] == "http"
    body = repr(scope.get(SCOPE_KEY)).encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": body})


def _serve(**kwargs):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(_whoami, host="127.0.0.1", port=port,
                                           log_level="error", http=PeerCertH11Protocol, **kwargs))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    return server, thread, port


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    d = tmp_path_factory.mktemp("pki")
    ca_key, ca_cert = create_ca(common_name="Test CA", key_size=2048)
    save_pem(ca_key, ca_cert, d, name="ca")
    key, cert = generate_service_cert(ca_key, ca_cert, "registry", sans=["localhost", "127.0.0.1"])
    save_pem(key, cert, d, name="server")
    key, cert = generate_client_cert(ca_key, ca_cert, "client-flask")
    save_pem(key, cert, d, name="client")
    return d


def test_the_protocol_puts_the_verified_client_name_in_every_request_scope(pki):
    import ssl

    server, thread, port = _serve(
        ssl_certfile=str(pki / "server_cert.pem"), ssl_keyfile=str(pki / "server_key.pem"),
        ssl_ca_certs=str(pki / "ca_cert.pem"), ssl_cert_reqs=ssl.CERT_REQUIRED)
    try:
        ctx = client_ssl_context(pki / "client_cert.pem", pki / "client_key.pem",
                                 pki / "ca_cert.pem")
        with httpx.Client(verify=ctx) as http:
            # two requests on one connection: both carry the name
            assert http.get(f"https://localhost:{port}/").text == "'client-flask'"
            assert http.get(f"https://localhost:{port}/again").text == "'client-flask'"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_without_tls_the_peer_name_is_none():
    server, thread, port = _serve()
    try:
        assert httpx.get(f"http://127.0.0.1:{port}/").text == "None"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_the_library_imports_without_torch_or_opacus():
    """The registry uses the mTLS, auth, middleware and audit parts of this
    library. They must load on a machine that has neither torch nor opacus
    (the registry image), and the DP names must still be importable."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"      # an import of either now raises ImportError
        "sys.modules['opacus'] = None\n"
        "import security\n"
        "import security.mtls.peer, security.auth, security.middleware, security.audit\n"
        "from security.dp.ledger import RUN_EPSILON_BUDGET, RUN_DELTA, RUN_ROUNDS\n"
        "assert (RUN_EPSILON_BUDGET, RUN_DELTA, RUN_ROUNDS) == (8.0, 1e-5, 2)\n"
        "print('ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_opacus_is_an_extra_not_a_hard_dependency():
    """Installing the library for its mTLS/auth parts must not pull in torch."""
    import tomllib
    from pathlib import Path

    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    hard = " ".join(project["dependencies"]).lower()
    assert "opacus" not in hard and "torch" not in hard
    extras = project["optional-dependencies"]
    assert any(d.startswith("opacus") for d in extras["dp"])
    assert any(d.startswith("opacus") for d in extras["test"])  # CI still runs the DP tests
    assert "uvicorn" in hard  # PeerCertH11Protocol
