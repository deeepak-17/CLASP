"""P2-side mTLS wiring against an *ephemeral test CA* (not P3's PKI).

Verifies the Cluster side of G1 only: the HTTP service started with
``uvicorn_ssl_kwargs`` accepts a client certificate signed by the configured CA
and refuses a client with no certificate or a certificate from another CA.
It does NOT verify P3's certificate issuance, identity mapping or a Flower
gRPC mTLS round — those are blocked by P3 (docs/INTEGRATION_BOUNDARIES.md).
Skipped only if ``cryptography`` is not installed.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
import ssl
import threading
import time

import httpx
import pytest

pytest.importorskip("cryptography")
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from cluster import server
from cluster.tls import (
    MTLSFiles,
    TLSConfigError,
    client_ssl_context,
    common_name_from_peercert,
    flower_certificates,
    uvicorn_ssl_kwargs,
)


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _issue(cn, issuer_key, issuer_name, *, is_ca=False, san=None, client=False):
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
    )
    if san:
        builder = builder.add_extension(x509.SubjectAlternativeName(san), critical=False)
    if client or not is_ca:
        eku = x509.ExtendedKeyUsageOID.CLIENT_AUTH if client else x509.ExtendedKeyUsageOID.SERVER_AUTH
        builder = builder.add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
    return key, builder.sign(issuer_key or key, hashes.SHA256())


def _write(tmp_path, stem, key, cert):
    cert_p, key_p = tmp_path / f"{stem}.crt", tmp_path / f"{stem}.key"
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_p, key_p


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pki")
    ca_key, ca_cert = _issue("test-ca", None, _name("test-ca"), is_ca=True)
    rogue_key, rogue_cert = _issue("rogue-ca", None, _name("rogue-ca"), is_ca=True)
    srv_key, srv_cert = _issue(
        "localhost", ca_key, ca_cert.subject,
        san=[x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))],
    )
    cli_key, cli_cert = _issue("client-1", ca_key, ca_cert.subject, client=True)
    bad_key, bad_cert = _issue("client-1", rogue_key, rogue_cert.subject, client=True)
    ca_p, _ = _write(tmp, "ca", ca_key, ca_cert)
    rogue_ca_p, _ = _write(tmp, "rogue-ca", rogue_key, rogue_cert)
    srv = MTLSFiles(ca_p, *_write(tmp, "server", srv_key, srv_cert))
    good = MTLSFiles(ca_p, *_write(tmp, "client", cli_key, cli_cert))
    rogue = MTLSFiles(ca_p, *_write(tmp, "rogue-client", bad_key, bad_cert))
    return {"server": srv, "client": good, "rogue_client": rogue, "rogue_ca": rogue_ca_p}


@pytest.fixture(scope="module")
def mtls_server(pki):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(
        server.app, host="127.0.0.1", port=port, log_level="error",
        **uvicorn_ssl_kwargs(pki["server"]),
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert srv.started, "uvicorn did not start"
    yield f"https://localhost:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


def test_client_with_a_certificate_from_the_configured_ca_is_served(mtls_server, pki):
    with httpx.Client(verify=client_ssl_context(pki["client"]), timeout=10) as c:
        r = c.get(f"{mtls_server}/healthz")
    assert r.status_code == 200 and r.json()["service"] == "cluster"


def test_client_without_a_certificate_is_refused(mtls_server, pki):
    ctx = ssl.create_default_context(cafile=str(pki["client"].ca_cert))  # trusts server, no client cert
    with pytest.raises((httpx.HTTPError, ssl.SSLError, OSError)), httpx.Client(
        verify=ctx, timeout=10
    ) as c:
        c.get(f"{mtls_server}/healthz")


def test_client_with_a_certificate_from_another_ca_is_refused(mtls_server, pki):
    with pytest.raises((httpx.HTTPError, ssl.SSLError, OSError)), httpx.Client(
        verify=client_ssl_context(pki["rogue_client"]), timeout=10
    ) as c:
        c.get(f"{mtls_server}/healthz")


def test_files_helpers(pki, tmp_path):
    kwargs = uvicorn_ssl_kwargs(pki["server"])
    assert kwargs["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert uvicorn_ssl_kwargs(pki["server"], require_client_cert=False)["ssl_cert_reqs"] == (
        ssl.CERT_OPTIONAL
    )
    ca, _cert, key = flower_certificates(pki["server"])
    assert ca.startswith(b"-----BEGIN CERTIFICATE") and key.startswith(b"-----BEGIN PRIVATE KEY")
    with pytest.raises(TLSConfigError, match="not found"):
        MTLSFiles(tmp_path / "x.crt", tmp_path / "y.crt", tmp_path / "z.key").checked()


def test_common_name_from_peercert_dict_shape():
    peercert = {"subject": ((("organizationName", "clasp"),), (("commonName", "client-1"),))}
    assert common_name_from_peercert(peercert) == "client-1"
    assert common_name_from_peercert({}) is None and common_name_from_peercert(None) is None
