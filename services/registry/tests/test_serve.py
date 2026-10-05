"""Registry launcher: plain HTTP by default, mTLS (D7) when certs are configured.

These start the real app on a real socket with a throwaway CA, then check the
handshake both ways: a client with a CA-signed cert gets in, a client without
one is refused at the TLS layer, and partial TLS config refuses to start.
"""
from __future__ import annotations

import datetime as dt
import socket
import ssl
import threading
import time

import httpx
import pytest
from registry.serve import TLSConfigError, build_config, tls_settings_from_env

x509 = pytest.importorskip("cryptography.x509")
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402


def _key():
    return ec.generate_private_key(ec.SECP256R1())


def _cert(subject_cn, key, issuer_cn, issuer_key, *, ca=False, san=None):
    now = dt.datetime.now(dt.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_cn)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_cn)]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=1))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if san:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(san)]), critical=False
        )
    return builder.sign(issuer_key, hashes.SHA256())


def _write(path, cert=None, key=None):
    if cert is not None:
        path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    else:
        path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    return path


@pytest.fixture
def pki(tmp_path):
    ca_key = _key()
    ca = _cert("clasp-test-ca", ca_key, "clasp-test-ca", ca_key, ca=True)
    srv_key, cli_key = _key(), _key()
    srv = _cert("registry", srv_key, "clasp-test-ca", ca_key, san="localhost")
    cli = _cert("edge-client", cli_key, "clasp-test-ca", ca_key)
    return {
        "ca": _write(tmp_path / "ca.pem", cert=ca),
        "server_cert": _write(tmp_path / "server.pem", cert=srv),
        "server_key": _write(tmp_path / "server.key", key=srv_key),
        "client_cert": _write(tmp_path / "client.pem", cert=cli),
        "client_key": _write(tmp_path / "client.key", key=cli_key),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running(tmp_path, monkeypatch):
    """Start the registry via build_config in a thread; yield its port."""
    import uvicorn

    servers = []

    def start(env: dict[str, str]):
        monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "data"))
        import registry.app as appmod

        appmod._store = None
        port = _free_port()
        config = build_config({**env, "CLASP_REGISTRY_HOST": "127.0.0.1",
                               "CLASP_REGISTRY_PORT": str(port)})
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started, "registry did not start"
        servers.append((server, thread))
        return port

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_no_tls_env_means_plain_http():
    assert tls_settings_from_env({}) is None


def test_partial_tls_config_refuses_to_start(pki):
    with pytest.raises(TLSConfigError, match="CLASP_TLS_CA"):
        tls_settings_from_env({"CLASP_TLS_CERT": str(pki["server_cert"]),
                               "CLASP_TLS_KEY": str(pki["server_key"])})


def test_missing_cert_file_refuses_to_start(pki, tmp_path):
    with pytest.raises(TLSConfigError, match="not found"):
        tls_settings_from_env({"CLASP_TLS_CERT": str(tmp_path / "nope.pem"),
                               "CLASP_TLS_KEY": str(pki["server_key"]),
                               "CLASP_TLS_CA": str(pki["ca"])})


def test_bad_port_refuses_to_start():
    with pytest.raises(TLSConfigError, match="PORT"):
        build_config({"CLASP_REGISTRY_PORT": "eighty"})


# --------------------------------------------------------------------------- #
# live sockets
# --------------------------------------------------------------------------- #
def test_plain_http_serves_healthz(running):
    port = running({})
    r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=5)
    assert r.status_code == 200 and r.json()["service"] == "registry"


def _tls_env(pki):
    return {"CLASP_TLS_CERT": str(pki["server_cert"]), "CLASP_TLS_KEY": str(pki["server_key"]),
            "CLASP_TLS_CA": str(pki["ca"])}


def test_mtls_admits_a_client_with_a_ca_signed_cert(running, pki):
    port = running(_tls_env(pki))
    ctx = ssl.create_default_context(cafile=str(pki["ca"]))
    ctx.load_cert_chain(str(pki["client_cert"]), str(pki["client_key"]))
    r = httpx.get(f"https://localhost:{port}/healthz", verify=ctx, timeout=5)
    assert r.status_code == 200


def test_mtls_refuses_a_client_without_a_cert(running, pki):
    port = running(_tls_env(pki))
    ctx = ssl.create_default_context(cafile=str(pki["ca"]))
    with pytest.raises((httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        httpx.get(f"https://localhost:{port}/healthz", verify=ctx, timeout=5)


def test_mtls_server_negotiates_tls_1_3_only(running, pki):
    port = running(_tls_env(pki))
    ctx = ssl.create_default_context(cafile=str(pki["ca"]))
    ctx.load_cert_chain(str(pki["client_cert"]), str(pki["client_key"]))
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    with pytest.raises((httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        httpx.get(f"https://localhost:{port}/healthz", verify=ctx, timeout=5)


# --------------------------------------------------------------------------- #
# container healthcheck
# --------------------------------------------------------------------------- #
def test_healthcheck_plain(running):
    from registry.healthcheck import check

    port = running({})
    assert check({"CLASP_REGISTRY_PORT": str(port)}) is True


def test_healthcheck_under_mtls_presents_a_client_cert(running, pki):
    from registry.healthcheck import check

    env = _tls_env(pki)
    port = running(env)
    assert check({**env, "CLASP_REGISTRY_PORT": str(port),
                  "CLASP_HEALTHCHECK_CERT": str(pki["client_cert"]),
                  "CLASP_HEALTHCHECK_KEY": str(pki["client_key"])}) is True


def test_healthcheck_reports_down(pki):
    from registry.healthcheck import check

    assert check({"CLASP_REGISTRY_PORT": str(_free_port())}) is False
