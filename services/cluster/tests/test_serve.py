"""``python -m cluster.serve``: mTLS switched on from the environment (demo plan v3 item 9).

Against an ephemeral test CA (the helpers of ``test_tls_p2_side``): the launcher
refuses a partial TLS config, serves TLS 1.3 only, and binds an upload's
``client_id`` to the CN of the verified client certificate.
"""

from __future__ import annotations

import socket
import ssl
import threading
import time

import httpx
import pytest

pytest.importorskip("cryptography")
import uvicorn
from cryptography import x509

from cluster import healthcheck, serve, server
from cluster.tls import MTLSFiles, TLSConfigError, client_ssl_context, outbound_ssl_context
from tests.conftest import trained_adapter
from tests.http_helpers import reset_server_state, restore_server_state, upload_body
from tests.test_tls_p2_side import _issue, _name, _write


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    import ipaddress

    tmp = tmp_path_factory.mktemp("serve-pki")
    ca_key, ca_cert = _issue("test-ca", None, _name("test-ca"), is_ca=True)
    srv_key, srv_cert = _issue(
        "cluster", ca_key, ca_cert.subject,
        san=[x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))],
    )
    ca_p, _ = _write(tmp, "ca", ca_key, ca_cert)
    srv_cert_p, srv_key_p = _write(tmp, "cluster", srv_key, srv_cert)
    out = {"ca": ca_p, "cert": srv_cert_p, "key": srv_key_p}
    for cn in ("alice", "bob"):
        key, cert = _issue(cn, ca_key, ca_cert.subject, client=True)
        out[cn] = MTLSFiles(ca_p, *_write(tmp, cn, key, cert))
    return out


def _env(pki, port, **extra):
    return {"CLASP_TLS_CERT": str(pki["cert"]), "CLASP_TLS_KEY": str(pki["key"]),
            "CLASP_TLS_CA": str(pki["ca"]), "CLASP_CLUSTER_HOST": "127.0.0.1",
            "CLASP_CLUSTER_PORT": str(port), **extra}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running(pki):
    port = _free_port()
    env = _env(pki, port)
    srv = uvicorn.Server(serve.build_config(env))
    srv.config.log_level = "error"
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert srv.started, "uvicorn did not start"
    yield f"https://localhost:{port}", env
    srv.should_exit = True
    thread.join(timeout=10)


def test_tls_env_is_all_or_nothing(pki):
    assert serve.tls_files_from_env({}) is None
    with pytest.raises(TLSConfigError, match="missing CLASP_TLS_CA"):
        serve.tls_files_from_env({"CLASP_TLS_CERT": str(pki["cert"]), "CLASP_TLS_KEY": str(pki["key"])})
    with pytest.raises(TLSConfigError, match="not found"):
        serve.tls_files_from_env({**_env(pki, 1), "CLASP_TLS_CA": str(pki["ca"]) + ".missing"})


def test_plain_config_has_no_tls_and_no_identity_binding():
    config = serve.build_config({"CLASP_CLUSTER_PORT": "8002"})
    assert config.ssl is None and server._identity.provider is None


def test_upload_is_bound_to_the_client_certificate(running, pki):
    url, env = running
    with httpx.Client(verify=client_ssl_context(pki["alice"]), timeout=10) as alice:
        assert alice.put(f"{url}/clusters/team/members",
                         json={"client_ids": ["alice", "bob"]}).status_code == 200
        ok = alice.post(f"{url}/clusters/team/uploads", json=upload_body("alice", trained_adapter(1)))
        assert ok.status_code == 201, ok.text
        spoof = alice.post(f"{url}/clusters/team/uploads", json=upload_body("bob", trained_adapter(2)))
        assert spoof.status_code == 403 and "'alice' may not upload as client 'bob'" in spoof.text
    # the test's server cert is serverAuth-only, so probe with a client cert
    probe = {**env, "CLASP_HEALTHCHECK_CERT": str(pki["alice"].cert),
             "CLASP_HEALTHCHECK_KEY": str(pki["alice"].key)}
    assert healthcheck.check(probe)


def test_tls_1_2_is_refused(running, pki):
    url, _ = running
    ctx = client_ssl_context(pki["alice"])
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    with pytest.raises((httpx.HTTPError, ssl.SSLError, OSError)), httpx.Client(verify=ctx, timeout=10) as c:
        c.get(f"{url}/healthz")


def test_outbound_context_follows_the_environment(pki):
    assert outbound_ssl_context({}) is None
    ctx = outbound_ssl_context(_env(pki, 1))
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.minimum_version == ssl.TLSVersion.TLSv1_3
