"""``python -m evaluation``: the registry check presents a certificate under mTLS."""
from __future__ import annotations

import ssl

import pytest

from evaluation import __main__ as entry


def test_a_plain_registry_needs_no_context():
    assert entry.tls_context("http://registry:8004", {}) is None


def test_an_https_registry_is_dialled_with_the_client_certificate(tmp_path):
    seen = {}

    class Context:
        minimum_version = None

        def load_cert_chain(self, cert, key):
            seen["chain"] = (cert, key)

    def fake(cafile=None):
        seen["ca"] = cafile
        return Context()

    env = {"CLASP_TLS_CLIENT_CERT": "/certs/demo-ui.pem", "CLASP_TLS_CLIENT_KEY": "/certs/demo-ui.key",
           "CLASP_TLS_CA": "/certs/ca.pem"}
    ctx = entry.tls_context("https://registry:8004", env, make=fake)
    assert seen == {"ca": "/certs/ca.pem", "chain": ("/certs/demo-ui.pem", "/certs/demo-ui.key")}
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3


def test_an_https_registry_without_the_certificate_variables_is_a_clear_error():
    with pytest.raises(ValueError, match="CLASP_TLS_CLIENT_CERT"):
        entry.tls_context("https://registry:8004", {"CLASP_TLS_CA": "/certs/ca.pem"})


def test_the_check_reports_a_missing_certificate_as_unreachable(monkeypatch, capsys):
    monkeypatch.setenv("CLASP_REGISTRY_URL", "https://registry:8004")
    for var in ("CLASP_TLS_CLIENT_CERT", "CLASP_TLS_CLIENT_KEY", "CLASP_TLS_CA"):
        monkeypatch.delenv(var, raising=False)
    assert entry.main() == 1
    assert "CLASP_TLS_CLIENT_CERT" in capsys.readouterr().err
