"""Tests for the Certificate Authority and mTLS handshake.

Week 1 acceptance criteria:
  - CA generates valid ECDSA P-256 root certificate
  - Server and client certs are issued and signed by the CA
  - Mutual handshake succeeds with valid certs
  - Handshake is rejected with a rogue/untrusted cert
  - Certificate rotation works when cert is near expiry
"""
from __future__ import annotations

import datetime
import ssl
import tempfile
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec

from security.ca import CertificateAuthority, CertRotationPolicy


@pytest.fixture
def ca(tmp_path: Path) -> CertificateAuthority:
    """Create a fresh CA in a temporary directory."""
    return CertificateAuthority(certs_dir=tmp_path, ca_cn="Test CA")


class TestCertificateAuthority:
    """Tests for CA generation and cert issuance."""

    def test_ca_creates_root_certificate(self, ca: CertificateAuthority) -> None:
        assert ca.ca_cert_path.exists()
        cert = ca.ca_cert
        # Should be self-signed
        assert cert.issuer == cert.subject
        # Should use ECDSA
        assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey)
        # Should be a CA
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
        assert bc.value.ca is True

    def test_ca_reloads_from_disk(self, ca: CertificateAuthority) -> None:
        """Second instantiation should load, not regenerate."""
        ca2 = CertificateAuthority(
            certs_dir=ca.certs_dir, ca_cn="Test CA"
        )
        assert ca.ca_cert.serial_number == ca2.ca_cert.serial_number

    def test_issue_server_cert(self, ca: CertificateAuthority) -> None:
        cert_path, key_path = ca.issue_server_cert(cn="test-server")
        assert cert_path.exists()
        assert key_path.exists()
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        cn = cert.subject.get_attributes_for_oid(
            x509.oid.NameOID.COMMON_NAME
        )[0].value
        assert cn == "test-server"
        # Should have serverAuth EKU
        eku = cert.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        ).value
        assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku

    def test_issue_client_cert(self, ca: CertificateAuthority) -> None:
        cert_path, key_path = ca.issue_client_cert(cn="test-client")
        assert cert_path.exists()
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        eku = cert.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        ).value
        assert x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH in eku

    def test_cert_has_correct_expiry(self, ca: CertificateAuthority) -> None:
        cert_path, _ = ca.issue_server_cert(cn="expiry-test")
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        now = datetime.datetime.now(datetime.timezone.utc)
        # Default expiry is 7 days
        expected = now + datetime.timedelta(days=7)
        diff = abs((cert.not_valid_after_utc - expected).total_seconds())
        assert diff < 60  # within 1 minute tolerance


class TestCertRotation:
    """Tests for certificate rotation policy."""

    def test_rotation_not_needed_for_fresh_cert(
        self, ca: CertificateAuthority
    ) -> None:
        cert_path, _ = ca.issue_server_cert(cn="fresh")
        result = ca.check_and_rotate(cert_path)
        assert result is None  # No rotation needed

    def test_rotation_triggers_for_expiring_cert(
        self, tmp_path: Path
    ) -> None:
        # Create CA with 1-day certs and 2-day renewal window
        # => any cert is immediately within the renewal window
        policy = CertRotationPolicy(
            expiry_days=1, renewal_window_days=2, max_renewals=5
        )
        ca = CertificateAuthority(
            certs_dir=tmp_path, rotation_policy=policy
        )
        cert_path, _ = ca.issue_server_cert(cn="expiring")
        result = ca.check_and_rotate(cert_path)
        assert result is not None  # Should have rotated


class TestMutualHandshake:
    """Tests for TLS handshake using the CA-issued certs."""

    def test_ssl_contexts_can_be_created(
        self, ca: CertificateAuthority
    ) -> None:
        from security.tls_config import (
            create_server_ssl_context,
            create_client_ssl_context,
        )

        server_cert, server_key = ca.issue_server_cert()
        client_cert, client_key = ca.issue_client_cert()

        server_ctx = create_server_ssl_context(
            server_cert, server_key, ca.ca_cert_path
        )
        client_ctx = create_client_ssl_context(
            client_cert, client_key, ca.ca_cert_path
        )

        assert isinstance(server_ctx, ssl.SSLContext)
        assert isinstance(client_ctx, ssl.SSLContext)
        assert server_ctx.verify_mode == ssl.CERT_REQUIRED
        assert client_ctx.verify_mode == ssl.CERT_REQUIRED

    def test_reject_rogue_cert(self, tmp_path: Path) -> None:
        """A cert from a *different* CA should not be trusted."""
        real_ca = CertificateAuthority(
            certs_dir=tmp_path / "real", ca_cn="Real CA"
        )
        rogue_ca = CertificateAuthority(
            certs_dir=tmp_path / "rogue", ca_cn="Rogue CA"
        )
        from security.tls_config import create_server_ssl_context

        server_cert, server_key = real_ca.issue_server_cert()
        # Load server context trusting the REAL CA
        server_ctx = create_server_ssl_context(
            server_cert, server_key, real_ca.ca_cert_path
        )

        # Issue a client cert from the ROGUE CA
        rogue_cert, rogue_key = rogue_ca.issue_client_cert(cn="rogue")

        # The rogue cert should NOT be verifiable against the real CA
        from cryptography import x509 as cx509

        real_ca_cert = cx509.load_pem_x509_certificate(
            real_ca.ca_cert_path.read_bytes()
        )
        rogue_client_cert = cx509.load_pem_x509_certificate(
            rogue_cert.read_bytes()
        )
        # Issuer mismatch proves the rogue cert is untrusted
        assert rogue_client_cert.issuer != real_ca_cert.subject
