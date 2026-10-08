"""Tests for security.mtls.ca — Local Certificate Authority."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa

from security.mtls.ca import create_ca, load_pem, save_pem
from security.mtls.certs import generate_service_cert


class TestCreateCA:
    """CA creation produces a valid self-signed root certificate."""

    def test_creates_rsa_key(self) -> None:
        ca_key, ca_cert = create_ca()
        assert isinstance(ca_key, rsa.RSAPrivateKey)
        assert ca_key.key_size == 4096

    def test_self_signed(self) -> None:
        ca_key, ca_cert = create_ca()
        # Issuer == Subject for a self-signed cert.
        assert ca_cert.issuer == ca_cert.subject

    def test_is_ca(self) -> None:
        _, ca_cert = create_ca()
        bc = ca_cert.extensions.get_extension_for_class(x509.BasicConstraints)
        assert bc.value.ca is True

    def test_custom_common_name(self) -> None:
        _, ca_cert = create_ca(common_name="Test CA")
        cn = ca_cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
        assert cn[0].value == "Test CA"

    def test_validity_period(self) -> None:
        _, ca_cert = create_ca(validity_days=30)
        now = datetime.now(timezone.utc)
        delta = ca_cert.not_valid_after_utc - now
        # Should be between 29 and 31 days (accounting for execution time).
        assert 28 <= delta.days <= 31

    def test_default_validity_one_year(self) -> None:
        _, ca_cert = create_ca()
        now = datetime.now(timezone.utc)
        delta = ca_cert.not_valid_after_utc - now
        assert 363 <= delta.days <= 366


class TestSavePemLoadPem:
    """PEM serialization round-trip."""

    def test_round_trip(self) -> None:
        ca_key, ca_cert = create_ca(common_name="Round-Trip CA")
        with tempfile.TemporaryDirectory() as tmp:
            key_path, cert_path = save_pem(ca_key, ca_cert, Path(tmp), name="test-ca")
            assert key_path.exists()
            assert cert_path.exists()
            # Naming convention: {name}_key.pem, {name}_cert.pem
            assert key_path.name == "test-ca_key.pem"
            assert cert_path.name == "test-ca_cert.pem"

            loaded_key, loaded_cert = load_pem(Path(tmp), name="test-ca")
            # The loaded cert should have the same subject.
            assert loaded_cert.subject == ca_cert.subject
            # The loaded key should have the same key size.
            assert loaded_key.key_size == ca_key.key_size


class TestSignCSR:
    """CSR signing produces a cert chained to the CA."""

    def test_signed_cert_issuer_matches_ca(self) -> None:
        ca_key, ca_cert = create_ca()
        # Generate a service cert (which internally creates a CSR).
        svc_key, svc_cert = generate_service_cert(ca_key, ca_cert, "test-service")
        assert svc_cert.issuer == ca_cert.subject

    def test_signed_cert_validity(self) -> None:
        ca_key, ca_cert = create_ca()
        _, svc_cert = generate_service_cert(ca_key, ca_cert, "test-service",
                                            validity_days=30)
        now = datetime.now(timezone.utc)
        delta = svc_cert.not_valid_after_utc - now
        assert 28 <= delta.days <= 31
