"""Tests for security.mtls.rotation — certificate expiry and rotation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from security.mtls.ca import create_ca, save_pem
from security.mtls.certs import generate_service_cert
from security.mtls.rotation import CertStatus, check_all_certs, check_expiry, rotate_cert


@pytest.fixture()
def pki(tmp_path: Path) -> dict:
    """Create a CA + one service cert for rotation tests."""
    ca_key, ca_cert = create_ca(common_name="Rotation Test CA", validity_days=365)
    save_pem(ca_key, ca_cert, tmp_path, name="ca")

    svc_key, svc_cert = generate_service_cert(
        ca_key, ca_cert, "test-service",
        sans=["localhost"],
        validity_days=90,
    )
    save_pem(svc_key, svc_cert, tmp_path, name="test-service")

    return {
        "dir": tmp_path,
        "ca_key": ca_key,
        "ca_cert": ca_cert,
        "svc_key": svc_key,
        "svc_cert": svc_cert,
    }


class TestCheckExpiry:
    """check_expiry returns correct certificate status."""

    def test_not_expired(self, pki: dict) -> None:
        # Naming: {name}_cert.pem
        status = check_expiry(pki["dir"] / "test-service_cert.pem")
        assert isinstance(status, CertStatus)
        assert status.expired is False
        assert status.days_remaining > 0

    def test_days_remaining_correct(self, pki: dict) -> None:
        status = check_expiry(pki["dir"] / "test-service_cert.pem")
        # Cert was created with 90-day validity.
        assert 88 <= status.days_remaining <= 91

    def test_needs_renewal_with_high_threshold(self, pki: dict) -> None:
        # 90-day cert with 100-day warn threshold should flag renewal.
        status = check_expiry(pki["dir"] / "test-service_cert.pem", warn_days=100)
        assert status.needs_renewal is True

    def test_no_renewal_with_low_threshold(self, pki: dict) -> None:
        # 90-day cert with 30-day warn threshold should NOT flag renewal.
        status = check_expiry(pki["dir"] / "test-service_cert.pem", warn_days=30)
        assert status.needs_renewal is False

    def test_subject_populated(self, pki: dict) -> None:
        status = check_expiry(pki["dir"] / "test-service_cert.pem")
        assert "test-service" in status.subject


class TestRotateCert:
    """rotate_cert re-issues a cert from the same CA."""

    def test_new_cert_is_valid(self, pki: dict) -> None:
        new_key, new_cert = rotate_cert(
            pki["ca_key"], pki["ca_cert"],
            pki["dir"] / "test-service_cert.pem",
            "test-service",
            validity_days=60,
        )
        # New cert should be signed by the same CA.
        assert new_cert.issuer == pki["ca_cert"].subject

    def test_new_cert_has_fresh_validity(self, pki: dict) -> None:
        _, new_cert = rotate_cert(
            pki["ca_key"], pki["ca_cert"],
            pki["dir"] / "test-service_cert.pem",
            "test-service",
            validity_days=60,
        )
        now = datetime.now(timezone.utc)
        delta = new_cert.not_valid_after_utc - now
        assert 58 <= delta.days <= 61

    def test_new_cert_different_serial(self, pki: dict) -> None:
        _, new_cert = rotate_cert(
            pki["ca_key"], pki["ca_cert"],
            pki["dir"] / "test-service_cert.pem",
            "test-service",
        )
        assert new_cert.serial_number != pki["svc_cert"].serial_number


class TestCheckAllCerts:
    """check_all_certs scans a directory for .pem cert files."""

    def test_finds_certs(self, pki: dict) -> None:
        statuses = check_all_certs(pki["dir"])
        # Should find ca_cert.pem and test-service_cert.pem (key files are *_key.pem, skipped).
        assert len(statuses) >= 2

    def test_all_valid(self, pki: dict) -> None:
        statuses = check_all_certs(pki["dir"])
        for s in statuses:
            assert s.expired is False
