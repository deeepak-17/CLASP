from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .certs import generate_service_cert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CertStatus:
    path: Path
    subject: str
    not_after: datetime.datetime
    days_remaining: int
    needs_renewal: bool
    expired: bool


def check_expiry(
    cert_path: str | Path,
    warn_days: int = 30,
) -> CertStatus:
    """Checks the expiry status of a given PEM certificate.
    
    Args:
        cert_path: Path to the PEM-encoded certificate.
        warn_days: Threshold of days remaining before warning.
        
    Returns:
        CertStatus dataclass with the expiry information.
    """
    cert_path_obj = Path(cert_path)
    cert_data = cert_path_obj.read_bytes()
    cert = x509.load_pem_x509_certificate(cert_data)
    
    not_after = cert.not_valid_after_utc
    now = datetime.datetime.now(datetime.timezone.utc)
    
    days_remaining = (not_after - now).days
    expired = days_remaining < 0
    needs_renewal = days_remaining <= warn_days
    
    try:
        subject_name = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    except IndexError:
        subject_name = str(cert.subject)
        
    if isinstance(subject_name, bytes):
        subject_name = subject_name.decode("utf-8")
        
    return CertStatus(
        path=cert_path_obj,
        subject=subject_name,
        not_after=not_after,
        days_remaining=days_remaining,
        needs_renewal=needs_renewal,
        expired=expired,
    )


def rotate_cert(
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    old_cert_path: str | Path,
    service_name: str,
    validity_days: int = 90,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Rotates a certificate by generating a new one with the same SANs.
    
    Args:
        ca_key: CA private key.
        ca_cert: CA certificate.
        old_cert_path: Path to the old certificate to extract extensions from.
        service_name: The Common Name for the new certificate.
        validity_days: Validity in days for the new certificate.
        
    Returns:
        A tuple of the new RSA private key and X.509 certificate.
    """
    old_cert_data = Path(old_cert_path).read_bytes()
    old_cert = x509.load_pem_x509_certificate(old_cert_data)
    
    sans = []
    try:
        san_ext = old_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        for name in san_ext.value.get_values_for_type(x509.DNSName):
            if name not in ("localhost", service_name):
                sans.append(name)
    except x509.ExtensionNotFound:
        pass
        
    return generate_service_cert(
        ca_key=ca_key,
        ca_cert=ca_cert,
        service_name=service_name,
        sans=sans,
        validity_days=validity_days,
    )


def check_all_certs(
    certs_dir: str | Path,
    warn_days: int = 30,
) -> list[CertStatus]:
    """Checks the expiry status of all certificates in a directory.
    
    Args:
        certs_dir: Directory containing *.pem files.
        warn_days: Threshold of days remaining before warning.
        
    Returns:
        List of CertStatus objects for each found certificate (that ends in _cert.pem or is a cert).
    """
    certs_dir_obj = Path(certs_dir)
    statuses = []
    
    for path in certs_dir_obj.glob("*.pem"):
        if path.name.endswith("_key.pem"):
            continue
            
        try:
            status = check_expiry(path, warn_days=warn_days)
            statuses.append(status)
        except (ValueError, OSError) as exc:
            # Skip files that aren't valid certs
            logger.debug("Skipping %s: %s", path, exc)
            continue
            
    return statuses
