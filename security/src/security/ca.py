"""CLASP local Certificate Authority for mTLS.

Generates a self-signed root CA (ECDSA P-256) and issues server/client
certificates for mutual-TLS between Edge and Cluster services.

References
----------
- NIST SP 1800-35 "Implementing a Zero Trust Architecture" (June 2025)
- RFC 8446 — TLS 1.3
- NIST SP 800-207 — Zero Trust Architecture
"""
from __future__ import annotations

import datetime
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CertRotationPolicy:
    """Policy for automatic certificate rotation."""
    expiry_days: int = 7           # certificate validity period
    renewal_window_days: int = 2   # renew when this many days remain
    max_renewals: int = 52         # cap on consecutive renewals


DEFAULT_CERTS_DIR = Path(__file__).resolve().parent.parent.parent / "certs"


# ---------------------------------------------------------------------------
# Certificate Authority
# ---------------------------------------------------------------------------

class CertificateAuthority:
    """A lightweight, file-backed Certificate Authority using ECDSA P-256.

    Designed for CLASP's on-premise deployment where a full PKI
    (HashiCorp Vault, step-ca) would be overkill.  All keys are stored as
    PEM files in *certs_dir*.  The CA root is self-signed.
    """

    def __init__(
        self,
        certs_dir: Path = DEFAULT_CERTS_DIR,
        ca_cn: str = "CLASP Root CA",
        rotation_policy: CertRotationPolicy | None = None,
    ) -> None:
        self.certs_dir = Path(certs_dir)
        self.certs_dir.mkdir(parents=True, exist_ok=True)
        self.ca_cn = ca_cn
        self.rotation_policy = rotation_policy or CertRotationPolicy()

        self._ca_key_path = self.certs_dir / "ca-key.pem"
        self._ca_cert_path = self.certs_dir / "ca.pem"

        # Load or generate on first use
        if self._ca_cert_path.exists() and self._ca_key_path.exists():
            self._ca_key = self._load_private_key(self._ca_key_path)
            self._ca_cert = self._load_certificate(self._ca_cert_path)
        else:
            self._ca_key, self._ca_cert = self._generate_ca()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ca_cert_path(self) -> Path:
        return self._ca_cert_path

    @property
    def ca_cert(self) -> x509.Certificate:
        return self._ca_cert

    def issue_server_cert(
        self,
        cn: str = "clasp-cluster",
        san_dns: Sequence[str] = ("localhost",),
        san_ips: Sequence[str] = ("127.0.0.1",),
        expiry_days: int | None = None,
    ) -> tuple[Path, Path]:
        """Issue a server certificate signed by the CA.

        Returns (cert_path, key_path).
        """
        return self._issue_cert(
            cn=cn,
            san_dns=san_dns,
            san_ips=san_ips,
            is_server=True,
            expiry_days=expiry_days or self.rotation_policy.expiry_days,
            filename_prefix="server",
        )

    def issue_client_cert(
        self,
        cn: str = "clasp-edge-0",
        expiry_days: int | None = None,
    ) -> tuple[Path, Path]:
        """Issue a client certificate signed by the CA.

        Returns (cert_path, key_path).
        """
        return self._issue_cert(
            cn=cn,
            san_dns=(),
            san_ips=(),
            is_server=False,
            expiry_days=expiry_days or self.rotation_policy.expiry_days,
            filename_prefix="client",
        )

    def check_and_rotate(self, cert_path: Path) -> Path | None:
        """Re-issue *cert_path* if it is within the renewal window.

        Returns the new cert path if rotated, else ``None``.
        """
        cert = self._load_certificate(cert_path)
        now = datetime.datetime.now(datetime.timezone.utc)
        remaining = cert.not_valid_after_utc - now
        window = datetime.timedelta(days=self.rotation_policy.renewal_window_days)

        if remaining <= window:
            cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            # Determine if server or client from EKU
            try:
                eku = cert.extensions.get_extension_for_class(
                    x509.ExtendedKeyUsage
                ).value
                is_server = x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku
            except x509.ExtensionNotFound:
                is_server = False

            prefix = "server" if is_server else "client"
            new_cert, _ = self._issue_cert(
                cn=cn,
                san_dns=(),
                san_ips=(),
                is_server=is_server,
                expiry_days=self.rotation_policy.expiry_days,
                filename_prefix=prefix,
            )
            return new_cert
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_ca(self) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
        """Generate a self-signed ECDSA P-256 root CA."""
        key = ec.generate_private_key(ec.SECP256R1())
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.ca_cn),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CLASP"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365 * 5))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(key, hashes.SHA256())
        )
        self._save_private_key(key, self._ca_key_path)
        self._save_certificate(cert, self._ca_cert_path)
        return key, cert

    def _issue_cert(
        self,
        cn: str,
        san_dns: Sequence[str],
        san_ips: Sequence[str],
        is_server: bool,
        expiry_days: int,
        filename_prefix: str,
    ) -> tuple[Path, Path]:
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CLASP"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)

        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=expiry_days))
        )

        # Extended Key Usage
        if is_server:
            eku = x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ])
        else:
            eku = x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ])
        builder = builder.add_extension(eku, critical=False)

        # Subject Alternative Names (server certs only)
        san_entries: list[x509.GeneralName] = []
        for dns in san_dns:
            san_entries.append(x509.DNSName(dns))
        for ip_str in san_ips:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(ip_str)))
        if san_entries:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_entries), critical=False
            )

        cert = builder.sign(self._ca_key, hashes.SHA256())

        cert_path = self.certs_dir / f"{filename_prefix}.pem"
        key_path = self.certs_dir / f"{filename_prefix}-key.pem"
        self._save_certificate(cert, cert_path)
        self._save_private_key(key, key_path)
        return cert_path, key_path

    # ------------------------------------------------------------------
    # PEM I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _save_private_key(key: ec.EllipticCurvePrivateKey, path: Path) -> None:
        path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    @staticmethod
    def _load_private_key(path: Path) -> ec.EllipticCurvePrivateKey:
        return serialization.load_pem_private_key(path.read_bytes(), password=None)  # type: ignore[return-value]

    @staticmethod
    def _save_certificate(cert: x509.Certificate, path: Path) -> None:
        path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    @staticmethod
    def _load_certificate(path: Path) -> x509.Certificate:
        return x509.load_pem_x509_certificate(path.read_bytes())
