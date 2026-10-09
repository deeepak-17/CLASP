from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import cryptography.exceptions
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from .ca import create_ca, save_pem


def generate_service_cert(
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    service_name: str,
    sans: list[str] | None = None,
    validity_days: int = 90,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Generates a service certificate signed by the given CA.
    
    Args:
        ca_key: CA private key.
        ca_cert: CA certificate.
        service_name: The Common Name for the service.
        sans: Optional Subject Alternative Names (DNS names).
        validity_days: Certificate validity in days.
        
    Returns:
        Tuple of (service_private_key, service_certificate).
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, service_name),
    ])
    
    dns_names = [x509.DNSName("localhost"), x509.DNSName(service_name)]
    if sans:
        for san in sans:
            dns_names.append(x509.DNSName(san))
            
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=validity_days)
        )
        .add_extension(
            x509.SubjectAlternativeName(dns_names),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=True,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    
    return private_key, cert


def generate_client_cert(
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    client_id: str,
    validity_days: int = 90,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Generates a client certificate signed by the given CA.
    
    Args:
        ca_key: CA private key.
        ca_cert: CA certificate.
        client_id: The Common Name for the client.
        validity_days: Certificate validity in days.
        
    Returns:
        Tuple of (client_private_key, client_certificate).
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, client_id),
    ])
    
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=validity_days)
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=True,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    
    return private_key, cert


def verify_cert(
    cert: x509.Certificate,
    ca_cert: x509.Certificate,
) -> bool:
    """Verifies that a certificate was signed by the given CA.
    
    Args:
        cert: The certificate to verify.
        ca_cert: The CA certificate.
        
    Returns:
        True if the signature is valid, False otherwise.
    """
    ca_public_key = ca_cert.public_key()
    
    if not isinstance(ca_public_key, rsa.RSAPublicKey):
        return False

    try:
        ca_public_key.verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            PKCS1v15(),
            cert.signature_hash_algorithm,
        )
        return True
    except (ValueError, TypeError, cryptography.exceptions.InvalidSignature):
        return False


def setup_all_certs(
    output_dir: Path,
    services: tuple[str, ...] = ("cluster", "registry", "evaluation"),
    clients: tuple[str, ...] = ("edge-0", "edge-1"),
    validity_days: int = 90,
) -> dict[str, Path]:
    """Generates a CA and certificates for all services and clients.
    
    Args:
        output_dir: The directory to save the certificates in.
        services: List of service names.
        clients: List of client IDs.
        validity_days: Validity in days for all generated certificates.
        
    Returns:
        A dictionary mapping the name (e.g., 'ca', 'cluster') to the certificate path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ca_key, ca_cert = create_ca(validity_days=validity_days * 2)
    _ca_key_path, ca_cert_path = save_pem(ca_key, ca_cert, output_dir, name="ca")
    
    paths = {"ca": ca_cert_path}
    
    for service in services:
        key, cert = generate_service_cert(ca_key, ca_cert, service, validity_days=validity_days)
        _, cert_path = save_pem(key, cert, output_dir, name=service)
        paths[service] = cert_path
        
    for client in clients:
        key, cert = generate_client_cert(ca_key, ca_cert, client, validity_days=validity_days)
        _, cert_path = save_pem(key, cert, output_dir, name=client)
        paths[client] = cert_path
        
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mTLS certificates.")
    parser.add_argument("--setup-all", action="store_true", help="Generate all certificates.")
    parser.add_argument("--output-dir", type=str, default="certs", help="Output directory for certificates.")
    
    args = parser.parse_args()
    
    if args.setup_all:
        paths = setup_all_certs(Path(args.output_dir))
        for name, path in paths.items():
            print(f"Generated {name} cert at {path}")
