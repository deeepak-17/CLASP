from __future__ import annotations

import datetime
import os
from pathlib import Path

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


def create_ca(
    common_name: str = "CLASP Root CA",
    validity_days: int = 365,
    key_size: int = 4096,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Creates a self-signed Root CA certificate and private key.
    
    Args:
        common_name: The Common Name for the CA.
        validity_days: Number of days the CA certificate is valid.
        key_size: RSA key size in bits.
        
    Returns:
        A tuple of the CA private key and the CA certificate.
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])
    
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=validity_days)
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(private_key.public_key()),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    
    return private_key, cert


def sign_csr(
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    csr: x509.CertificateSigningRequest,
    validity_days: int = 90,
) -> x509.Certificate:
    """Signs a Certificate Signing Request with the given CA.
    
    Args:
        ca_key: The CA's private key.
        ca_cert: The CA's certificate.
        csr: The Certificate Signing Request.
        validity_days: Number of days the signed certificate is valid.
        
    Returns:
        The signed certificate.
    """
    cert_builder = (
        x509.CertificateBuilder()
        .subject_name(csr.subject)
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=validity_days)
        )
    )
    
    for ext in csr.extensions:
        cert_builder = cert_builder.add_extension(ext.value, ext.critical)
        
    cert_builder = cert_builder.add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
        critical=False,
    )
    
    cert = cert_builder.sign(ca_key, hashes.SHA256())
    return cert


def save_pem(
    key: rsa.RSAPrivateKey,
    cert: x509.Certificate,
    directory: Path,
    name: str = "ca",
) -> tuple[Path, Path]:
    """Saves a private key and certificate to PEM files in the given directory.
    
    Args:
        key: The RSA private key.
        cert: The X.509 certificate.
        directory: Directory to save the files.
        name: Base name for the files.
        
    Returns:
        A tuple of the paths to the saved key and certificate.
    """
    directory.mkdir(parents=True, exist_ok=True)
    
    key_path = directory / f"{name}_key.pem"
    cert_path = directory / f"{name}_cert.pem"
    
    key_data = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # Write private key with restrictive permissions (owner-only: 0o600).
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key_data)
    finally:
        os.close(fd)
    
    cert_path.write_bytes(
        cert.public_bytes(serialization.Encoding.PEM)
    )
    
    return key_path, cert_path


def load_pem(
    directory: Path,
    name: str = "ca",
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Loads a private key and certificate from PEM files in the given directory.
    
    Args:
        directory: Directory containing the PEM files.
        name: Base name for the files.
        
    Returns:
        A tuple of the RSA private key and X.509 certificate.
    """
    key_path = directory / f"{name}_key.pem"
    cert_path = directory / f"{name}_cert.pem"
    
    key = serialization.load_pem_private_key(
        key_path.read_bytes(),
        password=None,
    )
    
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError("Key is not an RSA private key")
        
    cert = x509.load_pem_x509_certificate(
        cert_path.read_bytes()
    )
    
    return key, cert
