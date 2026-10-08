from __future__ import annotations

import ssl
from pathlib import Path


def server_ssl_context(
    cert_path: str | Path,
    key_path: str | Path,
    ca_cert_path: str | Path,
) -> ssl.SSLContext:
    """Creates an SSL context for a server enforcing mutual TLS.
    
    Args:
        cert_path: Path to the server's certificate.
        key_path: Path to the server's private key.
        ca_cert_path: Path to the CA certificate used to verify clients.
        
    Returns:
        An ssl.SSLContext configured for server-side mTLS.
    """
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    
    context.verify_mode = ssl.CERT_REQUIRED
    
    context.load_verify_locations(cafile=str(ca_cert_path))
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    
    return context


def client_ssl_context(
    cert_path: str | Path,
    key_path: str | Path,
    ca_cert_path: str | Path,
) -> ssl.SSLContext:
    """Creates an SSL context for a client enforcing mutual TLS.
    
    Args:
        cert_path: Path to the client's certificate.
        key_path: Path to the client's private key.
        ca_cert_path: Path to the CA certificate used to verify the server.
        
    Returns:
        An ssl.SSLContext configured for client-side mTLS.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    
    context.verify_mode = ssl.CERT_REQUIRED
    
    context.load_verify_locations(cafile=str(ca_cert_path))
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    
    return context
