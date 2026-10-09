"""SSL/TLS context factories for mutual TLS (mTLS).

TLS version policy
~~~~~~~~~~~~~~~~~~
Both ``server_ssl_context`` and ``client_ssl_context`` require **TLS 1.3**.
That is one D7 policy for every CLASP channel: the registry's server
(``registry.serve``) and the edge's client already require 1.3, and every
CLASP service runs on Python 3.11 / OpenSSL 3, so there is no legacy peer that
needs 1.2. A context from this module can talk to the registry and the edge
without being loosened or tightened by the caller.
"""
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
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    
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
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    
    context.verify_mode = ssl.CERT_REQUIRED
    
    context.load_verify_locations(cafile=str(ca_cert_path))
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    
    return context
