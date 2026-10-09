"""SSL/TLS context factories for mutual TLS (mTLS).

TLS version policy
~~~~~~~~~~~~~~~~~~
Both ``server_ssl_context`` and ``client_ssl_context`` set the minimum TLS
version to **TLS 1.2** (not 1.3).  This is a deliberate choice:

* TLS 1.2 is still considered secure and is required by NIST SP 800-52r2.
* Many production load-balancers and embedded devices do not yet support
  TLS 1.3; enforcing 1.3-only would break interoperability.
* If TLS 1.3-only is desired (e.g. for the edge↔cluster channel), callers
  can override ``context.minimum_version = ssl.TLSVersion.TLSv1_3`` after
  obtaining the context.

The previous security branch used TLS 1.3 as the minimum.  This was relaxed
to 1.2 for broader compatibility; see the PR discussion for details.
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
