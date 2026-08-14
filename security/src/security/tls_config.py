"""TLS 1.3 SSL context builders for mutual-TLS (mTLS).

Provides ready-to-use ``ssl.SSLContext`` objects for both the server side
(cluster aggregator) and client side (edge uploader).  Both enforce
``CERT_REQUIRED`` so that neither side can connect without a valid cert
signed by the CLASP root CA.

References
----------
- NIST SP 1800-35 "Implementing a Zero Trust Architecture" (June 2025)
- RFC 8446 — TLS 1.3
"""
from __future__ import annotations

import ssl
from pathlib import Path


def create_server_ssl_context(
    server_cert: str | Path,
    server_key: str | Path,
    ca_cert: str | Path,
) -> ssl.SSLContext:
    """Build an SSL context for the *server* (cluster) side of mTLS.

    - Loads the server's own certificate and private key.
    - Loads the CA bundle used to verify *client* certificates.
    - Enforces ``CERT_REQUIRED`` — clients without a valid cert are rejected.
    - Sets TLS 1.3 as the minimum protocol version.

    Parameters
    ----------
    server_cert : path to the server PEM certificate
    server_key  : path to the server PEM private key
    ca_cert     : path to the CA PEM certificate (for client verification)
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # TLS 1.3 minimum — eliminates 2-RTT handshakes from TLS 1.2
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3

    # Server's own identity
    ctx.load_cert_chain(certfile=str(server_cert), keyfile=str(server_key))

    # CA bundle to verify client certs
    ctx.load_verify_locations(cafile=str(ca_cert))

    # Mutual authentication — reject any client without a valid cert
    ctx.verify_mode = ssl.CERT_REQUIRED

    # Hardened defaults
    ctx.check_hostname = False  # server doesn't check its own hostname

    return ctx


def create_client_ssl_context(
    client_cert: str | Path,
    client_key: str | Path,
    ca_cert: str | Path,
    server_hostname: str = "localhost",
) -> ssl.SSLContext:
    """Build an SSL context for the *client* (edge) side of mTLS.

    - Loads the client's own certificate and private key.
    - Loads the CA bundle used to verify the *server* certificate.
    - Enforces ``CERT_REQUIRED`` and hostname checking.
    - Sets TLS 1.3 as the minimum protocol version.

    Parameters
    ----------
    client_cert     : path to the client PEM certificate
    client_key      : path to the client PEM private key
    ca_cert         : path to the CA PEM certificate (for server verification)
    server_hostname : expected hostname in the server certificate SAN
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # TLS 1.3 minimum
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3

    # Client's own identity (presented to the server for mutual auth)
    ctx.load_cert_chain(certfile=str(client_cert), keyfile=str(client_key))

    # CA bundle to verify the server's cert
    ctx.load_verify_locations(cafile=str(ca_cert))

    # Verify server cert
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True

    return ctx
