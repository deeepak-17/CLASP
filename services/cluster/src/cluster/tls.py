"""P2-side mTLS wiring helpers (G1) — configuration only, no certificates issued.

Cluster is a TLS *consumer*: it needs a CA certificate, its own server
certificate/key, and (on clients) a client certificate/key. Issuing those,
rotating them and mapping a certificate to a client identity is P3 (Security),
which has no code in the repository yet. This module therefore only turns file
paths into the settings uvicorn / Flower / ``ssl`` expect, so that the day P3
hands over files the Cluster side is one call away — and so the Cluster side
can be exercised today against an ephemeral CA created by the tests.

Known limits (kept here so nobody over-reads this module):
  * uvicorn verifies the client certificate during the handshake but does not
    expose the peer certificate to the ASGI app. Binding a verified certificate
    to ``client_id`` therefore needs either a terminating proxy that relays the
    verified subject (read through ``server.configure_identity``) or a different
    server. ``common_name_from_peercert`` is for the latter case.
  * ``flower_certificates`` only builds the tuple Flower's ``start_server`` /
    ``start_client`` accept; a Flower gRPC mTLS round has not been run here.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TLSConfigError(ValueError):
    """A required certificate/key file is missing or unreadable."""


@dataclass(frozen=True)
class MTLSFiles:
    """PEM files for one side of a mutually authenticated connection."""

    ca_cert: Path  # CA that signed the *peer's* certificate
    cert: Path  # this side's certificate chain
    key: Path  # this side's private key

    def checked(self) -> MTLSFiles:
        for label, path in (("ca_cert", self.ca_cert), ("cert", self.cert), ("key", self.key)):
            if not Path(path).is_file():
                raise TLSConfigError(f"mTLS {label} file not found: {path}")
        return self


def uvicorn_ssl_kwargs(files: MTLSFiles, *, require_client_cert: bool = True) -> dict[str, Any]:
    """Keyword arguments for ``uvicorn.run`` / ``uvicorn.Config``.

    With ``require_client_cert`` (the default and the only mTLS setting) a
    client that presents no certificate, or one not signed by ``ca_cert``, fails
    the handshake.
    """
    files.checked()
    return {
        "ssl_keyfile": str(files.key),
        "ssl_certfile": str(files.cert),
        "ssl_ca_certs": str(files.ca_cert),
        "ssl_cert_reqs": ssl.CERT_REQUIRED if require_client_cert else ssl.CERT_OPTIONAL,
    }


def client_ssl_context(files: MTLSFiles) -> ssl.SSLContext:
    """``ssl`` context for an HTTP client that must present ``files.cert`` and
    trust only ``files.ca_cert`` (hostname checking stays on)."""
    files.checked()
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(files.ca_cert))
    ctx.load_cert_chain(certfile=str(files.cert), keyfile=str(files.key))
    return ctx


def flower_certificates(files: MTLSFiles) -> tuple[bytes, bytes, bytes]:
    """``(ca, certificate, private_key)`` PEM bytes in the order Flower's
    ``certificates=`` argument takes them."""
    files.checked()
    return (files.ca_cert.read_bytes(), files.cert.read_bytes(), files.key.read_bytes())


def common_name_from_peercert(peercert: dict[str, Any] | None) -> str | None:
    """Subject CN from the dict ``ssl.SSLSocket.getpeercert()`` returns."""
    if not peercert:
        return None
    for rdn in peercert.get("subject", ()):
        for key, value in rdn:
            if key == "commonName":
                return str(value)
    return None
