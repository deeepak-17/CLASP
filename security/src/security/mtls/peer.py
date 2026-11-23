"""The verified peer certificate name, for services run by uvicorn.

uvicorn verifies the client certificate during the TLS handshake but does not
put it in the ASGI scope. :class:`PeerCertH11Protocol` closes that gap: when a
TLS connection opens it reads the verified peer certificate's CN off the
socket and adds it to the scope of every request on that connection under
``SCOPE_KEY``. :func:`peer_identity` reads it back from a request.

The cluster and the registry both serve with this protocol under mTLS, so the
name a route sees is always one the handshake verified against the CLASP CA.
Without TLS the name is ``None``.
"""
from __future__ import annotations

from typing import Any

from uvicorn.protocols.http.h11_impl import H11Protocol

#: Where :class:`PeerCertH11Protocol` puts the verified client CN in the ASGI scope.
SCOPE_KEY = "clasp.peer_common_name"


def common_name_from_peercert(peercert: dict[str, Any] | None) -> str | None:
    """Subject CN from the dict ``ssl.SSLSocket.getpeercert()`` returns."""
    if not peercert:
        return None
    for rdn in peercert.get("subject", ()):
        for key, value in rdn:
            if key == "commonName":
                return str(value)
    return None


class PeerCertH11Protocol(H11Protocol):
    """uvicorn's h11 protocol, plus the verified client CN in each request's scope."""

    def connection_made(self, transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        ssl_object = transport.get_extra_info("ssl_object")
        peer = common_name_from_peercert(ssl_object.getpeercert()) if ssl_object else None
        inner = self.app

        async def app(scope, receive, send):
            scope[SCOPE_KEY] = peer
            await inner(scope, receive, send)

        self.app = app


def peer_identity(request) -> str | None:
    """The CN :class:`PeerCertH11Protocol` set for this request's connection, or ``None``."""
    return request.scope.get(SCOPE_KEY)
