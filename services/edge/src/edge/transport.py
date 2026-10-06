"""Edge HTTP transport — plain, or mutually authenticated with P3's mTLS (D7).

D7 puts mTLS on ALL service channels, the registry included. The TLS policy —
TLS 1.3 minimum, CERT_REQUIRED, the CLASP root CA — belongs to P3 and lives in
``security.create_client_ssl_context``; the edge does not build an SSL context
of its own. What the edge owns is getting that context underneath the HTTP
client it already uses for both of its outbound seams:

    seam A    Edge -> Cluster   POST /uploads          (:func:`post_upload`)
    seam C1/2 Edge -> Registry  GET .../file, POST .../promote
              (``edge.registry_client.RegistryClient(session=...)``)

``requests`` takes no ``SSLContext`` directly, so a small transport adapter
hands P3's context to urllib3's pool manager. ``session.verify`` is pointed at
the same CA file, so requests never swaps in the public certifi bundle.

``security`` is imported lazily: a plain-HTTP session needs nothing from it,
and the edge stays importable before the security library is integrated.

    from edge.transport import mtls_session
    from edge.registry_client import RegistryClient

    session = mtls_session("certs/client.pem", "certs/client-key.pem", "certs/ca.pem")
    rc = RegistryClient("https://localhost:8004", session=session)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter

DEFAULT_UPLOAD_TIMEOUT = 600.0


class SSLContextAdapter(HTTPAdapter):
    """An ``HTTPAdapter`` whose connection pools all use one given SSLContext."""

    def __init__(self, ssl_context, **kwargs) -> None:
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().proxy_manager_for(*args, **kwargs)


def mtls_session(client_cert: Path | str, client_key: Path | str, ca_cert: Path | str,
                 server_hostname: str = "localhost") -> requests.Session:
    """A ``requests.Session`` that presents the edge's certificate and accepts
    only servers signed by the CLASP CA, built on P3's client SSL context."""
    try:
        from security import create_client_ssl_context
    except ImportError as exc:
        raise RuntimeError(
            "mTLS needs P3's security library (security.create_client_ssl_context), "
            "which the installed `security` package does not provide — integrate the "
            "security branch first.") from exc
    context = create_client_ssl_context(client_cert, client_key, ca_cert,
                                        server_hostname=server_hostname)
    session = requests.Session()
    session.verify = str(ca_cert)
    session.mount("https://", SSLContextAdapter(context))
    return session


def session(mtls: Optional[Dict[str, str]] = None) -> requests.Session:
    """Plain session, or an mTLS one from a dict with ``client_cert``,
    ``client_key``, ``ca_cert`` (and optionally ``server_hostname``)."""
    return mtls_session(**mtls) if mtls else requests.Session()


def post_upload(http: requests.Session, cluster_url: str, payload: Dict,
                timeout: float = DEFAULT_UPLOAD_TIMEOUT) -> Dict:
    """Seam A: ``POST /uploads`` one client adapter (``edge.wire.upload_payload``).

    Raises ``RuntimeError`` with the cluster's own message on anything but 201,
    so a rejected upload stops the round instead of being aggregated around.
    """
    r = http.post(f"{cluster_url.rstrip('/')}/uploads", json=payload, timeout=timeout)
    if r.status_code != 201:
        raise RuntimeError(f"upload of {payload.get('client_id')!r} rejected by the cluster: "
                           f"{r.status_code} {r.text[:400]}")
    return r.json()
