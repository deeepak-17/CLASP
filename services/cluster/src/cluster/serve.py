"""Launch the cluster service — plain HTTP by default, mutual TLS when certs are set (D7).

    python -m cluster.serve

Environment (the same names ``registry.serve`` reads, so one overlay can set both):

    CLASP_CLUSTER_HOST    bind address (default 0.0.0.0)
    CLASP_CLUSTER_PORT    port (default 8002)
    CLASP_TLS_CERT        server certificate (PEM)    \\
    CLASP_TLS_KEY         server private key (PEM)     > all three, or none
    CLASP_TLS_CA          CA that signs client certs  /
    CLASP_CLUSTER_BIND_IDENTITY   under TLS: "1" (default) refuses an upload whose
                          ``client_id`` is not the CN of the caller's certificate

With the three TLS files set, every connection must present a client
certificate signed by ``CLASP_TLS_CA`` and negotiate TLS 1.3 (the registry's
policy). A partial or unreadable TLS config is a startup error, never a silent
fallback to plain HTTP.

Binding ``client_id`` to the certificate
----------------------------------------
uvicorn verifies the client certificate during the handshake but does not put
it in the ASGI scope (the limit ``cluster.tls`` documents).
``security.mtls.peer.PeerCertH11Protocol`` closes that gap: when a TLS
connection opens it reads the verified peer certificate's CN off the socket and
adds it to the scope of every request on that connection under ``SCOPE_KEY``.
The identity hook ``server.configure_identity`` then reads it from there, so
the upload handlers' existing 401/403 checks apply to a real, verified
identity. The helpers live in the security library (the registry uses the same
ones) and are re-exported here under their old names.
"""
from __future__ import annotations

import logging
import os
import ssl
import sys
from collections.abc import Mapping
from typing import Any

import uvicorn
from security.mtls.peer import SCOPE_KEY, PeerCertH11Protocol, peer_identity

from cluster.tls import MTLSFiles, TLSConfigError, uvicorn_ssl_kwargs

__all__ = ["SCOPE_KEY", "PeerCertH11Protocol", "build_config", "main", "peer_identity",
           "tls_files_from_env"]

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8002
TLS_VARS = ("CLASP_TLS_CERT", "CLASP_TLS_KEY", "CLASP_TLS_CA")


def tls_files_from_env(env: Mapping[str, str]) -> MTLSFiles | None:
    """The server's mTLS files, ``None`` for plain HTTP, or ``TLSConfigError``."""
    present = {v: env[v] for v in TLS_VARS if env.get(v)}
    if not present:
        return None
    missing = [v for v in TLS_VARS if v not in present]
    if missing:
        raise TLSConfigError(f"mTLS needs all of {', '.join(TLS_VARS)}; missing {', '.join(missing)}")
    return MTLSFiles(
        ca_cert=present["CLASP_TLS_CA"], cert=present["CLASP_TLS_CERT"], key=present["CLASP_TLS_KEY"]
    ).checked()


def _port(env: Mapping[str, str]) -> int:
    raw = env.get("CLASP_CLUSTER_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw)
    except ValueError as e:
        raise TLSConfigError(f"CLASP_CLUSTER_PORT={raw!r} is not an integer") from e
    if not 0 < port < 65536:
        raise TLSConfigError(f"CLASP_CLUSTER_PORT={port} out of range")
    return port


def build_config(env: Mapping[str, str]) -> uvicorn.Config:
    """A loaded uvicorn config; with TLS, hardened to mTLS + TLS 1.3 and with
    uploads bound to the client certificate (unless switched off)."""
    from cluster import server

    files = tls_files_from_env(env)
    kwargs: dict[str, Any] = {}
    if files is None:
        logging.getLogger("cluster").warning(
            "serving plain HTTP with no client authentication — set CLASP_TLS_CERT, "
            "CLASP_TLS_KEY and CLASP_TLS_CA for mTLS (D7)")
        server.configure_identity(None)
    else:
        kwargs = {**uvicorn_ssl_kwargs(files), "http": PeerCertH11Protocol}
        bind = env.get("CLASP_CLUSTER_BIND_IDENTITY", "1").strip().lower() not in ("0", "false", "no")
        server.configure_identity(peer_identity if bind else None, require=bind)
    config = uvicorn.Config(
        server.app,
        host=env.get("CLASP_CLUSTER_HOST", DEFAULT_HOST),
        port=_port(env),
        workers=1,  # cluster state and its locks are per process (see _ClusterState)
        **kwargs,
    )
    config.load()
    if config.ssl is not None:
        config.ssl.minimum_version = ssl.TLSVersion.TLSv1_3
        config.ssl.verify_mode = ssl.CERT_REQUIRED
    return config


def main() -> int:
    try:
        config = build_config(os.environ)
    except (TLSConfigError, ssl.SSLError, OSError) as e:
        print(f"cluster: refusing to start: {e}", file=sys.stderr)
        return 2
    uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
