"""Launch the registry — plain HTTP by default, mutual TLS when certs are set (D7).

    python -m registry.serve

Environment:

    CLASP_REGISTRY_HOST   bind address (default 0.0.0.0)
    CLASP_REGISTRY_PORT   port (default 8004)
    CLASP_TLS_CERT        server certificate (PEM)    \\
    CLASP_TLS_KEY         server private key (PEM)     > all three, or none
    CLASP_TLS_CA          CA that signs client certs  /

With all three set, every connection must present a client certificate signed
by ``CLASP_TLS_CA`` and negotiate TLS 1.3 — the same policy as the security
library's ``create_server_ssl_context``. A partial or unreadable TLS config is
a startup error, never a silent fallback to plain HTTP.
"""
from __future__ import annotations

import logging
import os
import ssl
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import uvicorn

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8004
_TLS_VARS = ("CLASP_TLS_CERT", "CLASP_TLS_KEY", "CLASP_TLS_CA")


class TLSConfigError(ValueError):
    """The launcher's environment is inconsistent; refuse to start."""


@dataclass(frozen=True)
class TLSSettings:
    cert: Path
    key: Path
    ca: Path


def tls_settings_from_env(env: Mapping[str, str]) -> TLSSettings | None:
    present = {v: env[v] for v in _TLS_VARS if env.get(v)}
    if not present:
        return None
    missing = [v for v in _TLS_VARS if v not in present]
    if missing:
        raise TLSConfigError(
            f"mTLS needs all of {', '.join(_TLS_VARS)}; missing {', '.join(missing)}"
        )
    paths = {v: Path(p) for v, p in present.items()}
    for var, path in paths.items():
        if not path.is_file():
            raise TLSConfigError(f"{var}={path} not found")
    return TLSSettings(paths["CLASP_TLS_CERT"], paths["CLASP_TLS_KEY"], paths["CLASP_TLS_CA"])


def _port(env: Mapping[str, str]) -> int:
    raw = env.get("CLASP_REGISTRY_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw)
    except ValueError as e:
        raise TLSConfigError(f"CLASP_REGISTRY_PORT={raw!r} is not an integer") from e
    if not 0 < port < 65536:
        raise TLSConfigError(f"CLASP_REGISTRY_PORT={port} out of range")
    return port


def build_config(env: Mapping[str, str]) -> uvicorn.Config:
    """A loaded uvicorn config; with TLS, hardened to mTLS + TLS 1.3."""
    tls = tls_settings_from_env(env)
    kwargs: dict = {}
    if tls is None:
        logging.getLogger("registry").warning(
            "serving plain HTTP with no client authentication — set CLASP_TLS_CERT, "
            "CLASP_TLS_KEY and CLASP_TLS_CA for mTLS (D7)")
    else:
        kwargs = {
            "ssl_certfile": str(tls.cert),
            "ssl_keyfile": str(tls.key),
            "ssl_ca_certs": str(tls.ca),
            "ssl_cert_reqs": ssl.CERT_REQUIRED,
        }
    config = uvicorn.Config(
        "registry.app:app",
        host=env.get("CLASP_REGISTRY_HOST", DEFAULT_HOST),
        port=_port(env),
        workers=1,  # the write lock in registry.app is per-process
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
        print(f"registry: refusing to start: {e}", file=sys.stderr)
        return 2
    uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
