"""Launch the live panel — HTTPS when a page certificate is set (C1).

    python -m demo_ui.serve

Environment:

    CLASP_DEMO_UI_HOST     bind address (default 0.0.0.0)
    CLASP_DEMO_UI_PORT     port (default 8010)
    CLASP_PAGE_TLS_CERT    the pages' server certificate (PEM)  \\ both, or neither
    CLASP_PAGE_TLS_KEY     its private key (PEM)                /

A browser cannot easily present a client certificate, so the page is served
with a server certificate only and people are identified by their login (U5).
``scripts/make_dev_certs.sh`` issues ``pages/pages.pem`` for this, and
``scripts/run_services.py`` sets the two variables. Without them the panel
serves plain HTTP and says so. A partial or unreadable setting is a startup
error, never a silent fallback to plain HTTP.

The panel's calls *to* the cluster and the registry are mTLS with its own
client certificate (``CLASP_TLS_CLIENT_CERT`` / ``_KEY`` / ``CLASP_TLS_CA``,
see ``demo_ui.server``); that is separate from how its page is served.
"""
from __future__ import annotations

import logging
import os
import ssl
import sys
from collections.abc import Mapping
from pathlib import Path

import uvicorn

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8010
PAGE_TLS_VARS = ("CLASP_PAGE_TLS_CERT", "CLASP_PAGE_TLS_KEY")


class PageTLSError(ValueError):
    """The page's TLS settings are inconsistent; refuse to start."""


def page_tls_from_env(env: Mapping[str, str]) -> tuple[Path, Path] | None:
    """``(cert, key)``, ``None`` for plain HTTP, or :class:`PageTLSError`."""
    present = {v: env[v] for v in PAGE_TLS_VARS if env.get(v)}
    if not present:
        return None
    missing = [v for v in PAGE_TLS_VARS if v not in present]
    if missing:
        raise PageTLSError(f"HTTPS needs both of {', '.join(PAGE_TLS_VARS)}; missing "
                           f"{', '.join(missing)}")
    cert, key = (Path(present[v]) for v in PAGE_TLS_VARS)
    for var, path in zip(PAGE_TLS_VARS, (cert, key)):
        if not path.is_file():
            raise PageTLSError(f"{var}={path} not found")
    return cert, key


def build_config(env: Mapping[str, str]) -> uvicorn.Config:
    tls = page_tls_from_env(env)
    kwargs: dict = {}
    if tls is None:
        logging.getLogger("demo_ui").warning(
            "serving the panel over plain HTTP — set CLASP_PAGE_TLS_CERT and "
            "CLASP_PAGE_TLS_KEY for HTTPS (scripts/make_dev_certs.sh issues pages/pages.pem)")
    else:
        kwargs = {"ssl_certfile": str(tls[0]), "ssl_keyfile": str(tls[1])}
    try:
        port = int(env.get("CLASP_DEMO_UI_PORT", str(DEFAULT_PORT)))
    except ValueError as e:
        raise PageTLSError(f"CLASP_DEMO_UI_PORT={env.get('CLASP_DEMO_UI_PORT')!r} is not an "
                           f"integer") from e
    config = uvicorn.Config(
        "demo_ui.server:app",
        host=env.get("CLASP_DEMO_UI_HOST", DEFAULT_HOST),
        port=port,
        log_level="warning",
        workers=1,  # the round's state (and the login key) are per process
        **kwargs,
    )
    config.load()
    if config.ssl is not None:
        config.ssl.minimum_version = ssl.TLSVersion.TLSv1_2
    return config


def main() -> int:
    try:
        config = build_config(os.environ)
    except (PageTLSError, ssl.SSLError, OSError) as e:
        print(f"demo_ui: refusing to start: {e}", file=sys.stderr)
        return 2
    uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
