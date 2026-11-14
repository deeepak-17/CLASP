"""Container healthcheck that works with and without mTLS.

    python -m cluster.healthcheck      # exit 0 iff GET /healthz -> 200

Reads the same environment as ``cluster.serve``. Under mTLS it presents
CLASP_HEALTHCHECK_CERT / CLASP_HEALTHCHECK_KEY (falling back to the cluster's
own cert and key, which the CLASP CA signs) and verifies the server against
``CLASP_TLS_CA``; the hostname is not checked on this loopback probe.
"""
from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from collections.abc import Mapping

from cluster.serve import DEFAULT_PORT, tls_files_from_env

TIMEOUT_SECONDS = 3


def healthz_url_and_context(env: Mapping[str, str]) -> tuple[str, ssl.SSLContext | None]:
    port = env.get("CLASP_CLUSTER_PORT", str(DEFAULT_PORT))
    files = tls_files_from_env(env)
    if files is None:
        return f"http://127.0.0.1:{port}/healthz", None
    ctx = ssl.create_default_context(cafile=str(files.ca_cert))
    ctx.check_hostname = False  # loopback probe; the chain is still verified
    ctx.load_cert_chain(env.get("CLASP_HEALTHCHECK_CERT", str(files.cert)),
                        env.get("CLASP_HEALTHCHECK_KEY", str(files.key)))
    return f"https://127.0.0.1:{port}/healthz", ctx


def check(env: Mapping[str, str]) -> bool:
    try:
        url, ctx = healthz_url_and_context(env)
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS, context=ctx) as resp:
            return resp.status == 200
    except (OSError, ValueError) as e:
        print(f"cluster healthcheck failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    sys.exit(0 if check(os.environ) else 1)
