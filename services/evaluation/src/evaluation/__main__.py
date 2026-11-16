"""``python -m evaluation`` — the evaluation container's entrypoint.

The P5 harness (HumanEval/MBPP Pass@k, in-project eval, guard anchors, round
feed) runs as one-shot CLIs from the repository's ``scripts/``; this package
ships the metric every service imports (``evaluation.completion``). So the
container reports what it provides and, when ``CLASP_REGISTRY_URL`` is set,
whether the State Registry is reachable — then exits. Exit status is 1 only
when a registry URL was given and could not be reached.

The compose stack serves the registry over mutual TLS: for an ``https`` URL the
check presents ``CLASP_TLS_CLIENT_CERT`` / ``CLASP_TLS_CLIENT_KEY`` and trusts
only ``CLASP_TLS_CA``.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request

from evaluation import __version__


TLS_VARS = ("CLASP_TLS_CLIENT_CERT", "CLASP_TLS_CLIENT_KEY", "CLASP_TLS_CA")


def tls_context(url: str, env, make=ssl.create_default_context):
    """The client context for an ``https`` registry (mTLS, TLS 1.3), else ``None``."""
    if not url.startswith("https"):
        return None
    missing = [v for v in TLS_VARS if not env.get(v)]
    if missing:
        raise ValueError(f"{url} is https: set {', '.join(missing)}")
    ctx = make(cafile=env["CLASP_TLS_CA"])
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(env["CLASP_TLS_CLIENT_CERT"], env["CLASP_TLS_CLIENT_KEY"])
    return ctx


def main() -> int:
    print(f"clasp-evaluation {__version__}: evaluation.completion (D5 in-project metric)")
    url = os.environ.get("CLASP_REGISTRY_URL")
    if not url:
        print("CLASP_REGISTRY_URL not set; skipping registry check")
        return 0
    try:
        context = tls_context(url, os.environ)
        with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=5,  # noqa: S310
                                    context=context) as resp:
            print(f"registry {url}: {json.loads(resp.read().decode('utf-8'))}")
        return 0
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"registry {url} unreachable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
