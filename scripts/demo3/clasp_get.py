"""GET (or POST) one URL of the cluster or the registry over mutual TLS, and print the answer.

    python scripts/demo3/clasp_get.py https://localhost:8002/clusters --certs .runtime/certs/admin
    python scripts/demo3/clasp_get.py https://<A_IP>:8002/healthz --certs edge-certs/client-flask
    python scripts/demo3/clasp_get.py https://localhost:8004/healthz --certs .runtime/certs/admin --no-cert

Windows' own curl cannot present a PEM client certificate, so the demo's
inspection commands use this instead. ``--certs`` is a directory holding
``ca.pem`` and one certificate/key pair (``admin.pem``/``admin.key``,
``client.pem``/``client.key``, ``demo-ui.pem``/``demo-ui.key`` ...).
``--no-cert`` presents no certificate, to show the handshake being refused.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path


def find_pair(certs: Path) -> tuple[Path, Path]:
    for pem in sorted(certs.glob("*.pem")):
        key = pem.with_suffix(".key")
        if pem.name not in ("ca.pem", "page.pem") and key.is_file():
            return pem, key
    raise SystemExit(f"no certificate/key pair (<name>.pem + <name>.key) in {certs}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="call a CLASP service over mutual TLS")
    ap.add_argument("url")
    ap.add_argument("--certs", type=Path, required=True, help="directory with ca.pem and a cert/key pair")
    ap.add_argument("--no-cert", action="store_true", help="present no client certificate")
    ap.add_argument("--post", action="store_true", help="POST an empty JSON object instead of GET")
    ap.add_argument("--server-hostname", default=None,
                    help="name to check the server certificate against, when not the URL host")
    args = ap.parse_args(argv)

    ca = args.certs / "ca.pem"
    if not ca.is_file():
        raise SystemExit(f"{ca} not found")
    ctx = ssl.create_default_context(cafile=str(ca))
    who = "no client certificate"
    if not args.no_cert:
        pem, key = find_pair(args.certs)
        ctx.load_cert_chain(str(pem), str(key))
        who = f"certificate {pem.name}"
    handler = urllib.request.HTTPSHandler(context=ctx)
    if args.server_hostname:
        # check the certificate against this name instead of the URL's host
        import http.client

        class _Conn(http.client.HTTPSConnection):
            def connect(self):
                http.client.HTTPConnection.connect(self)
                self.sock = ctx.wrap_socket(self.sock, server_hostname=args.server_hostname)

        class _Handler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(_Conn, req)

        handler = _Handler()
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(args.url, method="POST" if args.post else "GET",
                                 data=b"{}" if args.post else None,
                                 headers={"Content-Type": "application/json"})
    print(f"{req.method} {args.url}   ({who})")
    try:
        with opener.open(req, timeout=15) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        print(f"  refused before any HTTP answer: {type(reason).__name__}: {reason}")
        return 2
    print(f"  HTTP {status}")
    try:
        text = json.dumps(json.loads(body), indent=2)
    except ValueError:
        text = body.decode("utf-8", errors="replace")
    sys.stdout.write("  " + text.replace("\n", "\n  ") + "\n")
    return 0 if status < 400 else 1


if __name__ == "__main__":
    sys.exit(main())
