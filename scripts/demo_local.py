"""The whole multi-laptop demo on ONE laptop — the fallback (demo plan v3 item 14).

    python scripts/demo_local.py                                  # client-flask + client-numpy
    python scripts/demo_local.py client-flask client-requests client-numpy
    python scripts/demo_local.py --mtls                           # same, over mTLS

Starts registry + cluster + demo_ui (``run_services.py --ui``), registers the
clusters, then one ``edge.webui`` process per client on :8020, :8021, ... —
each a separate edge in its own process, exactly what laptops B and C run, only
pointed at localhost. Open http://127.0.0.1:8010 for the live panel and the
printed edge pages to upload. Ctrl-C stops everything.

``--mtls`` makes throwaway certs (scripts/make_dev_certs.sh, into
.runtime/certs-local) and runs every hop over mTLS with per-edge certificates.
``--registry-data`` keeps this run's versions apart from .runtime/registry-data.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / ".runtime"


def bash() -> str:
    """Git Bash on Windows (``bash`` alone resolves to WSL's System32 stub there)."""
    import shutil

    if os.name == "nt":
        git = shutil.which("git")
        if git:
            # git.exe sits in <Git>/cmd, <Git>/bin or <Git>/mingw64/bin
            for root in Path(git).resolve().parents[:3]:
                if (root / "bin" / "bash.exe").exists() and (root / "usr" / "bin").is_dir():
                    return str(root / "bin" / "bash.exe")
        raise SystemExit("--mtls needs Git Bash (openssl) to run scripts/make_dev_certs.sh")
    return shutil.which("bash") or "bash"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="multi-laptop demo on one laptop")
    ap.add_argument("clients", nargs="*", default=["client-flask", "client-numpy"])
    ap.add_argument("--mtls", action="store_true")
    ap.add_argument("--registry-data", type=Path, default=RUNTIME / "registry-data-local")
    ap.add_argument("--first-port", type=int, default=8020)
    args = ap.parse_args(argv)

    py = sys.executable
    run_services = [py, str(REPO_ROOT / "scripts" / "run_services.py"), "--ui", "--background",
                    "--host", "127.0.0.1", "--registry-data", str(args.registry_data)]
    certs = RUNTIME / "certs-local"
    if args.mtls:
        env = dict(os.environ, CLASP_EDGE_CLIENTS=" ".join(args.clients), MSYS_NO_PATHCONV="1")
        subprocess.run([bash(), "scripts/make_dev_certs.sh", certs.relative_to(REPO_ROOT).as_posix()],
                       cwd=str(REPO_ROOT), env=env, check=True)
        run_services += ["--mtls-dir", str(certs)]
    else:
        # run_services.py serves mTLS by default; this script's default stays plain HTTP
        run_services += ["--insecure"]
    subprocess.run(run_services, cwd=str(REPO_ROOT), check=True)

    scheme = "https" if args.mtls else "http"
    context = None
    if args.mtls:  # the panel's page is HTTPS with the pages certificate
        import ssl

        context = ssl.create_default_context(cafile=str(certs / "ca.pem"))
    req = urllib.request.Request(f"{scheme}://localhost:8010/api/live/register", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30, context=context) as r:
            print("registered:", json.loads(r.read())["registered"], flush=True)
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        # the panel asks for a login (.runtime/users/panel.json): the operator registers
        print("the panel asks for a login: log in and press 'Register edge laptops'", flush=True)

    edges = []
    for i, client in enumerate(args.clients):
        port = args.first_port + i
        cmd = [py, "-m", "edge.webui", "--cluster-url", f"{scheme}://localhost:8002",
               "--client-id", client, "--port", str(port)]
        if args.mtls:
            edge_certs = certs / "edges" / client
            cmd += ["--mtls-dir", str(edge_certs), "--tls-cert", str(edge_certs / "page.pem"),
                    "--tls-key", str(edge_certs / "page.key")]
        log = (RUNTIME / f"edge-{client}.log").open("w", encoding="utf-8")
        edges.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT))
        print(f"edge {client:22s} -> {scheme}://127.0.0.1:{port}", flush=True)
    print(f"live panel               -> {scheme}://localhost:8010\nctrl-c to stop everything",
          flush=True)
    if args.mtls:
        print(f"(import {certs / 'ca.pem'} into the browser, or accept its warning)", flush=True)
    try:
        while all(p.poll() is None for p in edges):
            time.sleep(1)
        print("an edge page exited; see .runtime/edge-*.log")
    except KeyboardInterrupt:
        pass
    finally:
        for p in edges:
            p.terminate()
        subprocess.run([py, str(REPO_ROOT / "scripts" / "run_services.py"), "--stop"], cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
