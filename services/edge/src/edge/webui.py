"""The edge laptop's own page: pick an adapter, optionally train briefly, upload.

    python -m edge.webui --cluster-url http://192.168.43.10:8002 --client-id client-flask
    # then open http://127.0.0.1:8020

One page per edge laptop (demo plan v3 item 3). It is a thin face on the edge
commands, not a service: uploads go through :func:`edge.upload.upload_adapter`
and a short fine-tune runs ``python -m edge.train_client`` as a subprocess, so
the page can do nothing the command line cannot.

* Adapters are offered from the edge's artifact directories (``--adapters-root``,
  default ``services/edge/artifacts``): every ``<set>/<client>/adapter`` with a
  safetensors file. The browser picks one by id; it never sends a path.
* The status panel polls the cluster's ``GET /clusters`` so the operator sees
  the round number and which clients have already uploaded.
* One job (upload or train) runs at a time; its log streams into the page.

Standard library only (``http.server``), so the page runs in the edge's
existing environment. It binds 127.0.0.1 by default: it is the operator's page
for this laptop, not something the network should reach.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import requests

from edge import upload
from edge.config import REPO_ROOT

DEFAULT_PORT = 8020
DEFAULT_ADAPTERS_ROOT = REPO_ROOT / "services" / "edge" / "artifacts"
#: Where a fine-tune started from the page writes (``<root>/live/<client>/adapter``).
LIVE_SET = "live"
PAGE = Path(__file__).with_name("webui_page.html")


def find_adapters(root: Path) -> List[Dict]:
    """Every ``<set>/<client>/adapter`` under ``root`` that holds trained weights."""
    out = []
    for weights in sorted(Path(root).glob("*/*/adapter/adapter_model.safetensors")):
        adapter = weights.parent
        manifest = upload.training_manifest(adapter)
        privacy = manifest.get("privacy") or {}
        d3 = manifest.get("d3") or {}
        out.append({
            "id": f"{adapter.parent.parent.name}/{adapter.parent.name}",
            "set": adapter.parent.parent.name,
            "client_id": upload.default_client_id(adapter),
            "cluster": upload.manifest_cluster(manifest),
            "num_examples": upload.manifest_num_examples(manifest),
            "epsilon": privacy.get("epsilon"),
            "d3_alpha": d3.get("alpha"),
            "trained_utc": manifest.get("utc"),
            "size_mb": round(weights.stat().st_size / 1e6, 1),
            "path": adapter,
        })
    return out


class Job:
    """The one upload or training run in flight, and its log."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.kind: Optional[str] = None
        self.running = False
        self.lines: List[str] = []
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.started: Optional[float] = None

    def begin(self, kind: str) -> bool:
        with self.lock:
            if self.running:
                return False
            self.kind, self.running, self.lines = kind, True, []
            self.result, self.error, self.started = None, None, time.time()
            return True

    def log(self, line: str) -> None:
        with self.lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")
            del self.lines[:-400]

    def finish(self, result: Optional[Dict] = None, error: Optional[str] = None) -> None:
        with self.lock:
            self.running, self.result, self.error = False, result, error

    def snapshot(self) -> Dict:
        with self.lock:
            return {"kind": self.kind, "running": self.running, "lines": list(self.lines),
                    "result": self.result, "error": self.error, "started": self.started}


class EdgePage:
    """State and actions behind the page (kept apart from HTTP for testing)."""

    def __init__(self, *, cluster_url: str, client_id: Optional[str], adapters_root: Path,
                 http: Optional[requests.Session] = None, corpus_root: Optional[Path] = None,
                 python: str = sys.executable) -> None:
        self.cluster_url = cluster_url
        self.client_id = client_id
        self.adapters_root = Path(adapters_root)
        self.corpus_root = corpus_root
        self.http = http or requests.Session()
        self.python = python
        self.job = Job()

    # -- reads --------------------------------------------------------------
    def adapters(self) -> List[Dict]:
        found = find_adapters(self.adapters_root)
        if self.client_id:
            found = [a for a in found if a["client_id"] == self.client_id]
        return found

    def status(self) -> Dict:
        out: Dict = {"cluster_url": self.cluster_url, "client_id": self.client_id,
                     "secure": self.cluster_url.startswith("https"), "reachable": False}
        try:
            clusters = upload.list_clusters(self.http, self.cluster_url, timeout=5)
        except upload.UploadError as exc:
            out["error"], out["hint"] = str(exc).split("\n")[0], exc.hint
            return out
        out["reachable"] = True
        out["clusters"] = clusters
        mine = [cid for cid, c in clusters.items() if self.client_id in c.get("members", [])]
        out["my_cluster"] = mine[0] if mine else None
        return out

    # -- actions ------------------------------------------------------------
    def _adapter(self, adapter_id: str) -> Dict:
        for a in find_adapters(self.adapters_root):
            if a["id"] == adapter_id:
                return a
        raise KeyError(adapter_id)

    def start_upload(self, adapter_id: str) -> Dict:
        adapter = self._adapter(adapter_id)
        if not self.job.begin("upload"):
            raise RuntimeError("another job is still running")

        def run() -> None:
            try:
                res = upload.upload_adapter(
                    adapter["path"], cluster_url=self.cluster_url,
                    client_id=self.client_id or adapter["client_id"], http=self.http,
                    log=self.job.log)
                self.job.finish(result={k: v for k, v in res.items() if k != "privacy"}
                                | {"epsilon": res["privacy"]["epsilon"]})
            except upload.UploadError as exc:
                self.job.log(f"REFUSED: {exc}")
                self.job.finish(error=str(exc))
            except Exception as exc:  # noqa: BLE001 - surface anything on the page
                self.job.log(f"FAILED: {type(exc).__name__}: {exc}")
                self.job.finish(error=f"{type(exc).__name__}: {exc}")

        threading.Thread(target=run, daemon=True).start()
        return {"started": "upload", "adapter": adapter_id}

    def train_command(self, *, client: str, max_steps: int, dp: bool,
                      cluster_adapter: Optional[str]) -> List[str]:
        cmd = [self.python, "-m", "edge.train_client", "--client", client,
               "--max-steps", str(int(max_steps)), "--skip-base-eval",
               "--out-dir", str(self.adapters_root / LIVE_SET)]
        if self.corpus_root:
            cmd += ["--corpus-root", str(self.corpus_root)]
        if cluster_adapter:
            cmd += ["--cluster-adapter", cluster_adapter]
        if dp:
            cmd.append("--dp")
        return cmd

    def start_train(self, *, client: str, max_steps: int, dp: bool,
                    cluster_adapter: Optional[str]) -> Dict:
        if not 1 <= int(max_steps) <= 2000:
            raise ValueError("max_steps must be between 1 and 2000")
        if "/" not in client or ".." in client:
            raise ValueError("client must look like 'web/client-flask'")
        if cluster_adapter and not (Path(cluster_adapter) / "adapter_config.json").exists():
            raise ValueError(f"{cluster_adapter} is not an adapter directory")
        cmd = self.train_command(client=client, max_steps=max_steps, dp=dp,
                                 cluster_adapter=cluster_adapter)
        if not self.job.begin("train"):
            raise RuntimeError("another job is still running")

        def run() -> None:
            self.job.log("$ " + " ".join(cmd[1:]))
            try:
                env = dict(os.environ, PYTHONUNBUFFERED="1")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", env=env)
                for line in proc.stdout:
                    self.job.log(line.rstrip())
                code = proc.wait()
            except OSError as exc:
                self.job.finish(error=f"could not start training: {exc}")
                return
            if code == 0:
                self.job.log(f"done — the new adapter is listed under '{LIVE_SET}/'")
                self.job.finish(result={"exit_code": 0})
            else:
                self.job.finish(error=f"edge.train_client exited with {code}")

        threading.Thread(target=run, daemon=True).start()
        return {"started": "train", "command": cmd[1:]}


def make_handler(page: EdgePage):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:  # keep the console for the job log
            pass

        def _send(self, code: int, body, ctype: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._send(200, page.status())
            elif self.path == "/api/adapters":
                self._send(200, {"adapters": [{k: v for k, v in a.items() if k != "path"}
                                              for a in page.adapters()]})
            elif self.path == "/api/job":
                self._send(200, page.job.snapshot())
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self) -> None:
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n <= 1 << 16 else None
                if body is None:
                    return self._send(413, {"detail": "request too large"})
                if self.path == "/api/upload":
                    return self._send(202, page.start_upload(str(body["adapter"])))
                if self.path == "/api/train":
                    return self._send(202, page.start_train(
                        client=str(body["client"]), max_steps=int(body.get("max_steps", 50)),
                        dp=bool(body.get("dp")), cluster_adapter=body.get("cluster_adapter") or None))
                return self._send(404, {"detail": "not found"})
            except KeyError as exc:
                self._send(404, {"detail": f"unknown adapter or field {exc}"})
            except (ValueError, TypeError) as exc:
                self._send(422, {"detail": str(exc)})
            except RuntimeError as exc:
                self._send(409, {"detail": str(exc)})

    return Handler


def serve(page: EdgePage, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(page))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="this edge laptop's upload page")
    ap.add_argument("--cluster-url", default=upload.DEFAULT_CLUSTER_URL,
                    help="cluster on laptop A, e.g. http://192.168.43.10:8002 (env CLASP_CLUSTER_URL)")
    ap.add_argument("--client-id", default=os.environ.get("CLASP_CLIENT_ID"),
                    help="this laptop's client (e.g. client-flask); default: offer every adapter")
    ap.add_argument("--adapters-root", type=Path, default=DEFAULT_ADAPTERS_ROOT)
    ap.add_argument("--corpus-root", type=Path, default=None,
                    help="client corpora for a live fine-tune (edge.train_client's default if unset)")
    ap.add_argument("--mtls-dir", type=Path, default=None,
                    help="this edge's certs (ca.pem, client.pem, client.key)")
    ap.add_argument("--server-hostname", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    try:
        http = upload.session_for(args.mtls_dir, args.server_hostname)
    except upload.UploadError as exc:
        print(f"edge.webui: {exc}", file=sys.stderr)
        return 1
    page = EdgePage(cluster_url=args.cluster_url, client_id=args.client_id,
                    adapters_root=args.adapters_root, http=http, corpus_root=args.corpus_root)
    httpd = serve(page, args.host, args.port)
    print(f"edge page for {args.client_id or 'all clients'} -> http://{args.host}:{args.port}  "
          f"(cluster {args.cluster_url}); ctrl-c to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
