"""The edge laptop's own page: pick an adapter, optionally train briefly, upload.

    python -m edge.webui --cluster-url http://192.168.43.10:8002 --client-id client-flask
    # then open http://127.0.0.1:8020

One page per edge laptop (demo plan v3 item 3). It is a thin face on the edge
commands, not a service: uploads go through :func:`edge.upload.upload_adapter`
and a short fine-tune runs ``python -m edge.train_client`` as a subprocess, so
the page can do nothing the command line cannot.

* Adapters are offered from the edge's artifact directories (``--adapters-root``,
  default ``services/edge/artifacts``): every ``<set>/<client>/adapter`` with a
  safetensors file, newest first. The browser picks one by id; it never sends
  a path.
* Only differentially private adapters are offered or sent (D7): one upload
  without an ε makes the whole round's composite ε ``None`` on the cluster.
  ``--allow-non-dp`` lifts that for experiments.
* Each fine-tune started here writes its own ``live-<timestamp>`` set, so a new
  run adds an adapter instead of overwriting the last one.
* The status panel polls the cluster's ``GET /clusters`` and tracks the adapter
  this page last sent: sent, the rest of the cluster arriving, merged, and back
  on this laptop.
* The way back: once laptop A has merged a round, the page pulls the cluster's
  merged adapter (:func:`edge.upload.pull_cluster_adapter`, the same cluster
  service and session as the upload) into ``<root>/from_cluster/<cluster>-round<N>``,
  and the next fine-tune trains on top of it (D3, ``--cluster-adapter``).
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
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import requests

from edge import upload
from edge.config import REPO_ROOT

DEFAULT_PORT = 8020
DEFAULT_ADAPTERS_ROOT = REPO_ROOT / "services" / "edge" / "artifacts"
#: Prefix of the set a fine-tune started from the page writes
#: (``<root>/live-<YYYYmmdd-HHMMSS>/<client>/adapter``), one set per run.
LIVE_SET = "live"
#: Where merged cluster adapters pulled back from laptop A land. Flat PEFT
#: directories (no ``<client>/adapter`` level), so they are never offered for upload.
PULLED_SET = "from_cluster"
PAGE = Path(__file__).with_name("webui_page.html")


def find_pulled(root: Path) -> List[Dict]:
    """Merged cluster adapters pulled into ``<root>/from_cluster``, newest round first."""
    out = []
    for m in Path(root, PULLED_SET).glob("*/pull_manifest.json"):
        if (m.parent / "adapter_model.safetensors").exists():
            info = json.loads(m.read_text(encoding="utf-8"))
            out.append({**info, "id": m.parent.name, "path": m.parent})
    return sorted(out, key=lambda p: (p["round_id"], p["id"]), reverse=True)


def find_adapters(root: Path) -> List[Dict]:
    """Every ``<set>/<client>/adapter`` under ``root`` that holds trained weights,
    newest first."""
    out = []
    for weights in Path(root).glob("*/*/adapter/adapter_model.safetensors"):
        adapter = weights.parent
        manifest = upload.training_manifest(adapter)
        privacy = manifest.get("privacy") or {}
        d3 = manifest.get("d3") or {}
        set_name = adapter.parent.parent.name
        trained = manifest.get("utc") or datetime.fromtimestamp(
            weights.stat().st_mtime, timezone.utc).isoformat()
        out.append({
            "id": f"{set_name}/{adapter.parent.name}",
            "set": set_name,
            "trained_here": set_name == LIVE_SET or set_name.startswith(LIVE_SET + "-"),
            "client_id": upload.default_client_id(adapter),
            "cluster": upload.manifest_cluster(manifest),
            "num_examples": upload.manifest_num_examples(manifest),
            "steps": (manifest.get("results") or {}).get("optimizer_steps"),
            "epsilon": privacy.get("epsilon"),
            "d3_alpha": d3.get("alpha"),
            "trained_utc": trained,
            "size_mb": round(weights.stat().st_size / 1e6, 1),
            "path": adapter,
        })
    return sorted(out, key=lambda a: (a["trained_utc"], a["id"]), reverse=True)


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
                 python: str = sys.executable, allow_non_dp: bool = False) -> None:
        self.cluster_url = cluster_url
        self.client_id = client_id
        self.adapters_root = Path(adapters_root)
        self.corpus_root = corpus_root
        self.http = http or requests.Session()
        self.python = python
        self.allow_non_dp = allow_non_dp
        self.job = Job()
        #: The last adapter this page sent: id, cluster and round it went into.
        self.last_sent: Optional[Dict] = None

    # -- reads --------------------------------------------------------------
    def adapters(self) -> List[Dict]:
        found = find_adapters(self.adapters_root)
        if self.client_id:
            found = [a for a in found if a["client_id"] == self.client_id]
        if not self.allow_non_dp:
            found = [a for a in found if a["epsilon"] is not None]
        return found

    def status(self) -> Dict:
        out: Dict = {"cluster_url": self.cluster_url, "client_id": self.client_id,
                     "secure": self.cluster_url.startswith("https"), "reachable": False,
                     "allow_non_dp": self.allow_non_dp, "last_sent": self.last_sent}
        try:
            clusters = upload.list_clusters(self.http, self.cluster_url, timeout=5)
        except upload.UploadError as exc:
            out["error"], out["hint"] = str(exc).split("\n")[0], exc.hint
            return out
        out["reachable"] = True
        out["clusters"] = clusters
        mine = [cid for cid, c in clusters.items() if self.client_id in c.get("members", [])]
        out["my_cluster"] = mine[0] if mine else None
        out["pulled"] = self._pulled_public(self.latest_pulled(out["my_cluster"]))
        return out

    def latest_pulled(self, cluster_id: Optional[str]) -> Optional[Dict]:
        if cluster_id is None:
            return None
        return next((p for p in find_pulled(self.adapters_root)
                     if p["cluster_id"] == cluster_id), None)

    @staticmethod
    def _pulled_public(p: Optional[Dict]) -> Optional[Dict]:
        return {k: v for k, v in p.items() if k != "path"} if p else None

    # -- actions ------------------------------------------------------------
    def _adapter(self, adapter_id: str) -> Dict:
        for a in find_adapters(self.adapters_root):
            if a["id"] == adapter_id:
                return a
        raise KeyError(adapter_id)

    def start_upload(self, adapter_id: str) -> Dict:
        adapter = self._adapter(adapter_id)
        if adapter["epsilon"] is None and not self.allow_non_dp:
            raise ValueError(f"{adapter_id} was trained without differential privacy; "
                             f"train one with DP (or start the page with --allow-non-dp)")
        if not self.job.begin("upload"):
            raise RuntimeError("another job is still running")

        def run() -> None:
            try:
                res = upload.upload_adapter(
                    adapter["path"], cluster_url=self.cluster_url,
                    client_id=self.client_id or adapter["client_id"], http=self.http,
                    log=self.job.log)
                self.last_sent = {"adapter": adapter_id, "cluster_id": res["cluster_id"],
                                  "round_id": res["round_id"], "utc": time.time()}
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

    def start_pull(self) -> Dict:
        status = self.status()
        if not status["reachable"]:
            raise RuntimeError(f"cluster not reachable at {self.cluster_url}")
        cid = status["my_cluster"]
        if cid is None:
            raise ValueError("this laptop is not a member of any cluster yet; "
                             "laptop A registers the clusters first")
        if not self.job.begin("pull"):
            raise RuntimeError("another job is still running")

        def run() -> None:
            try:
                res = upload.pull_cluster_adapter(
                    self.adapters_root / PULLED_SET, cluster_id=cid,
                    cluster_url=self.cluster_url, http=self.http, log=self.job.log)
                self.job.finish(result={k: v for k, v in res.items() if k != "path"})
            except upload.UploadError as exc:
                self.job.log(f"REFUSED: {exc}")
                self.job.finish(error=str(exc))
            except Exception as exc:  # noqa: BLE001 - surface anything on the page
                self.job.log(f"FAILED: {type(exc).__name__}: {exc}")
                self.job.finish(error=f"{type(exc).__name__}: {exc}")

        threading.Thread(target=run, daemon=True).start()
        return {"started": "pull", "cluster_id": cid}

    def train_command(self, *, client: str, max_steps: int, dp: bool,
                      cluster_adapter: Optional[str], out_set: str = LIVE_SET) -> List[str]:
        cmd = [self.python, "-m", "edge.train_client", "--client", client,
               "--max-steps", str(int(max_steps)), "--skip-base-eval",
               "--out-dir", str(self.adapters_root / out_set)]
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
        name = client.rstrip("/").rsplit("/", 1)[-1]
        if self.client_id and name != self.client_id:
            # Its adapter would be filtered out of this page and never offered.
            raise ValueError(f"this page is for {self.client_id}; train "
                             f"'<cluster>/{self.client_id}', not '{client}'")
        if not dp and not self.allow_non_dp:
            raise ValueError("only differentially private adapters can be sent from this page; "
                             "train with DP (or start the page with --allow-non-dp)")
        if cluster_adapter and not (Path(cluster_adapter) / "adapter_config.json").exists():
            raise ValueError(f"{cluster_adapter} is not an adapter directory")
        # One set per run, so a new adapter is added beside the last, not over it.
        out_set = f"{LIVE_SET}-{time.strftime('%Y%m%d-%H%M%S')}"
        cmd = self.train_command(client=client, max_steps=max_steps, dp=dp,
                                 cluster_adapter=cluster_adapter, out_set=out_set)
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
            adapter_id = f"{out_set}/{name}"
            if code == 0 and (self.adapters_root / adapter_id / "adapter").is_dir():
                self.job.log(f"done — the new adapter is {adapter_id}, selected above")
                self.job.finish(result={"exit_code": 0, "adapter": adapter_id})
            elif code == 0:
                self.job.finish(error=f"training finished but wrote no adapter at {adapter_id}")
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
                if self.path == "/api/pull":
                    return self._send(202, page.start_pull())
                if self.path == "/api/train":
                    # D3: train on top of the merged adapter last pulled back
                    # from laptop A. The browser asks for it; it never names a path.
                    base = None
                    if body.get("from_cluster"):
                        base = page.latest_pulled(page.status().get("my_cluster"))
                        if base is None:
                            raise ValueError("no merged cluster adapter on this laptop yet; "
                                             "pull it after laptop A merges a round")
                    return self._send(202, page.start_train(
                        client=str(body["client"]), max_steps=int(body.get("max_steps", 50)),
                        dp=bool(body.get("dp")),
                        cluster_adapter=str(base["path"]) if base else None))
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
    ap.add_argument("--allow-non-dp", action="store_true",
                    help="also offer and send adapters trained without differential privacy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    try:
        http = upload.session_for(args.mtls_dir, args.server_hostname)
    except upload.UploadError as exc:
        print(f"edge.webui: {exc}", file=sys.stderr)
        return 1
    page = EdgePage(cluster_url=args.cluster_url, client_id=args.client_id,
                    adapters_root=args.adapters_root, http=http, corpus_root=args.corpus_root,
                    allow_non_dp=args.allow_non_dp)
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
