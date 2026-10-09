"""Upload one trained client adapter to a (remote) cluster — seam A, per cluster.

    python -m edge.upload --cluster-url http://192.168.43.10:8002 \\
        --adapter services/edge/artifacts/round2_d3/client-flask/adapter

What it does, in order:

1. ``GET /clusters`` on the cluster service: which cluster this client is a
   member of, and which round that cluster is buffering. The cluster is taken
   from the membership laptop A registered (``PUT /clusters/{id}/members``),
   else ``--cluster``, else the D1 cluster named in the training manifest
   (``client_id: "web/client-flask"``).
2. Builds the upload body from the adapter directory (``edge.wire``): fp32
   tensors, the adapter's own LoRA hyperparameters, the FedAvg weight (the
   client's training block count, from the manifest ``edge.train_client``
   wrote) and the DP ``privacy`` block that run recorded (D7).
3. ``POST /clusters/{id}/uploads`` — the cluster-addressed route, which answers
   404 for an unregistered cluster instead of opening a new one.

Over mTLS (``--mtls-dir``, the per-edge directory ``scripts/make_dev_certs.sh``
writes) the cluster checks that ``client_id`` is the CN of this edge's
certificate, so the certificate decides who may upload as whom.

The cluster's refusals are translated into what to do about them: 404 (cluster
not registered yet), 409 (stale round: someone aggregated in between), 403
(assigned to another cluster, or a certificate for a different client).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests

from edge import transport, wire

DEFAULT_CLUSTER_URL = os.environ.get("CLASP_CLUSTER_URL", "http://localhost:8002")


class UploadError(RuntimeError):
    """An upload the cluster refused or could not be asked; ``hint`` says what to do."""

    def __init__(self, message: str, *, status: Optional[int] = None, hint: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n  -> {self.hint}" if self.hint else base


@dataclass(frozen=True)
class Target:
    cluster_id: str
    round_id: int
    members: tuple
    how: str  # how the cluster was chosen, for the log


def training_manifest(adapter_dir: Path) -> Dict:
    """``<client>/manifest.json`` beside ``<client>/adapter/``, or ``{}``."""
    path = Path(adapter_dir).parent / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def default_client_id(adapter_dir: Path) -> str:
    """``.../client-flask/adapter`` -> ``client-flask`` (the demo's client ids)."""
    adapter_dir = Path(adapter_dir)
    return (adapter_dir.parent if adapter_dir.name == "adapter" else adapter_dir).name


def manifest_cluster(manifest: Dict) -> Optional[str]:
    """D1's static cluster from the manifest's ``client_id`` ("web/client-flask")."""
    cid = str(manifest.get("client_id") or "")
    return cid.split("/", 1)[0] if "/" in cid else None


def manifest_num_examples(manifest: Dict) -> Optional[int]:
    n = (manifest.get("data") or {}).get("train", {}).get("n_chunks")
    return int(n) if n else None


def _http_error(r: requests.Response) -> str:
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return detail if isinstance(detail, str) else json.dumps(detail)


def list_clusters(http: requests.Session, cluster_url: str, timeout: float = 15.0) -> Dict:
    url = f"{cluster_url.rstrip('/')}/clusters"
    try:
        r = http.get(url, timeout=timeout)
    except requests.exceptions.SSLError as exc:
        raise UploadError(f"TLS handshake with {cluster_url} failed: {exc}",
                          hint="check --mtls-dir (this edge's ca.pem/client.pem/client.key) "
                               "and that laptop A's IP is in the cluster certificate's SANs") from exc
    except requests.exceptions.ConnectionError as exc:
        raise UploadError(f"cluster not reachable at {cluster_url}: {exc}",
                          hint="is laptop A's stack up, on the same network, and is port 8002 "
                               "open in its firewall?") from exc
    if r.status_code != 200:
        raise UploadError(f"GET /clusters -> {r.status_code}: {_http_error(r)}", status=r.status_code)
    return r.json()["clusters"]


def resolve_target(clusters: Dict, client_id: str, requested: Optional[str],
                   from_manifest: Optional[str]) -> Target:
    """Which cluster and round to upload into (see the module docstring)."""
    assigned = [cid for cid, c in clusters.items() if client_id in c.get("members", [])]
    if assigned:
        cid = assigned[0]
        if requested and requested != cid:
            raise UploadError(
                f"{client_id} is assigned to cluster {cid!r}, not {requested!r}", status=403,
                hint=f"drop --cluster, or re-assign it on laptop A (PUT /clusters/{requested}/members)")
        how = "assigned on the cluster"
    else:
        cid = requested or from_manifest
        how = "--cluster" if requested else "training manifest (D1)"
        if cid is None:
            raise UploadError(f"{client_id} is not a member of any cluster and none was named",
                              hint="pass --cluster, or register the clusters in demo_ui first")
    if cid not in clusters:
        raise UploadError(
            f"cluster {cid!r} is not registered on the cluster service "
            f"(registered: {sorted(clusters) or 'none'})", status=404,
            hint="on laptop A, press 'Register clusters' in demo_ui (PUT /clusters/{id}/members)")
    c = clusters[cid]
    return Target(cid, int(c["round_id"]), tuple(c.get("members", [])), how)


def upload_adapter(adapter_dir: Path | str, *, cluster_url: str = DEFAULT_CLUSTER_URL,
                   client_id: Optional[str] = None, cluster_id: Optional[str] = None,
                   num_examples: Optional[int] = None, round_id: Optional[int] = None,
                   seed: Optional[int] = None, http: Optional[requests.Session] = None,
                   timeout: float = transport.DEFAULT_UPLOAD_TIMEOUT,
                   log=print) -> Dict:
    """Upload one adapter directory; returns the cluster's answer plus what was sent."""
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise UploadError(f"{adapter_dir} has no adapter_model.safetensors",
                          hint="point --adapter at a trained client's adapter/ directory")
    http = http or requests.Session()
    manifest = training_manifest(adapter_dir)
    client_id = client_id or default_client_id(adapter_dir)
    n = num_examples or manifest_num_examples(manifest)
    if not n:
        raise UploadError(f"no training manifest beside {adapter_dir} to read the sample count from",
                          hint="pass --num-examples (the client's number of training blocks)")

    target = resolve_target(list_clusters(http, cluster_url), client_id, cluster_id,
                            manifest_cluster(manifest))
    rnd = target.round_id if round_id is None else round_id
    log(f"edge.upload  {client_id} -> cluster {target.cluster_id!r} round {rnd} "
        f"({target.how}) at {cluster_url}")

    privacy = wire.privacy_from_training_manifest(adapter_dir)
    payload = wire.upload_payload_from_dir(
        adapter_dir, client_id=client_id, cluster_id=target.cluster_id, round_id=rnd,
        num_examples=n, seed=seed if seed is not None else manifest.get("seed"),
        privacy=privacy)
    size_mb = sum(len(t["data_b64"]) for t in payload["tensors"]) * 3 / 4 / 1e6
    log(f"  {len(payload['tensors'])} tensors, {payload['num_layers']} layers, "
        f"rank {payload['rank']}, {size_mb:.1f} MB fp32, weight {n} blocks, "
        f"epsilon {privacy['epsilon'] if privacy['epsilon'] is not None else 'none (DP off)'}")

    url = f"{cluster_url.rstrip('/')}/clusters/{target.cluster_id}/uploads"
    t0 = time.monotonic()
    try:
        r = http.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise UploadError(f"upload to {url} failed: {exc}") from exc
    seconds = round(time.monotonic() - t0, 2)
    if r.status_code == 201:
        body = r.json()
        log(f"  accepted in {seconds}s: {body['pending_uploads']} upload(s) now pending "
            f"in {target.cluster_id!r} round {body['round_id']}")
        return {"response": body, "cluster_id": target.cluster_id, "client_id": client_id,
                "round_id": rnd, "num_examples": n, "privacy": privacy, "seconds": seconds,
                "cluster_url": cluster_url}
    detail = _http_error(r)
    hints = {
        404: "the cluster was not registered; press 'Register clusters' in demo_ui on laptop A",
        409: ("the cluster moved to a new round (an aggregate ran) — run this again to upload "
              "into the current round" if "round_id" in detail else
              "this adapter was trained from another cluster's adapter; pull the current one"),
        403: ("this edge's certificate is for a different client — use --client-id matching "
              "the certificate's CN" if "authenticated identity" in detail else
              "this client is assigned to another cluster"),
        401: "the cluster requires a client certificate: pass --mtls-dir",
        422: "the cluster could not parse or combine this adapter with the round's others",
        429: "the cluster is full for this round",
    }
    raise UploadError(f"cluster refused the upload ({r.status_code}): {detail}",
                      status=r.status_code, hint=hints.get(r.status_code, ""))


def session_for(mtls_dir: Optional[Path], server_hostname: Optional[str] = None) -> requests.Session:
    """Plain session, or mTLS from a directory holding ca.pem, client.pem, client.key."""
    if mtls_dir is None:
        return requests.Session()
    mtls_dir = Path(mtls_dir)
    files = {"client_cert": mtls_dir / "client.pem", "client_key": mtls_dir / "client.key",
             "ca_cert": mtls_dir / "ca.pem"}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise UploadError(f"missing certificate files {missing}",
                          hint="copy .runtime/certs/edges/<client_id>/ from laptop A")
    return transport.mtls_session(**files, server_hostname=server_hostname)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="upload a trained client adapter to a cluster (seam A)")
    ap.add_argument("--adapter", type=Path, required=True,
                    help="the client's adapter/ directory (adapter_model.safetensors + config)")
    ap.add_argument("--cluster-url", default=DEFAULT_CLUSTER_URL,
                    help="cluster service, e.g. http://<laptop A IP>:8002 (env CLASP_CLUSTER_URL)")
    ap.add_argument("--client-id", default=None, help="default: the adapter's client directory name")
    ap.add_argument("--cluster", default=None,
                    help="cluster id when this client is not assigned to one (default: manifest)")
    ap.add_argument("--num-examples", type=int, default=None,
                    help="FedAvg weight; default: training blocks from the manifest")
    ap.add_argument("--round", type=int, default=None, help="default: the cluster's current round")
    ap.add_argument("--mtls-dir", type=Path, default=None,
                    help="this edge's certs: ca.pem, client.pem, client.key")
    ap.add_argument("--server-hostname", default=None,
                    help="name to check the cluster's certificate against, when not the URL host")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    try:
        http = session_for(args.mtls_dir, args.server_hostname)
        out = upload_adapter(args.adapter, cluster_url=args.cluster_url, client_id=args.client_id,
                             cluster_id=args.cluster, num_examples=args.num_examples,
                             round_id=args.round, http=http,
                             # with --json, stdout carries only the JSON
                             log=(lambda m: print(m, file=sys.stderr)) if args.json else print)
    except UploadError as exc:
        print(f"edge.upload: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
