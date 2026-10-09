"""End-to-end registry demo against a LIVE registry — the compose `demo-seed` job.

    python -m registry.demo --url http://registry:8004 [--report out.json]

Walks every P4 capability over real HTTP, asserting each response:

    1. save a cluster adapter and a client adapter        (versioning, sha256)
    2. compose them into a pre-merged composite            (D6)
    3. save a better client v2, PROMOTE it with build-on-promote composite (D5+D6)
    4. save a bad client v3, D5 ROLLS IT BACK                (D5, real rule)
    5. restore drill on a 1.3B-sized client: its composite follows, timed
       on the composite the edge fetches            (D11: <= 10 s)
    6. lineage + audit trail, then a retention dry run      (D9)

Adapter names are suffixed with a run id so repeated runs never collide with
each other or with real adapters. Tensor values are synthetic (seeded); shapes,
keys and hyperparameters follow the real adapters. Exits non-zero on the first
unexpected response.
"""
from __future__ import annotations

import argparse
import json
import secrets
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .composite import serialize_canonical

RESTORE_NFR_SECONDS = 10.0
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_TIMEOUT = 60


class DemoFailure(RuntimeError):
    pass


def make_adapter(seed: int, *, layers: int, hidden: int, rank: int = 16) -> bytes:
    """A PEFT-keyed LoRA payload with an embedded adapter_config (lora_alpha == r)."""
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(layers):
        for m in MODULES:
            key = f"base_model.model.model.layers.{layer}.self_attn.{m}"
            sd[f"{key}.lora_A.weight"] = rng.standard_normal((rank, hidden), dtype=np.float32) * 0.01
            sd[f"{key}.lora_B.weight"] = rng.standard_normal((hidden, rank), dtype=np.float32) * 0.01
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": rank,
           "target_modules": list(MODULES), "use_rslora": False}
    return serialize_canonical(sd, {"adapter_config": json.dumps(cfg), "format": "pt"})


class Registry:
    """Minimal stdlib HTTP client — the demo needs no extra dependencies."""

    def __init__(self, base_url: str, context: ssl.SSLContext | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.context = context

    def _send(self, req: urllib.request.Request) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT, context=self.context) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get(self, path: str, *, raw: bool = False):
        status, body = self._send(urllib.request.Request(self.base + path))
        return status, (body if raw else json.loads(body or b"null"))

    def post_json(self, path: str, body: dict):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        status, raw = self._send(req)
        return status, json.loads(raw or b"null")

    def save(self, name: str, payload: bytes, meta: dict):
        boundary = f"clasp-{secrets.token_hex(8)}"
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="meta"\r\n\r\n'
            f"{json.dumps(meta)}\r\n".encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{name}.safetensors"\r\nContent-Type: application/octet-stream\r\n\r\n'
            .encode() + payload + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        req = urllib.request.Request(
            f"{self.base}/adapters/{name}/versions", data=b"".join(parts), method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        status, raw = self._send(req)
        return status, json.loads(raw or b"null")


def _expect(step: str, status: int, want: int, body) -> None:
    if status != want:
        raise DemoFailure(f"{step}: expected HTTP {want}, got {status}: {body}")


def _eval(name: str, version: int, sim: float, pass1: float) -> dict:
    metrics = lambda s: {"edit_similarity": s, "exact_match": s / 2,  # noqa: E731
                         "perplexity": 3.0, "n_examples": 40}
    return {
        "eval": {
            "adapter": {"name": name, "version": version, "kind": "client", "cluster_id": "web"},
            "in_project": metrics(sim),
            "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": pass1}}],
            "baseline_in_project": metrics(0.60),
            "baseline_noise_band": 0.02,
        },
        "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.30}}],
    }


def _restore_drill(reg: Registry, run_id: str, hp: dict, *, layers: int, hidden: int,
                   log) -> float:
    """D11, timed on what the edge serves: restore a client, read back its composite.

    A 1.3B-sized cluster + client v1/v2 and their composites. Restoring the
    client must move the served composite back to the v1 build; the clock
    covers the restore and the edge's fetch of the composite's metadata + payload.
    """
    cluster, client, composite = f"nfr-cluster-{run_id}", f"nfr-{run_id}", f"nfr-composite-{run_id}"
    status, body = reg.save(cluster, make_adapter(5, layers=layers, hidden=hidden),
                            {"kind": "cluster", "hparams": hp, "aggregation": "svd_exact",
                             "source_clients": [client]})
    _expect("save nfr cluster", status, 201, body)
    compose = {"cluster": cluster, "client": client, "alpha": 0.5, "beta": 1.0}
    for seed in (6, 7):
        status, body = reg.save(client, make_adapter(seed, layers=layers, hidden=hidden),
                                {"kind": "client", "hparams": hp})
        _expect("save nfr client", status, 201, body)
        status, body = reg.post_json(f"/adapters/{composite}/compose", compose)
        _expect("compose nfr", status, 201, body)
    status, good = reg.get(f"/adapters/{composite}/versions/1/file", raw=True)
    _expect("fetch nfr composite v1", status, 200, good)

    start = time.perf_counter()
    status, body = reg.post_json(f"/adapters/{client}/restore", {"reason": "D11 restore drill"})
    _expect("restore", status, 200, body)
    status, active = reg.get(f"/adapters/{composite}/active")
    status, blob = reg.get(f"/adapters/{composite}/versions/{active['ref']['version']}/file",
                           raw=True)
    elapsed = time.perf_counter() - start
    if active["composed_from"]["client_version"] != 1 or blob != good:
        raise DemoFailure(f"restore: composite still serves {active['composed_from']}")
    if elapsed > RESTORE_NFR_SECONDS:
        raise DemoFailure(f"restore NFR: {elapsed:.2f}s > {RESTORE_NFR_SECONDS}s")
    log("restore_nfr", seconds=round(elapsed, 3), payload_mb=round(len(blob) / 2**20, 1),
        served_client_version=1, composites_moved=len(body["composites"]),
        budget_seconds=RESTORE_NFR_SECONDS)
    return elapsed


def run_demo(reg: Registry, *, run_id: str, layers: int = 2, hidden: int = 64,
             nfr_layers: int = 24, nfr_hidden: int = 2048) -> dict:
    cluster, client, composite = f"cluster-web-{run_id}", f"flask-{run_id}", f"composite-flask-{run_id}"
    hp = {"rank": 16, "lora_alpha": 16, "target_modules": list(MODULES)}
    report: dict = {"run_id": run_id, "steps": []}

    def log(step: str, **detail) -> None:
        report["steps"].append({"step": step, **detail})
        print(f"ok  {step}  {json.dumps(detail, default=str)}", flush=True)

    status, body = reg.get("/healthz")
    _expect("healthz", status, 200, body)
    log("healthz", contracts=body["contracts"])

    status, body = reg.save(cluster, make_adapter(1, layers=layers, hidden=hidden), {
        "kind": "cluster", "hparams": hp, "aggregation": "svd_exact", "round": 1,
        "cluster_id": "web", "source_clients": ["flask", "requests", "werkzeug"],
        "privacy": {"epsilon": 3.0}})
    _expect("save cluster", status, 201, body)
    status, body = reg.save(client, make_adapter(2, layers=layers, hidden=hidden), {
        "kind": "client", "hparams": hp, "round": 1, "cluster_id": "web",
        "privacy": {"epsilon": 2.5}})
    _expect("save client v1", status, 201, body)
    log("save", cluster_sha256=body["sha256"][:12])

    status, body = reg.post_json(f"/adapters/{composite}/compose", {
        "cluster": cluster, "client": client, "alpha": 0.5, "beta": 1.0})
    _expect("compose", status, 201, body)
    log("compose", version=body["ref"]["version"], rank=body["hparams"]["rank"],
        epsilon=body["privacy"]["epsilon"])

    status, body = reg.save(client, make_adapter(3, layers=layers, hidden=hidden),
                            {"kind": "client", "hparams": hp, "round": 2, "cluster_id": "web"})
    _expect("save client v2", status, 201, body)
    promote = _eval(client, 2, sim=0.70, pass1=0.30)
    promote["composite"] = {"name": composite, "cluster": cluster, "client": client,
                            "alpha": 0.5, "beta": 1.0}
    status, body = reg.post_json(f"/adapters/{client}/promote", promote)
    _expect("promote v2", status, 200, body)
    if body["action"] != "promote" or body["composite"]["composed_from"]["client_version"] != 2:
        raise DemoFailure(f"promote v2: unexpected decision {body}")
    log("promote", action=body["action"], composite_version=body["composite"]["ref"]["version"])

    status, body = reg.save(client, make_adapter(4, layers=layers, hidden=hidden),
                            {"kind": "client", "hparams": hp, "round": 3, "cluster_id": "web"})
    _expect("save client v3", status, 201, body)
    status, body = reg.post_json(f"/adapters/{client}/promote", _eval(client, 3, 0.61, 0.20))
    _expect("promote v3", status, 200, body)
    if body["action"] != "rollback" or body["active_version_after"] != 2:
        raise DemoFailure(f"promote v3: expected rollback to v2, got {body}")
    log("rollback", reason=body["reason"])

    report["restore_seconds"] = _restore_drill(reg, run_id, hp, layers=nfr_layers,
                                               hidden=nfr_hidden, log=log)

    status, body = reg.get(f"/adapters/{composite}/lineage")
    _expect("lineage", status, 200, body)
    log("lineage", parents=[v["parents"] for v in body["versions"]])
    status, body = reg.get(f"/adapters/{client}/promotions")
    log("audit_trail", decisions=[d["action"] for d in body["decisions"]])
    status, body = reg.post_json(f"/adapters/{client}/gc", {"keep_last": 1})
    _expect("gc dry run", status, 200, body)
    log("gc_dry_run", would_delete=body["deleted"], kept=body["kept"])
    report["ok"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8004")
    ap.add_argument("--report", type=Path, help="write the JSON report here")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /healthz")
    args = ap.parse_args(argv)

    reg = Registry(args.url)
    deadline = time.time() + args.wait
    while True:
        try:
            if reg.get("/healthz")[0] == 200:
                break
        except OSError:
            pass
        if time.time() > deadline:
            print(f"registry at {args.url} not healthy after {args.wait}s", file=sys.stderr)
            return 1
        time.sleep(1)

    try:
        report = run_demo(reg, run_id=args.run_id or time.strftime("%Y%m%d%H%M%S"))
    except DemoFailure as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print("registry demo: all steps passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
