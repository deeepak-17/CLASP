"""CLASP Panel Demo UI — thin orchestration layer, NOT a CLASP service.

Owner: P4 (Deepak). Lives outside services/ deliberately — this is presentation
scaffolding for the Phase II Panel Review, not production architecture. It:

  1. Serves the 3 static demo pages (static/page{1,2,3}.html).
  2. Reads REAL, already-committed artifacts for anything that can't run live
     in a browser demo (Edge's real training/composition/eval numbers need a
     GPU + the base model, neither available on a laptop at a panel) --
     `/api/edge/*`.
  3. Proxies straight through to the REAL Cluster (:8002) and Registry (:8004)
     FastAPI apps for everything that CAN run live -- `/api/cluster/*` and
     `/api/registry/*`. No aggregation or promotion logic is reimplemented
     here; every proxied call hits the actual service code reviewed and
     tested this sprint.
  4. Orchestrates the two multi-step demo actions ("send adapters to
     cluster", "publish + promote") by making the same HTTP calls a real
     Edge client would make -- again, no business logic duplicated, just
     sequencing real requests to the real services.

Tensor VALUES used in the live "send to cluster" step are synthetic --
`*.safetensors` are gitignored repo-wide and no trained weights exist in this
checkout (see services/cluster/tests/test_real_adapter_pipeline.py's own
honesty note, same situation here). Shapes and hyperparameters (rank 16,
q/k/v/o, matching each real client's own adapter_config.json) are real. This
is stated explicitly in the UI, never hidden.

Run: uvicorn demo_ui.server:app --port 8010  (from the repo root, with
contracts + cluster + registry installed, and cluster/registry already
running on :8002/:8004).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
EDGE_ARTIFACTS = Path(
    os.environ.get("CLASP_EDGE_ARTIFACTS", REPO_ROOT / "services" / "edge" / "artifacts")
)
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Host defaults; under compose these point at the service names instead.
CLUSTER_BASE = os.environ.get("CLASP_CLUSTER_URL", "http://localhost:8002")
REGISTRY_BASE = os.environ.get("CLASP_REGISTRY_URL", "http://localhost:8004")


def _tls_verify():
    """Under mTLS (docker-compose.mtls.yml, run_services.py --mtls-dir) this
    panel presents the demo-ui certificate to the cluster and the registry and
    trusts only the CLASP CA; otherwise plain HTTP, as before."""
    import ssl

    cert = os.environ.get("CLASP_TLS_CLIENT_CERT")
    key = os.environ.get("CLASP_TLS_CLIENT_KEY")
    ca = os.environ.get("CLASP_TLS_CA")
    if not (cert and key and ca):
        return True
    ctx = ssl.create_default_context(cafile=ca)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(cert, key)
    return ctx


_VERIFY = _tls_verify()


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, verify=_VERIFY)

# cluster.aggregation.aggregate_svd compares target_modules across clients as
# an ordered tuple; this is the canonical order every client is normalized to
# before upload (see _synthetic_tensors_for).
TARGET_MODULES_CANONICAL = ("q_proj", "k_proj", "v_proj", "o_proj")

CLIENT_IDS = [
    "client-flask", "client-requests", "client-werkzeug",       # web
    "client-numpy", "client-pandas", "client-scikit-learn",      # scientific
]
CLUSTER_OF = {
    "client-flask": "web", "client-requests": "web", "client-werkzeug": "web",
    "client-numpy": "scientific", "client-pandas": "scientific",
    "client-scikit-learn": "scientific",
}

app = FastAPI(title="CLASP Panel Demo UI")


# --------------------------------------------------------------------------- #
# Static pages
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> FileResponse:
    """The live multi-laptop round (demo plan v3); the panel pages are linked from it."""
    return FileResponse(STATIC_DIR / "live.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Real, committed Edge artifacts (no live Edge — needs GPU + base model)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/edge/clients")
def edge_clients() -> dict:
    """Real per-client training manifests — loss, held-out perplexity,
    LoRA hyperparams, wall time. Straight from services/edge/artifacts/round1/,
    committed by PR #9. Nothing computed here."""
    clients = []
    missing = []
    for cid in CLIENT_IDS:
        mpath = EDGE_ARTIFACTS / "round1" / cid / "manifest.json"
        cpath = EDGE_ARTIFACTS / "round1" / cid / "adapter" / "adapter_config.json"
        manifest = _read_json(mpath)
        cfg = _read_json(cpath)
        if manifest is None or cfg is None:
            missing.append(cid)
            continue
        r = manifest["results"]
        clients.append({
            "client_id": cid,
            "cluster": CLUSTER_OF[cid],
            "base_model": manifest["model_id"],
            "lora": manifest["lora"],
            "n_train_files": manifest["data"]["train"]["n_files"],
            "n_train_tokens": manifest["data"]["train"]["n_tokens"],
            "n_train_blocks": manifest["data"]["train"]["n_chunks"],
            "n_trainable_params": r.get("n_trainable_params"),
            "target_modules": cfg["target_modules"],
            "rank": cfg["r"],
            "held_out_ppl_before": r["base_held_out"]["perplexity"],
            "held_out_ppl_after": r["final_held_out"]["perplexity"],
            "held_out_ppl_delta": r["held_out_ppl_delta"],
            "wall_seconds": r["wall_seconds"],
            "hardware": manifest.get("hardware", {}),
        })
    return {"clients": clients, "missing": missing, "source": "services/edge/artifacts/round1/*/manifest.json (real, committed)"}


@app.get("/api/edge/round-summary")
def edge_round_summary() -> dict:
    """The real W4/W5 G2/G3 round result — composite merge, TTFT, alpha sweep,
    provisional promotion numbers. From services/edge/artifacts/round1_out/round_manifest.json,
    produced by `python -m edge.round` on a real RTX 2050 run (PR #9)."""
    data = _read_json(EDGE_ARTIFACTS / "round1_out" / "round_manifest.json")
    if data is None:
        raise HTTPException(404, "round1_out/round_manifest.json not found — run edge.round first")
    return data


@app.get("/api/edge/cluster-aggregate-manifest/{cluster_id}")
def edge_cluster_manifest(cluster_id: str) -> dict:
    """Real SVD-vs-naive reconstruction-error numbers from Adithyaa's real
    aggregation run (D2's justification), one per cluster."""
    data = _read_json(EDGE_ARTIFACTS / "clusters" / cluster_id / "aggregate_manifest.json")
    if data is None:
        raise HTTPException(404, f"no aggregate_manifest.json for cluster {cluster_id!r}")
    return data


# --------------------------------------------------------------------------- #
# Straight proxy to the REAL live services — no logic duplicated
# --------------------------------------------------------------------------- #
async def _proxy(base: str, path: str, request: Request) -> Response:
    url = f"{base}/{path}"
    async with _client(30.0) as client:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}
        try:
            resp = await client.request(
                request.method, url, params=request.query_params,
                content=body, headers=headers,
            )
        except httpx.ConnectError as e:
            raise HTTPException(
                502, f"{base} is not reachable — is it running? ({e})"
            ) from e
    return Response(
        content=resp.content, status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


@app.api_route("/api/cluster/{path:path}", methods=["GET", "POST", "PUT"])
async def proxy_cluster(path: str, request: Request) -> Response:
    return await _proxy(CLUSTER_BASE, path, request)


@app.api_route("/api/registry/{path:path}", methods=["GET", "POST"])
async def proxy_registry(path: str, request: Request) -> Response:
    return await _proxy(REGISTRY_BASE, path, request)


# --------------------------------------------------------------------------- #
# Demo orchestration — sequences real HTTP calls, computes nothing itself
# --------------------------------------------------------------------------- #
def _require_json(path: Path, what: str) -> dict:
    """`_read_json`, but a missing artifact fails as a clean, specific HTTP
    error instead of a TypeError 500 three lines later.

    /api/edge/clients already None-checks and reports what's missing; the
    demo actions need the same treatment, because the failure mode here is
    a blank 500 in the middle of a live panel demo rather than a message
    naming the file to go and look at.
    """
    data = _read_json(path)
    if data is None:
        raise HTTPException(
            503,
            f"{what} is missing at {path.relative_to(REPO_ROOT)} — the demo "
            f"reads real committed artifacts, so this client cannot be sent "
            f"to the cluster until that file exists",
        )
    return data


def _synthetic_tensors_for(cid: str) -> dict:
    """Shapes + hyperparams match this client's REAL adapter_config.json
    exactly (rank, target_modules, alpha). Tensor VALUES are synthetic --
    the real trained safetensors are gitignored and not present in this
    checkout. Labeled as such everywhere this appears in the UI."""
    cfg = _require_json(
        EDGE_ARTIFACTS / "round1" / cid / "adapter" / "adapter_config.json",
        f"{cid}'s adapter_config.json",
    )
    rank = cfg["r"]
    alpha = float(cfg["lora_alpha"])
    # Canonical order, not each client's own adapter_config.json key order --
    # real training runs list the same {q,k,v,o}_proj set in different JSON
    # orders, and cluster.aggregation.aggregate_svd compares target_modules as
    # an ordered tuple across clients, so a literal per-client order breaks
    # aggregation with "all client adapters must share target_modules" even
    # though the set is identical. The set, not the order, is what's real here.
    modules = [m for m in TARGET_MODULES_CANONICAL if m in cfg["target_modules"]]
    hidden = 64  # small stand-in width; real width (2048) is what OOM'd the CI box in PR #10's own tests
    rng = np.random.default_rng(abs(hash(cid)) % (2**32))
    tensors = []
    for name in modules:
        a = (rng.normal(0.0, 1.0 / rank, size=(rank, hidden))).astype(np.float32)
        b = rng.normal(0.0, 1.0 / hidden, size=(hidden, rank)).astype(np.float32)
        for part, arr in ((("lora_A"), a), (("lora_B"), b)):
            tensors.append({
                "name": f"layers.0.{name}.{part}.weight",
                "dtype": "float32",
                "shape": list(arr.shape),
                "data_b64": _b64(arr),
            })
    return {"rank": rank, "alpha": alpha, "target_modules": modules, "tensors": tensors}


def _b64(arr: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


@app.post("/api/demo/send-to-cluster/{cluster_id}")
async def demo_send_to_cluster(cluster_id: str) -> dict:
    """Seam A, live: POST one real /uploads call per client in this cluster,
    then a real /aggregate call. Exercises cluster's actual validation and
    aggregation, unchanged.

    Cluster now tracks each project cluster's upload buffer separately
    (multi-cluster HTTP service, landed on integration/panel) — every call
    below passes cluster_id explicitly rather than relying on the server's
    single default bucket, which is what made web/scientific mutually
    exclusive in this file's first version."""
    members = [c for c, cl in CLUSTER_OF.items() if cl == cluster_id]
    if not members:
        raise HTTPException(404, f"no clients mapped to cluster {cluster_id!r}")
    manifests = {
        c: _require_json(EDGE_ARTIFACTS / "round1" / c / "manifest.json", f"{c}'s manifest.json")
        for c in members
    }

    uploads = []
    async with _client(30.0) as client:
        # Cluster's real /uploads validates round_id against that cluster's
        # own counter (hardening added after this sprint's PR #10 review) --
        # ask it what round this specific cluster is on rather than assuming 0.
        health = await client.get(f"{CLUSTER_BASE}/healthz")
        clusters_health = health.json().get("clusters", {}) if health.status_code == 200 else {}
        current_round = clusters_health.get(cluster_id, {}).get("round_id", 0)

        for cid in members:
            synth = _synthetic_tensors_for(cid)
            weight = manifests[cid]["data"]["train"]["n_chunks"]
            payload = {
                "client_id": cid, "cluster_id": cluster_id, "round_id": current_round,
                "rank": synth["rank"], "target_modules": synth["target_modules"],
                "alpha": synth["alpha"], "num_layers": 1,
                "num_examples": weight, "tensors": synth["tensors"],
            }
            resp = await client.post(f"{CLUSTER_BASE}/uploads", json=payload)
            uploads.append({"client_id": cid, "status_code": resp.status_code, "body": resp.json()})
            if resp.status_code != 201:
                return {"uploads": uploads, "aggregate": None,
                        "error": f"upload for {cid} failed"}

        agg_resp = await client.post(
            f"{CLUSTER_BASE}/aggregate",
            json={"cluster_id": cluster_id, "aggregation": "svd", "include_manifest": True},
        )
        aggregate = {"status_code": agg_resp.status_code, "body": agg_resp.json()}

    return {
        "uploads": uploads, "aggregate": aggregate,
        "note": "tensor values are synthetic placeholders (real weights are gitignored "
                "and absent from this checkout); shapes/hyperparams match each client's "
                "real adapter_config.json; aggregation itself is the real "
                "cluster.aggregation code, unchanged.",
    }


@app.post("/api/demo/publish-to-registry/{cluster_id}")
async def demo_publish_to_registry(cluster_id: str) -> dict:
    """Seam B, live: cluster's own native POST /adapters/{cluster_id}/publish
    now does exactly what this endpoint used to hand-roll (fetch the active
    aggregate, embed adapter_config.json in the safetensors header, POST to
    the registry) — including the lora_dropout fix Adithyaa's review on #12
    caught in the e2e test's version of this reassembly. Multi-cluster state
    on the cluster server also means the cross-cluster mislabeling this
    function used to guard against by hand can no longer happen server-side:
    each cluster_id has its own buffer and its own active aggregate."""
    async with _client(30.0) as client:
        publish_resp = await client.post(
            f"{CLUSTER_BASE}/adapters/{cluster_id}/publish",
            json={"registry_url": REGISTRY_BASE, "set_active": True},
        )
    if publish_resp.status_code != 201:
        raise HTTPException(publish_resp.status_code, publish_resp.text)
    body = publish_resp.json()
    return {"status_code": publish_resp.status_code, "body": body["version"]}


@app.post("/api/demo/promote/{adapter_name}")
async def demo_promote(adapter_name: str) -> dict:
    """Seam C2, live: build a REAL EvalResult from the committed round manifest's
    actual measured numbers (not fabricated for the demo) and POST it to the
    real /promote endpoint — the D5 two-sided rule runs for real."""
    round_data = _read_json(EDGE_ARTIFACTS / "round1_out" / "round_manifest.json")
    if round_data is None:
        raise HTTPException(404, "round_manifest.json not found")

    async with _client(30.0) as client:
        active_resp = await client.get(f"{REGISTRY_BASE}/adapters/{adapter_name}/active")
        if active_resp.status_code != 200:
            raise HTTPException(502, f"registry has no active version for {adapter_name!r} yet")
        active = active_resp.json()
        version = active["ref"]["version"]

        # Real measured numbers from round1_out, picking one client on this cluster.
        candidate = next(
            (r for cid, r in round_data["clients"].items() if r["cluster"] == adapter_name),
            None,
        )
        if candidate is None:
            raise HTTPException(404, f"no round1 client result for cluster {adapter_name!r}")
        fs = candidate["full_split"]
        body = {
            "eval": {
                "adapter": {"name": adapter_name, "version": version, "kind": "cluster", "cluster_id": adapter_name},
                "in_project": {
                    "edit_similarity": 0.5, "exact_match": 0.3,
                    "perplexity": fs["composite_ppl"], "n_examples": fs["n_tokens"],
                },
                "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.30}}],
                "baseline_in_project": {
                    "edit_similarity": 0.45, "exact_match": 0.25,
                    "perplexity": fs["base_ppl"], "n_examples": fs["n_tokens"],
                },
                "baseline_noise_band": candidate["promotion"]["noise_band"],
                "seed": 0,
            },
            "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.31}}],
        }
        promote_resp = await client.post(f"{REGISTRY_BASE}/adapters/{adapter_name}/promote", json=body)
    return {
        "status_code": promote_resp.status_code,
        "body": promote_resp.json() if promote_resp.status_code == 200 else promote_resp.text,
        "note": "perplexity numbers are real (round1_out/round_manifest.json); "
                "edit_similarity/exact_match are placeholders since P5's in-project "
                "metric (B5) is not implemented yet -- HumanEval guard is a stand-in "
                "for the same reason (evalplus needs Unix `resource`, unavailable on Windows).",
    }


@app.post("/api/demo/demonstrate-rollback/{adapter_name}")
async def demo_demonstrate_rollback(adapter_name: str) -> dict:
    """Every real measured client in round1_out improved (see round_manifest.json
    -- all 6 personalization deltas are negative/good), so there is no genuine
    regression in this run to promote-and-reject. To show the D5 rule actually
    BLOCKING a bad promotion, not just approving good ones, this saves one more
    real version (same real aggregated bytes) and evaluates it against an
    intentionally regressed candidate. Labeled as such everywhere it appears."""
    async with _client(30.0) as client:
        # Duplicate whatever is CURRENTLY active, not a hardcoded v1 — after a
        # second publish the active version is v2, and duplicating v1's bytes
        # there would quietly stage a rollback demo on top of stale content
        # that no longer matches the metadata copied from `active` below.
        active_resp = await client.get(f"{REGISTRY_BASE}/adapters/{adapter_name}/active")
        if active_resp.status_code != 200:
            raise HTTPException(
                404, f"no active version for {adapter_name!r} to duplicate — publish it first"
            )
        active = active_resp.json()
        active_version = active["ref"]["version"]
        file_resp = await client.get(
            f"{REGISTRY_BASE}/adapters/{adapter_name}/versions/{active_version}/file"
        )
        if file_resp.status_code != 200:
            raise HTTPException(
                404, f"active version v{active_version} of {adapter_name!r} has no stored payload"
            )
        meta = {
            "kind": "cluster",
            "hparams": active["hparams"],
            "aggregation": active["aggregation"],
            "round": active["round"],
            "cluster_id": adapter_name,
            "source_clients": active["source_clients"],
            "set_active": True,
        }
        save_resp = await client.post(
            f"{REGISTRY_BASE}/adapters/{adapter_name}/versions",
            files={"file": ("adapter.safetensors", file_resp.content, "application/octet-stream")},
            data={"meta": json.dumps(meta)},
        )
        if save_resp.status_code != 201:
            raise HTTPException(save_resp.status_code, save_resp.text)
        new_version = save_resp.json()["ref"]["version"]

        body = {
            "eval": {
                "adapter": {"name": adapter_name, "version": new_version, "kind": "cluster", "cluster_id": adapter_name},
                "in_project": {"edit_similarity": 0.30, "exact_match": 0.10, "perplexity": 5.0, "n_examples": 40},
                "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.18}}],
                "baseline_in_project": {"edit_similarity": 0.45, "exact_match": 0.25, "perplexity": 3.4, "n_examples": 40},
                "baseline_noise_band": 0.01,
                "seed": 0,
            },
            "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.31}}],
        }
        promote_resp = await client.post(f"{REGISTRY_BASE}/adapters/{adapter_name}/promote", json=body)
    return {
        "status_code": promote_resp.status_code,
        "body": promote_resp.json() if promote_resp.status_code == 200 else promote_resp.text,
        "note": "intentionally regressed synthetic candidate — no real measured regression "
                "exists in this run's data (all 6 real clients improved); this exists purely "
                "to prove ROLLBACK is reachable, not only PROMOTE.",
    }


@app.get("/api/health")
async def health() -> dict:
    """Liveness of the two real backing services, for the pre-demo checklist."""
    out = {"cluster": "unreachable", "registry": "unreachable",
           "mtls": _VERIFY is not True}
    async with _client(3.0) as client:
        for name, base in (("cluster", CLUSTER_BASE), ("registry", REGISTRY_BASE)):
            try:
                r = await client.get(f"{base}/healthz")
                out[name] = "ok" if r.status_code == 200 else f"unhealthy ({r.status_code})"
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass
    return out


# --------------------------------------------------------------------------- #
# Live multi-laptop round (demo plan v3 items 7, 8) — the cluster-addressed
# routes, real uploads from the edge laptops, measured evaluation for Promote
# --------------------------------------------------------------------------- #
#: Which clients belong to which cluster for the live round (D1). Override with
#: CLASP_DEMO_CLUSTERS='{"web": ["client-flask", ...], ...}' when the laptops
#: present different clients.
LIVE_CLUSTERS: dict[str, list[str]] = (
    json.loads(os.environ["CLASP_DEMO_CLUSTERS"]) if os.environ.get("CLASP_DEMO_CLUSTERS")
    else {cl: [c for c in CLIENT_IDS if CLUSTER_OF[c] == cl] for cl in ("web", "scientific")}
)
#: Last aggregate / publish / promote per cluster, for the panel to redraw.
_live: dict[str, dict] = {}


def live_registry_name(cluster_id: str) -> str:
    """The registry adapter a cluster publishes to — the same name
    scripts/demo_round.py and evaluation's round feed use."""
    return f"cluster-{cluster_id}"


def _live_cluster(cluster_id: str) -> str:
    if cluster_id not in LIVE_CLUSTERS:
        raise HTTPException(404, f"{cluster_id!r} is not one of the demo's clusters "
                                 f"{sorted(LIVE_CLUSTERS)}")
    return cluster_id


async def _send(client: httpx.AsyncClient, method: str, url: str, **kw) -> httpx.Response:
    try:
        return await client.request(method, url, **kw)
    except httpx.TransportError as e:
        raise HTTPException(502, f"{url.split('/')[2]} is not reachable — is it running? ({e})") from e


def _detail(resp: httpx.Response):
    try:
        body = resp.json()
        return body.get("detail", body) if isinstance(body, dict) else body
    except ValueError:
        return resp.text


@app.get("/api/live/plan")
def live_plan() -> dict:
    return {"clusters": LIVE_CLUSTERS,
            "registry_names": {c: live_registry_name(c) for c in LIVE_CLUSTERS},
            "cluster_url": CLUSTER_BASE, "registry_url": REGISTRY_BASE, "mtls": _VERIFY is not True}


@app.post("/api/live/register")
async def live_register() -> dict:
    """Step 1: register both clusters and their members on the cluster service
    (PUT /clusters/{id}/members). Until this runs, edge uploads to
    /clusters/{id}/uploads answer 404."""
    out = {}
    async with _client(30.0) as client:
        for cid, members in LIVE_CLUSTERS.items():
            r = await _send(client, "PUT", f"{CLUSTER_BASE}/clusters/{cid}/members",
                            json={"client_ids": members})
            if r.status_code != 200:
                raise HTTPException(r.status_code, f"register {cid}: {_detail(r)}")
            out[cid] = r.json()["members"]
    return {"registered": out}


async def _registry_summary(client: httpx.AsyncClient, cluster_id: str) -> dict:
    name = live_registry_name(cluster_id)
    r = await _send(client, "GET", f"{REGISTRY_BASE}/adapters/{name}/versions")
    if r.status_code != 200:
        return {"name": name, "active": None, "versions": []}
    body = r.json()
    return {
        "name": name, "active": body.get("active"),
        "versions": [{
            "version": v["ref"]["version"], "round": v.get("round"),
            "aggregation": v.get("aggregation"), "source_clients": v.get("source_clients", []),
            "epsilon": (v.get("privacy") or {}).get("epsilon"),
            "sha256": v.get("sha256", "")[:12], "created_at": v.get("created_at"),
        } for v in body.get("versions", [])],
    }


@app.get("/api/live/state")
async def live_state() -> dict:
    """What the live panel draws: each cluster's round, members, who has
    uploaded, its registry versions, and the last action results."""
    async with _client(10.0) as client:
        r = await _send(client, "GET", f"{CLUSTER_BASE}/clusters")
        if r.status_code != 200:
            raise HTTPException(502, f"GET /clusters -> {r.status_code}: {_detail(r)}")
        listing = r.json()["clusters"]
        out = {}
        for cid in LIVE_CLUSTERS:
            out[cid] = {
                "planned_members": LIVE_CLUSTERS[cid],
                "registered": cid in listing,
                "cluster": listing.get(cid),
                "registry": await _registry_summary(client, cid),
                "last": _live.get(cid, {}),
            }
    return {"clusters": out}


@app.post("/api/live/aggregate/{cluster_id}")
async def live_aggregate(cluster_id: str) -> dict:
    """Step 3: SVD-aggregate whatever the edge laptops uploaded into this
    cluster's round (POST /clusters/{id}/aggregate). Returns the aggregation
    manifest, not the tensors."""
    cid = _live_cluster(cluster_id)
    async with _client(1800.0) as client:
        r = await _send(client, "POST", f"{CLUSTER_BASE}/clusters/{cid}/aggregate",
                        json={"aggregation": "svd", "include_manifest": True})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"aggregate {cid}: {_detail(r)}")
        m = await _send(client, "GET", f"{CLUSTER_BASE}/clusters/{cid}/aggregate/manifest")
    manifest = m.json()
    _live[cid] = {"aggregate": manifest}  # a new round: older publish/promote no longer apply
    return {"manifest": manifest}


@app.post("/api/live/publish/{cluster_id}")
async def live_publish(cluster_id: str) -> dict:
    """Step 4: the cluster publishes its active aggregate to the registry as a
    new version of cluster-<id> (seam B, the cluster's own /publish)."""
    cid = _live_cluster(cluster_id)
    async with _client(900.0) as client:
        r = await _send(client, "POST", f"{CLUSTER_BASE}/adapters/{cid}/publish",
                        json={"registry_url": REGISTRY_BASE, "adapter_name": live_registry_name(cid),
                              "set_active": True})
    if r.status_code != 201:
        raise HTTPException(r.status_code, f"publish {cid}: {_detail(r)}")
    v = r.json()["version"]
    summary = {"name": v["ref"]["name"], "version": v["ref"]["version"], "sha256": v["sha256"],
               "round": v.get("round"), "source_clients": v.get("source_clients", []),
               "epsilon": (v.get("privacy") or {}).get("epsilon")}
    _live.setdefault(cid, {})["publish"] = summary
    _live[cid].pop("promote", None)
    return {"version": summary}


@app.post("/api/live/promote/{cluster_id}")
async def live_promote(cluster_id: str) -> dict:
    """Step 5: D5 on the version just published, with evaluation's measured
    in-project metric, HumanEval guard and noise band (``evidence.py``)
    instead of constants. The registry decides PROMOTE or ROLLBACK."""
    from demo_ui.evidence import EvidenceMissing, evidence_for

    cid = _live_cluster(cluster_id)
    name = live_registry_name(cid)
    try:
        ev = evidence_for(cid)
    except EvidenceMissing as e:
        raise HTTPException(503, str(e)) from e
    async with _client(60.0) as client:
        reg = await _registry_summary(client, cid)
        if reg["active"] is None:
            raise HTTPException(409, f"{name} has no active version — publish first")
        if len(reg["versions"]) < 2:
            raise HTTPException(
                409, f"{name} has only v{reg['active']}: D5 compares a candidate against the "
                     f"version before it, so publish a second round first")
        body = {
            "eval": {
                "adapter": {"name": name, "version": reg["active"], "kind": "cluster",
                            "cluster_id": cid},
                "in_project": ev["in_project"],
                "guard": ev["guard"],
                "baseline_in_project": ev["baseline_in_project"],
                "baseline_noise_band": ev["noise_band"],
                "seed": 0,
            },
            "baseline_guard": ev["baseline_guard"],
        }
        r = await _send(client, "POST", f"{REGISTRY_BASE}/adapters/{name}/promote", json=body)
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"promote {name}: {_detail(r)}")
    sent = {k: body["eval"][k] for k in ("in_project", "baseline_in_project", "guard",
                                         "baseline_noise_band")}
    sent["baseline_guard"] = body["baseline_guard"]
    result = {"decision": r.json(), "candidate_version": reg["active"], "sent": sent,
              "provenance": ev["provenance"]}
    _live.setdefault(cid, {})["promote"] = result
    return result
