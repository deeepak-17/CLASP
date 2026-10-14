"""CLASP live round panel — thin orchestration layer, NOT a CLASP service.

Owner: P4 (Deepak). Lives outside services/ deliberately: it is the panel
laptop A shows during the multi-laptop demo (docs/multi-laptop-demo.md). It

  1. serves the live round page (``static/live.html``);
  2. drives the REAL cluster (:8002) and registry (:8004) through their own
     HTTP APIs — register members, aggregate, publish, promote — computing
     nothing itself (``/api/live/*``);
  3. sends Promote the numbers evaluation measured (``evidence.py``), never
     constants, and carries a ROLLBACK back to the cluster the edges pull from;
  4. proxies ``/api/cluster/*`` and ``/api/registry/*`` straight through, for
     inspection.

The edge laptops upload their own trained adapters (``edge.webui``); nothing
here makes up tensors or metrics. The panel pages that did (synthetic
uploads, hardcoded promote metrics, a staged regression) were removed.

Run: uvicorn demo_ui.server:app --port 8010  (from the repo root, with
contracts + cluster + registry installed, and cluster/registry already
running on :8002/:8004) — or ``python scripts/run_services.py --ui``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    """The live multi-laptop round (demo plan v3)."""
    return FileResponse(STATIC_DIR / "live.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
        decision = r.json()
        restore = None
        if decision.get("action") == "rollback":
            # The registry repointed `active`; the edges pull from the cluster,
            # so it must serve the restored version too, or the next round
            # would build on the merge D5 just rejected.
            rr = await _send(client, "POST", f"{CLUSTER_BASE}/clusters/{cid}/adapters/active/restore",
                             json={"registry_url": REGISTRY_BASE, "adapter_name": name,
                                   "version": decision["active_version_after"]})
            restore = rr.json() if rr.status_code == 200 else {
                "error": f"cluster restore -> {rr.status_code}: {_detail(rr)}"}
    sent = {k: body["eval"][k] for k in ("in_project", "baseline_in_project", "guard",
                                         "baseline_noise_band")}
    sent["baseline_guard"] = body["baseline_guard"]
    result = {"decision": decision, "candidate_version": reg["active"], "sent": sent,
              "provenance": ev["provenance"]}
    if restore is not None:
        result["cluster_restore"] = restore
    _live.setdefault(cid, {})["promote"] = result
    return result
