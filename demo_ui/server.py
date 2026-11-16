"""CLASP live round panel — thin orchestration layer, NOT a CLASP service.

Owner: P4 (Deepak). Lives outside services/ deliberately: it is the panel
laptop A shows during the multi-laptop demo (docs/multi-laptop-demo.md). It

  1. serves the live round page (``static/live.html``);
  2. drives the REAL cluster (:8002) and registry (:8004) through their own
     HTTP APIs — register members, aggregate, publish, promote — computing
     nothing itself (``/api/live/*``);
  3. sends Promote the numbers evaluation measured (``evidence.py``), never
     constants, and carries the registry's decision on the candidate back to
     the cluster the edges pull from (``candidate/promote`` or
     ``candidate/reject``): a published version is a candidate, a promotion
     makes it live, a rejected candidate is never served;
  4. proxies ``/api/cluster/*`` and ``/api/registry/*`` straight through, for
     inspection.

People log in (U5): with a users file configured (``CLASP_USERS_FILE``, written
by ``scripts/make_users.py``) every ``/api/*`` route except ``/api/health`` and
``/api/login`` needs ``Authorization: Bearer <token>`` for a user whose role is
``operator``. ``POST /api/login`` gives the token; it lasts 8 hours and is
signed with ``CLASP_JWT_SECRET`` (or the file ``CLASP_JWT_SECRET_FILE`` names;
with neither, a key made for this process). With no users file the panel runs
open, and says so when it starts (the fall-back, C11).

In front of every route (U4), in the order a request meets them: security
headers (a Content-Security-Policy that allows this origin and the page's
typeface, nothing inline), a 1 MiB body limit, a rate limit (600 requests a
minute per address in bursts of 60, ``CLASP_RATE_LIMIT_PER_MIN``;
``/api/health`` is exempt) and the audit recorder: every refused login or call
and every accepted action is a line in the audit trail (``CLASP_AUDIT_LOG``;
the console when it is not set), under the logged-in person's name.

The edge laptops upload their own trained adapters (``edge.webui``); nothing
here makes up tensors or metrics. The panel pages that did (synthetic
uploads, hardcoded promote metrics, a staged regression) were removed.

Run: uvicorn demo_ui.server:app --port 8010  (from the repo root, with
contracts + cluster + registry installed, and cluster/registry already
running on :8002/:8004) — or ``python scripts/run_services.py --ui``.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from security.audit import AccessAuditMiddleware, ServiceAudit
from security.auth.hmac_signer import SecretConfigError
from security.auth.jwt_handler import JWTConfig, TokenError, verify_token
from security.auth.users import (
    ROLE_OPERATOR,
    UserStore,
    UserStoreError,
    bearer_claims,
    issue_token,
    jwt_config_from_env,
)
from security.middleware import (
    RateLimiter,
    RateLimitMiddleware,
    RequestValidationMiddleware,
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
    rate_limit_from_env,
)

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

_log = logging.getLogger("demo_ui")

# --------------------------------------------------------------------------- #
# Logins (U5)
# --------------------------------------------------------------------------- #
USERS_ENV = "CLASP_USERS_FILE"
#: ``/api`` routes that need no token: liveness, and the login itself.
OPEN_API_PATHS = frozenset({"/api/health", "/api/login"})
_login_cache: dict = {}


def _login_settings() -> tuple[UserStore, JWTConfig] | None:
    """The users and token settings, or ``None`` when no users file is
    configured (the page then runs open). Raises ``UserStoreError`` /
    ``SecretConfigError`` when what is configured cannot be used.

    The users file is re-read when it changes; the signing key is kept for as
    long as its configuration is the same, so tokens survive a users edit.
    """
    path = os.environ.get(USERS_ENV)
    if not path:
        return None
    key = (os.environ.get("CLASP_JWT_SECRET"), os.environ.get("CLASP_JWT_SECRET_FILE"))
    if _login_cache.get("jwt_key") != key:
        config, generated = jwt_config_from_env()
        if generated:
            _log.warning("CLASP_JWT_SECRET is not set: logins are signed with a key made for "
                         "this process and end when it stops")
        _login_cache.update(jwt_key=key, jwt=config)
    try:
        stat = os.stat(path)
        stamp = (path, stat.st_mtime_ns, stat.st_size)
    except OSError as e:
        raise UserStoreError(f"{USERS_ENV}={path} cannot be read ({type(e).__name__})") from e
    if _login_cache.get("stamp") != stamp:
        _login_cache.update(stamp=stamp, users=UserStore(Path(path)))
    return _login_cache["users"], _login_cache["jwt"]


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=401,
                        headers={"WWW-Authenticate": "Bearer"})


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if not os.environ.get(USERS_ENV):
        _log.warning("%s is not set: the panel runs without a login; anyone who can reach "
                     "this port can drive the round (scripts/make_users.py writes a users file)",
                     USERS_ENV)
    yield


app = FastAPI(title="CLASP Panel Demo UI", lifespan=_lifespan)


# --------------------------------------------------------------------------- #
# In front of every route (U4)
# --------------------------------------------------------------------------- #
#: The panel takes small JSON bodies only (a login); nothing is uploaded here.
MAX_BODY_BYTES = 1024 * 1024
rate_limiter = RateLimiter(rate_limit_from_env())
audit_trail = ServiceAudit("demo-ui")
#: The library's headers, with a policy this page can live with: everything
#: from this origin, plus the typeface it loads from Google Fonts. No inline
#: script or style is allowed, and the page has none (static/live.js).
PAGE_HEADERS = SecurityHeadersConfig(content_security_policy=(
    "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; frame-ancestors 'none'; base-uri 'self'"))


def _person(request: Request) -> dict | None:
    """The claims of the caller's login token, or ``None`` (no login configured,
    no token, or a token that does not verify)."""
    try:
        settings = _login_settings()
        return bearer_claims(request.headers.get("authorization"), settings[1]) \
            if settings else None
    except (TokenError, UserStoreError, SecretConfigError):
        return None


def _audit_identity(request: Request) -> str | None:
    return (_person(request) or {}).get("sub")


def _audit_role(request: Request, _name: str | None) -> str:
    claims = _person(request)
    if claims:
        return str(claims.get("role"))
    return "anonymous" if os.environ.get(USERS_ENV) else "open"


@app.middleware("http")
async def require_login(request: Request, call_next):
    """Every ``/api/*`` call except health and login needs an operator's token."""
    path = request.url.path
    if not path.startswith("/api/") or path in OPEN_API_PATHS:
        return await call_next(request)
    try:
        settings = _login_settings()
    except (UserStoreError, SecretConfigError) as e:
        return _unauthorized(f"logins are misconfigured: {e}")
    if settings is None:
        return await call_next(request)
    try:
        claims = bearer_claims(request.headers.get("authorization"), settings[1])
    except TokenError as e:
        return _unauthorized(str(e))
    if claims.get("role") != ROLE_OPERATOR:
        return JSONResponse(
            {"detail": f"this panel is for the operator role; {claims.get('sub')!r} is "
                       f"{claims.get('role')!r}"}, status_code=403)
    return await call_next(request)


# Added innermost first (after require_login above, which therefore runs last):
# a request meets the security headers, the body limit, the rate limit, the
# audit recorder, then the login check and the route.
app.add_middleware(AccessAuditMiddleware, audit=audit_trail.logger, identity=_audit_identity,
                   role=_audit_role, service="demo-ui", exempt_paths=("/api/health",))
app.add_middleware(RateLimitMiddleware, limiter=rate_limiter, exempt_paths=("/api/health",))
app.add_middleware(RequestValidationMiddleware, max_body_bytes=MAX_BODY_BYTES)
app.add_middleware(SecurityHeadersMiddleware, config=PAGE_HEADERS)


@app.post("/api/login")
async def login(request: Request) -> dict:
    """``{"username", "password"}`` -> ``{"token", "role", "expires_at"}``, or 401."""
    try:
        settings = _login_settings()
    except (UserStoreError, SecretConfigError) as e:
        raise HTTPException(500, f"logins are misconfigured: {e}") from e
    if settings is None:
        raise HTTPException(404, "logins are not enabled on this page (no users file)")
    users, config = settings
    try:
        body = await request.json()
    except ValueError:
        body = {}
    username, password = (body or {}).get("username"), (body or {}).get("password")
    token = (issue_token(users, username, password, config)
             if isinstance(username, str) and isinstance(password, str) else None)
    if token is None:
        raise HTTPException(401, "wrong username or password")
    claims = verify_token(token, config)
    expires = datetime.datetime.fromtimestamp(claims["exp"], datetime.timezone.utc)
    return {"token": token, "role": claims["role"], "expires_at": expires.isoformat()}


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
           "mtls": _VERIFY is not True, "login_required": bool(os.environ.get(USERS_ENV))}
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
#: The evaluation of the published candidate (scripts/evaluate_candidate.py on
#: this laptop's GPU), one subprocess per cluster at a time.
EVALUATE_CMD = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate_candidate.py")]
_eval_jobs: dict[str, dict] = {}
_eval_lock = threading.Lock()


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
        return {"name": name, "active": None, "candidate": None, "versions": []}
    body = r.json()
    # The registry owns the rules: ``candidate`` is the newest undecided version
    # above the active one (what Evaluate and Promote judge), ``rejected`` every
    # version it refused (a REJECT, or a D5 ROLLBACK from an older volume).
    rejected = set(body.get("rejected") or ())
    return {
        "name": name, "active": body.get("active"),
        "candidate": body.get("candidate"),
        "versions": [{
            "version": v["ref"]["version"], "round": v.get("round"),
            "aggregation": v.get("aggregation"), "source_clients": v.get("source_clients", []),
            "epsilon": (v.get("privacy") or {}).get("epsilon"),
            "sha256": v.get("sha256", "")[:12], "digest": v.get("sha256"),
            "created_at": v.get("created_at"),
            "rejected": v["ref"]["version"] in rejected,
        } for v in body.get("versions", [])],
    }


async def _privacy_summary(client: httpx.AsyncClient, cluster_id: str) -> dict:
    """D7: what each of the cluster's clients has spent of its run's privacy
    budget, from the registry's ledger (``GET /privacy/ledger``): the latest
    row per client. Never raises: a registry that cannot answer costs the
    page these lines, with the reason under ``error``."""
    out: dict = {"budget": None, "rounds_total": None, "clients": [], "error": None}
    try:
        r = await _send(client, "GET", f"{REGISTRY_BASE}/privacy/ledger",
                        params={"cluster_id": cluster_id})
        if r.status_code != 200:
            out["error"] = _step_error("registry GET /privacy/ledger", resp=r)
            return out
        body = r.json()
        latest: dict[str, dict] = {}
        for row in body.get("rows", []):  # oldest first: a later row replaces an earlier one
            latest[row["client_id"]] = {
                "client_id": row["client_id"],
                "epsilon_cumulative": row["epsilon_cumulative"],
                "rounds_accounted": row.get("rounds_accounted"),
            }
        out.update(budget=body.get("budget"), rounds_total=body.get("rounds_total"),
                   clients=sorted(latest.values(), key=lambda c: c["client_id"]))
    except (HTTPException, httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        out["error"] = _step_error("registry GET /privacy/ledger", exc=e)
    return out


#: Only one reconciliation runs at a time, so two page polls (or a poll and a
#: Promote) never send the same cluster step twice.
_reconciling = False


def _step_error(what: str, resp: httpx.Response | None = None, exc: Exception | None = None) -> str:
    if resp is not None:
        return f"{what} -> {resp.status_code}: {_detail(resp)}"
    detail = exc.detail if isinstance(exc, HTTPException) else exc
    return f"{what}: {detail}"


def _already_done(row: dict, cluster: dict | None) -> bool:
    """A cluster 409 means the step is already in effect when, for a promote,
    the cluster serves the row's sha256, or, for a reject, it has no pending
    candidate or its pending candidate is another one."""
    if cluster is None:
        return False
    if row["action"] == "promote":
        return cluster.get("active_sha256") == row["sha256"]
    return cluster.get("candidate_sha256") in (None, "") or cluster["candidate_sha256"] != row["sha256"]


async def _apply_cluster_step(client: httpx.AsyncClient, row: dict) -> tuple[bool, dict]:
    """Send one row's step to its cluster: (landed, result or ``{"error": ...}``).
    An error with ``conflict`` is a 409 that does not show the step as in effect."""
    what = f"cluster candidate/{row['action']}"
    try:
        cr = await _send(client, "POST",
                         f"{CLUSTER_BASE}/clusters/{row['cluster_id']}/adapters/candidate/{row['action']}",
                         json={"sha256": row["sha256"]})
        if cr.status_code == 200:
            return True, cr.json()
        if cr.status_code == 409:
            listing = await _send(client, "GET", f"{CLUSTER_BASE}/clusters")
            cluster = (listing.json().get("clusters") or {}).get(row["cluster_id"]) \
                if listing.status_code == 200 else None
            if _already_done(row, cluster):
                return True, {"already_applied": True,
                              "active_sha256": (cluster or {}).get("active_sha256")}
            return False, {"error": _step_error(what, resp=cr), "conflict": True}
        return False, {"error": _step_error(what, resp=cr)}
    except (HTTPException, httpx.HTTPError, ValueError) as e:
        return False, {"error": _step_error(what, exc=e)}


async def _report_sync(client: httpx.AsyncClient, row: dict, landed: bool,
                       error: str | None) -> str | None:
    """``POST /sync/{id}/done``; the error text when the registry could not be told."""
    try:
        rep = await _send(client, "POST", f"{REGISTRY_BASE}/sync/{row['id']}/done",
                          json={"ok": landed, "error": error})
        if rep.status_code != 200:
            return _step_error("registry POST /sync/done", resp=rep)
    except (HTTPException, httpx.HTTPError) as e:
        return _step_error("registry POST /sync/done", exc=e)
    return None


async def _reconcile_cluster_sync(client: httpx.AsyncClient) -> dict:
    """Apply the registry's pending cluster steps (``GET /sync/pending``) and
    report each outcome (``POST /sync/{id}/done``). Never raises: a step that
    does not land stays pending in the registry, and is returned here.

    Rows apply per cluster in id order; once one stays pending the cluster's
    later rows wait for the next pass. A ``promote`` the cluster answers 409 to
    (it is neither serving that sha nor holding it as its candidate) is
    superseded, and marked done, when a later row of the cluster is already done
    or lands in this pass.

    Returns ``{"pending": [rows still open, with attempts and last_error],
    "results": {sync id: {adapter, version, landed, result}}, "error": why the
    list itself was unavailable}``.
    """
    global _reconciling
    while _reconciling:
        await asyncio.sleep(0.05)
    _reconciling = True
    try:
        results: dict[int, dict] = {}

        def record(row: dict, landed: bool, result: dict) -> None:
            results[row["id"]] = {"adapter": row["adapter"], "version": row["version"],
                                  "landed": landed, "result": result}

        try:
            r = await _send(client, "GET", f"{REGISTRY_BASE}/sync/pending")
            if r.status_code != 200:
                return {"pending": [], "results": results,
                        "error": _step_error("registry GET /sync/pending", resp=r)}
            rows = sorted(r.json()["pending"], key=lambda x: x["id"])
        except (HTTPException, httpx.HTTPError, ValueError, KeyError) as e:
            return {"pending": [], "results": results,
                    "error": _step_error("registry GET /sync/pending", exc=e)}
        still: list[dict] = []
        blocked: set[str] = set()                 # clusters with a row left pending
        held: dict[str, list[tuple[dict, dict]]] = {}  # 409 promotes awaiting a later row

        async def finish(row: dict, landed: bool, result: dict, error: str | None) -> None:
            lost = await _report_sync(client, row, landed, error)
            if lost:
                landed, result, error = False, {"error": lost}, lost
            record(row, landed, result)
            if not landed:
                still.append({**row, "attempts": row["attempts"] + 1, "last_error": error})
                blocked.add(row["cluster_id"])

        for row in rows:
            cid = row["cluster_id"]
            if cid in blocked:
                still.append(row)  # waits for the earlier row
                continue
            landed, result = await _apply_cluster_step(client, row)
            if not landed and result.get("conflict") and row["action"] == "promote":
                if row.get("superseded_by") is not None:
                    await finish(row, True, {"superseded_by": row["superseded_by"]},
                                 f"superseded by sync {row['superseded_by']}")
                else:
                    held.setdefault(cid, []).append((row, result))
                continue
            await finish(row, landed, result, None if landed else result["error"])
            if landed:
                for old, _ in held.pop(cid, []):
                    await finish(old, True, {"superseded_by": row["id"]},
                                 f"superseded by sync {row['id']}")
        for pend in held.values():  # nothing later landed: they stay pending
            for old, result in pend:
                await finish(old, False, result, result["error"])
        still.sort(key=lambda x: x["id"])
        return {"pending": still, "results": results, "error": None}
    finally:
        _reconciling = False


@app.get("/api/live/state")
async def live_state() -> dict:
    """What the live panel draws: each cluster's round, members, who has
    uploaded, its registry versions, and the last action results. First it
    finishes any cluster promote/reject the registry still has open, so a
    restarted panel completes the job; a step that does not land is reported
    under ``cluster_sync`` (and per cluster), never raised."""
    async with _client(10.0) as client:
        sync = await _reconcile_cluster_sync(client)
        r = await _send(client, "GET", f"{CLUSTER_BASE}/clusters")
        if r.status_code != 200:
            raise HTTPException(502, f"GET /clusters -> {r.status_code}: {_detail(r)}")
        listing = r.json()["clusters"]
        out = {}
        for cid in LIVE_CLUSTERS:
            reg = await _registry_summary(client, cid)
            cluster = listing.get(cid)
            out[cid] = {
                "planned_members": LIVE_CLUSTERS[cid],
                "registered": cid in listing,
                "cluster": cluster,
                # the aggregate awaiting evaluation and promote / reject
                "candidate_sha256": (cluster or {}).get("candidate_sha256"),
                "registry": reg,
                # D7: each client's spend of its run's budget, as the registry recorded it
                "privacy": await _privacy_summary(client, cid),
                "last": _live.get(cid, {}),
                "cluster_sync": [p for p in sync["pending"] if p["cluster_id"] == cid],
                "evaluation": _eval_snapshot(cid, tail=4),
                "evaluated_version": _evaluated(cid, reg),
            }
    return {"clusters": out, "cluster_sync": sync["pending"],
            "cluster_sync_error": sync["error"]}


def _digest(reg: dict, version: int | None) -> str | None:
    return next((v["digest"] for v in reg["versions"] if v["version"] == version), None)


def _evaluated(cluster_id: str, reg: dict) -> int | None:
    """The pending candidate's version, if it has its own evaluation (version
    and sha256)."""
    from demo_ui.evidence import evaluated_version

    cand = reg.get("candidate")
    if cand is None:
        return None
    return cand if evaluated_version(cluster_id, cand, _digest(reg, cand)) else None


def _eval_snapshot(cluster_id: str, tail: int | None = None) -> dict:
    with _eval_lock:
        job = dict(_eval_jobs.get(cluster_id) or {"running": False, "exit_code": None,
                                                  "lines": []})
        job["lines"] = list(job["lines"][-tail:] if tail else job["lines"])
    return job


@app.post("/api/live/evaluate/{cluster_id}", status_code=202)
def live_evaluate(cluster_id: str) -> dict:
    """Step 4b: score the pending candidate and the version the registry
    serves (the bare base model when nothing is active yet) on this laptop's
    GPU (scripts/evaluate_candidate.py), so Promote judges the candidate on its
    own measurements. Runs in the background; poll GET."""
    from demo_ui import evidence

    cid = _live_cluster(cluster_id)
    cmd = [*EVALUATE_CMD, "--cluster", cid, "--registry-url", REGISTRY_BASE,
           "--registry-name", live_registry_name(cid), "--out-dir", str(evidence.LIVE_EVAL_DIR)]
    with _eval_lock:
        if (_eval_jobs.get(cid) or {}).get("running"):
            raise HTTPException(409, f"{cid} is already being evaluated")
        job = _eval_jobs[cid] = {"running": True, "exit_code": None, "lines": [],
                                 "started": time.time(), "command": cmd[1:]}
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))

    def run() -> None:
        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace")
            for line in proc.stdout:
                with _eval_lock:
                    job["lines"].append(line.rstrip())
                    del job["lines"][:-300]
            code = proc.wait()
        except OSError as e:
            with _eval_lock:
                job["lines"].append(f"could not start the evaluation: {e}")
            code = -1
        with _eval_lock:
            job["running"], job["exit_code"] = False, code

    threading.Thread(target=run, daemon=True).start()
    return {"started": cid, "command": cmd[1:]}


@app.get("/api/live/evaluate/{cluster_id}")
def live_evaluate_state(cluster_id: str) -> dict:
    return _eval_snapshot(_live_cluster(cluster_id))


@app.post("/api/live/aggregate/{cluster_id}")
async def live_aggregate(cluster_id: str) -> dict:
    """Step 3: SVD-aggregate whatever the edge laptops uploaded into this
    cluster's round (POST /clusters/{id}/aggregate). The result is the
    cluster's candidate; edges keep pulling the active adapter until it is
    promoted. Returns the aggregation manifest, not the tensors."""
    cid = _live_cluster(cluster_id)
    async with _client(1800.0) as client:
        r = await _send(client, "POST", f"{CLUSTER_BASE}/clusters/{cid}/aggregate",
                        json={"aggregation": "svd", "include_manifest": True})
        if r.status_code != 200:
            detail = _detail(r)
            if isinstance(detail, dict) and detail.get("error") == "candidate_pending":
                raise HTTPException(409, (
                    f"{cid} already has a merge waiting for the registry's decision "
                    f"({str(detail.get('sha256') or '?')[:12]}); publish and decide it "
                    "(promote or reject) before merging again."))
            raise HTTPException(r.status_code, f"aggregate {cid}: {detail}")
        m = await _send(client, "GET", f"{CLUSTER_BASE}/clusters/{cid}/aggregate/manifest")
    manifest = m.json()
    _live[cid] = {"aggregate": manifest}  # a new round: older publish/promote no longer apply
    return {"manifest": manifest}


@app.post("/api/live/publish/{cluster_id}")
async def live_publish(cluster_id: str) -> dict:
    """Step 4: the cluster publishes its candidate aggregate to the registry as
    a new, inactive version of cluster-<id> (seam B, the cluster's own
    /publish). Nothing is served from it until Promote."""
    cid = _live_cluster(cluster_id)
    async with _client(900.0) as client:
        r = await _send(client, "POST", f"{CLUSTER_BASE}/adapters/{cid}/publish",
                        json={"registry_url": REGISTRY_BASE, "adapter_name": live_registry_name(cid)})
    if r.status_code != 201:
        raise HTTPException(r.status_code, f"publish {cid}: {_detail(r)}")
    v = r.json()["version"]
    summary = {"name": v["ref"]["name"], "version": v["ref"]["version"], "sha256": v["sha256"],
               "round": v.get("round"), "source_clients": v.get("source_clients", []),
               "epsilon": (v.get("privacy") or {}).get("epsilon")}
    _live.setdefault(cid, {})["publish"] = summary
    _live[cid].pop("promote", None)
    return {"version": summary}


def _cluster_outcome(sync: dict, sync_id: int | None) -> dict:
    """What happened to the cluster step recorded with this decision
    (``sync_id``): the cluster's answer, ``applied`` when an earlier
    reconciliation already finished it, or ``error`` (with ``pending``) while it
    has not landed."""
    if sync_id is None:
        return {"error": "the registry recorded no cluster step for this decision"}
    if sync["error"]:
        return {"pending": True, "error": f"{sync['error']} (cluster not yet updated; will retry)"}
    entry = sync["results"].get(sync_id)
    if entry is not None:
        if entry["landed"]:
            return entry["result"]
        return {"pending": True,
                "error": f"{entry['result']['error']} (cluster not yet updated; will retry)"}
    for row in sync["pending"]:
        if row["id"] == sync_id:  # waiting behind an earlier step of the cluster
            why = row.get("last_error") or "waiting for an earlier cluster step"
            return {"pending": True, "error": f"{why} (cluster not yet updated; will retry)"}
    return {"applied": True}  # rows leave pending only when done: a state poll got there first


@app.post("/api/live/promote/{cluster_id}")
async def live_promote(cluster_id: str) -> dict:
    """Step 5: D5 on the pending candidate, with evaluation's measured
    in-project metric, line-completion guard and noise band (``evidence.py``)
    instead of constants, in the body the evaluator signed (U4). The registry
    verifies the signature and decides PROMOTE or REJECT; the cluster
    then promotes or rejects its candidate (matched by the registry's sha256
    for that version), so the edges pull a merge only once it is promoted."""
    from demo_ui.evidence import EvidenceMissing, evidence_for

    cid = _live_cluster(cluster_id)
    name = live_registry_name(cid)
    async with _client(60.0) as client:
        reg = await _registry_summary(client, cid)
        cand = reg["candidate"]
        if cand is None:
            raise HTTPException(409, f"{name} has no candidate above the active version — "
                                     f"aggregate and publish first")
        digest = _digest(reg, cand)
        try:
            # the candidate's own measurements, matched by version and sha256
            ev = evidence_for(cid, version=cand, sha256=digest)
        except EvidenceMissing as e:
            raise HTTPException(503, str(e)) from e
        signed = ev.get("promote")
        if signed:
            # U4: the evaluator built and signed this body; it goes to the
            # registry exactly as written, so the signature still matches
            body = {"eval": signed["eval"], "baseline_guard": signed["baseline_guard"],
                    "candidate_sha256": signed["candidate_sha256"]}
            if signed.get("eval_signature"):
                body["eval_signature"] = signed["eval_signature"]
        else:
            body = {
                "eval": {
                    "adapter": {"name": name, "version": cand, "kind": "cluster",
                                "cluster_id": cid},
                    "in_project": ev["in_project"],
                    "guard": ev["guard"],
                    "baseline_in_project": ev["baseline_in_project"],
                    "baseline_noise_band": ev["noise_band"],
                    "seed": 0,
                },
                "baseline_guard": ev["baseline_guard"],
                # the bytes this evidence was matched to (version and sha256)
                "candidate_sha256": digest,
            }
        r = await _send(client, "POST", f"{REGISTRY_BASE}/adapters/{name}/promote", json=body)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"promote {name}: {_detail(r)}")
        decision = r.json()
        step = "promote" if decision.get("action") == "promote" else "reject"
        # the registry recorded the cluster's step with its decision; apply it
        # now, and if it does not land the next state call retries it
        sync = await _reconcile_cluster_sync(client)
    cluster_result = _cluster_outcome(sync, decision.get("cluster_sync_id"))
    if cluster_result.get("applied"):
        async with _client(10.0) as client:  # show what the cluster serves now
            try:
                lst = await _send(client, "GET", f"{CLUSTER_BASE}/clusters")
                cluster_result["active_sha256"] = lst.json()["clusters"][cid]["active_sha256"]
            except (HTTPException, httpx.HTTPError, ValueError, KeyError):
                pass
    sent = {k: body["eval"][k] for k in ("in_project", "baseline_in_project", "guard",
                                         "baseline_noise_band")}
    sent["baseline_guard"] = body["baseline_guard"]
    result = {"decision": decision, "candidate_version": cand, "sent": sent,
              "provenance": ev["provenance"], "cluster": {"step": step, **cluster_result}}
    _live.setdefault(cid, {})["promote"] = result
    return result
