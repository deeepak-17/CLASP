"""CLASP State Registry — FastAPI app (P4, Deepak).

Endpoints:

    GET  /healthz                                  liveness
    GET  /adapters                                 list adapter names
    POST /adapters/{name}/versions                 save a new immutable version
    GET  /adapters/{name}/versions                 list versions + metadata
    GET  /adapters/{name}/versions/{v}             metadata for one version
    GET  /adapters/{name}/versions/{v}/file        download the safetensors blob
    GET  /adapters/{name}/active                   metadata for the active version
    GET  /adapters/{name}/lineage                  versions + parents + decisions
    POST /adapters/{name}/compose                  store a pre-merged D6 composite
    POST /adapters/{name}/promote                  D5 two-sided rule on the active version
                                                   (+ optional build-on-promote composite)
    POST /adapters/{name}/restore                  operator restore / rollback drill
    POST /adapters/{name}/gc                       retention: keep last N + ever-live
    POST /gc                                       retention across every adapter
    GET  /adapters/{name}/promotions               promotion/rollback audit trail

Every mutating route runs under one process-wide write lock: FastAPI serves
sync routes from a threadpool, and without it two concurrent saves race for
the same version number (one gets a spurious 409) and a GC can delete the
version a concurrent restore is pointing at. The service runs one worker.

Storage errors map to HTTP status in one place (see the exception handlers):
unknown adapter/version -> 404, bad name -> 422, version race -> 409, and a
corrupt on-disk record -> a clean 500 with the reason, never a traceback.
"""
from __future__ import annotations

import json
import os
import threading

from contracts import (
    CONTRACTS_VERSION,
    AdapterKind,
    AggregationMethod,
    CompositeProvenance,
    EvalResult,
    GuardMetrics,
    LoRAHyperParams,
    PrivacySpec,
    PromotionAction,
)
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .composition import (
    CompositionError,
    PlannedComposite,
    parse_compose_request,
    plan_composite,
    store_composite,
)
from .lineage import build_lineage
from .promotion import decide, restore_decision
from .retention import RetentionPlan, collect_referenced, default_keep_last, gc_adapter
from .storage import (
    AdapterNotFound,
    InvalidAdapterName,
    KindMismatch,
    RegistryStore,
    StorageError,
    VersionExists,
    _metadata_to_dict,
    _promotion_decision_to_dict,
)

app = FastAPI(
    title="CLASP State Registry",
    version=__version__,
    summary="Versioned safetensors adapter storage with two-sided promotion (P4).",
)

#: Uploads above this are refused with 413 before they reach storage.
#: A 6.7B-class rank-16 adapter is ~100 MB; 1 GiB leaves ample headroom.
MAX_UPLOAD_ENV = "CLASP_MAX_UPLOAD_BYTES"
DEFAULT_MAX_UPLOAD = 1024 * 1024 * 1024

_store: RegistryStore | None = None
_WRITE_LOCK = threading.RLock()


def get_store() -> RegistryStore:
    """Lazily construct the store so CLASP_REGISTRY_DATA is read at request time."""
    global _store
    if _store is None:
        _store = RegistryStore()
    return _store


# --------------------------------------------------------------------------- #
# storage error -> HTTP status, in one place
# --------------------------------------------------------------------------- #
@app.exception_handler(AdapterNotFound)
async def _not_found(_: Request, exc: AdapterNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})


@app.exception_handler(InvalidAdapterName)
async def _bad_name(_: Request, exc: InvalidAdapterName) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(VersionExists)
async def _version_race(_: Request, exc: VersionExists) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(KindMismatch)
async def _kind_conflict(_: Request, exc: KindMismatch) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(StorageError)
async def _data_error(_: Request, exc: StorageError) -> JSONResponse:
    # L1: e.g. a corrupt/hand-edited `active` pointer — clean 500, not a crash.
    return JSONResponse(status_code=500, content={"detail": f"registry data error: {exc}"})


def _max_upload() -> int:
    try:
        return int(os.environ.get(MAX_UPLOAD_ENV, DEFAULT_MAX_UPLOAD))
    except ValueError as e:
        raise HTTPException(500, f"registry misconfigured: {MAX_UPLOAD_ENV}: {e}") from e


def _read_capped(file: UploadFile) -> bytes:
    limit = _max_upload()
    payload = file.file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(413, f"payload exceeds {limit} bytes ({MAX_UPLOAD_ENV})")
    return payload


def _parse_save_meta(raw: str) -> dict:
    """Decode the save envelope into ``RegistryStore.save`` keyword arguments."""
    try:
        m = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"meta is not valid JSON: {e}") from e
    if not isinstance(m, dict):
        raise HTTPException(422, "meta must be a JSON object")
    try:
        composed = m.get("composed_from")
        if not isinstance(m.get("set_active", True), bool):
            raise ValueError("set_active must be a boolean")
        return {
            "kind": AdapterKind(m.get("kind", "client")),
            "hparams": LoRAHyperParams.from_json(m.get("hparams")),
            "privacy": PrivacySpec.from_json(m["privacy"]) if m.get("privacy") else None,
            "aggregation": AggregationMethod(m["aggregation"]) if m.get("aggregation") else None,
            "round": m.get("round"),
            "seed": m.get("seed", 0),
            "cluster_id": m.get("cluster_id"),
            "source_clients": tuple(m.get("source_clients", ())),
            "set_active": m.get("set_active", True),
            "composed_from": CompositeProvenance.from_json(composed) if composed else None,
        }
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"invalid meta: {e}") from e


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "registry", "contracts": CONTRACTS_VERSION}


@app.get("/adapters")
def list_adapters() -> dict:
    return {"adapters": get_store().list_adapters()}


@app.post("/adapters/{name}/versions", status_code=201)
def save_version(
    name: str,
    file: UploadFile = File(..., description="safetensors payload"),
    meta: str = Form("{}", description="JSON metadata envelope"),
) -> dict:
    """Write a new immutable version and return its metadata."""
    kwargs = _parse_save_meta(meta)
    payload = _read_capped(file)
    try:
        with _WRITE_LOCK:
            written = get_store().save(name, payload, **kwargs)
    except (VersionExists, InvalidAdapterName, KindMismatch):  # mapped by the handlers
        raise
    except StorageError as e:  # bad payload / inconsistent envelope: client error
        raise HTTPException(422, str(e)) from e
    return _metadata_to_dict(written)


@app.get("/adapters/{name}/versions")
def list_versions(name: str) -> dict:
    store = get_store()
    versions = store.list_versions(name)
    return {
        "name": name,
        "active": store.get_active(name),
        "versions": [_metadata_to_dict(store.get_metadata(name, v)) for v in versions],
    }


@app.get("/adapters/{name}/versions/{version}")
def get_version(name: str, version: int) -> dict:
    return _metadata_to_dict(get_store().get_metadata(name, version))


@app.get("/adapters/{name}/versions/{version}/file")
def download_version(name: str, version: int) -> Response:
    payload = get_store().load_payload(name, version)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}-v{version}.safetensors"'},
    )


@app.get("/adapters/{name}/active")
def get_active(name: str) -> dict:
    store = get_store()
    active = store.get_active(name)
    if active is None:
        raise HTTPException(404, f"no active version for {name}")
    return _metadata_to_dict(store.get_metadata(name, active))


@app.get("/adapters/{name}/lineage")
def get_lineage(name: str) -> dict:
    """Every version with its parents (clients / cluster+client) and decisions."""
    return build_lineage(get_store(), name)


@app.post("/adapters/{name}/compose", status_code=201)
def compose(name: str, response: Response, body: dict = Body(...)) -> dict:
    """Store ``alpha*cluster + beta*client`` as one COMPOSITE version of ``name``.

    Body: ``{"cluster": "cluster-web" | {"name", "version"}, "client": ...,
    "alpha": 0.5, "beta": 1.0, "base_model"?: str}``. Part versions default to
    each part's active version. Re-composing identical inputs returns the
    existing version with 200 instead of writing a duplicate.
    """
    store = get_store()
    with _WRITE_LOCK:
        try:
            plan = plan_composite(store, parse_compose_request(body), name)
        except CompositionError as e:
            raise HTTPException(422, str(e)) from e
        meta, created = store_composite(store, name, plan)
    if not created:
        response.status_code = 200
    return _metadata_to_dict(meta)


def _composite_on_promote(spec: object, candidate: str) -> tuple[str, PlannedComposite]:
    """Validate a promote body's ``composite`` block and plan it (writes nothing)."""
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
        raise HTTPException(422, "composite must be an object with a 'name'")
    try:
        req = parse_compose_request(spec)
    except CompositionError as e:
        raise HTTPException(422, f"invalid composite: {e}") from e
    if candidate not in (req.cluster_name, req.client_name):
        raise HTTPException(
            422, f"composite must include the promoted adapter {candidate!r} as cluster or client"
        )
    try:
        return spec["name"], plan_composite(get_store(), req, spec["name"])
    except CompositionError as e:
        raise HTTPException(422, f"composite cannot be built: {e}") from e


def _parse_promote_body(body: dict) -> tuple[EvalResult, tuple[GuardMetrics, ...]]:
    try:
        if not isinstance(body.get("eval"), dict) or not isinstance(
                body["eval"].get("in_project"), dict):
            raise ValueError("eval.in_project is required (the D5 primary metric)")
        eval_result = EvalResult.from_json(body["eval"])
        baseline_guard = tuple(GuardMetrics.from_json(g) for g in body.get("baseline_guard", ()))
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise HTTPException(422, f"invalid promote payload: {e}") from e
    return eval_result, baseline_guard


@app.post("/adapters/{name}/promote")
def promote(name: str, body: dict = Body(...)) -> dict:
    """Apply the D5 two-sided rule to the currently-active version.

    Saves auto-activate; this endpoint is the checkpoint that confirms or
    reverts that activation once evaluation lands. Body::

        {"eval": <EvalResult>, "baseline_guard": [<GuardMetrics>, ...],
         "composite"?: {"name", "cluster", "client", "alpha", "beta"}}

    ``baseline_guard`` isn't part of the EvalResult contract — it's an
    API-boundary extension, same pattern as `save`'s ``meta`` envelope.

    With ``composite`` (D6), a PROMOTE also stores the pre-merged composite
    built from the parts' active versions — i.e. including this candidate.
    The composite is built in memory *before* anything is written, so a bad
    composite request rejects the whole call and records no decision. A
    ROLLBACK builds nothing; the previous composite stays active.
    """
    with _WRITE_LOCK:
        return _promote(name, body)


def _promote(name: str, body: dict) -> dict:
    store = get_store()
    eval_result, baseline_guard = _parse_promote_body(body)
    if eval_result.adapter.name != name:
        raise HTTPException(
            422, f"eval.adapter.name {eval_result.adapter.name!r} does not match path {name!r}"
        )

    active_before = store.get_active(name)
    if active_before is None:
        raise HTTPException(404, f"no active version for {name}")
    if active_before != eval_result.adapter.version:
        raise HTTPException(
            409,
            f"candidate v{eval_result.adapter.version} is not the active version "
            f"(active is v{active_before}) — promotion only evaluates the current active",
        )

    try:
        decision = decide(
            eval_result,
            baseline_guard=baseline_guard,
            active_version_before=active_before,
            previous_version=store.previous_version(name, eval_result.adapter.version),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    wants_composite = "composite" in body
    planned = None
    if wants_composite and decision.action == PromotionAction.PROMOTE:
        planned = _composite_on_promote(body["composite"], name)

    if decision.action == PromotionAction.ROLLBACK:
        store.set_active(name, decision.active_version_after)
    store.record_promotion(name, decision)
    result = _promotion_decision_to_dict(decision)
    if wants_composite:
        composite_name, plan = planned if planned else (None, None)
        result["composite"] = (
            _metadata_to_dict(store_composite(store, composite_name, plan)[0]) if plan else None
        )
    return result


@app.post("/adapters/{name}/restore")
def restore(name: str, body: dict = Body(...)) -> dict:
    """Repoint ``active`` by hand — the rollback drill and the D11 restore path.

    Body: ``{"reason": str, "to_version"?: int}``; ``to_version`` defaults to
    the version before the current active one. Recorded in the audit trail.
    """
    with _WRITE_LOCK:
        return _restore(name, body)


def _restore(name: str, body: dict) -> dict:
    reason, to_version = body.get("reason"), body.get("to_version")
    if not isinstance(reason, str) or not reason.strip():
        raise HTTPException(422, "reason is required (it goes in the audit trail)")
    if to_version is not None and (isinstance(to_version, bool) or not isinstance(to_version, int)):
        raise HTTPException(422, f"to_version must be an integer, got {to_version!r}")

    store = get_store()
    active = store.get_active(name)
    if active is None:
        store.list_versions(name)  # 404 for an unknown adapter
        raise HTTPException(409, f"{name} has no active version to restore from")
    if to_version is None:
        to_version = store.previous_version(name, active)
        if to_version is None:
            raise HTTPException(409, f"{name} v{active} is the first version — nothing to restore")
    if to_version == active:
        raise HTTPException(409, f"{name} v{to_version} is already active")

    store.get_metadata(name, to_version)  # 404 if that version does not exist
    candidate = store.get_metadata(name, active).ref
    decision = restore_decision(candidate, to_version=to_version, reason=reason.strip())
    store.set_active(name, to_version)
    store.record_promotion(name, decision)
    return _promotion_decision_to_dict(decision)


def _parse_gc_body(body: dict) -> tuple[int, bool]:
    keep_last, dry_run = body.get("keep_last"), body.get("dry_run", True)
    if keep_last is None:
        try:
            keep_last = default_keep_last()
        except ValueError as e:
            raise HTTPException(500, f"registry misconfigured: {e}") from e
    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
        raise HTTPException(422, f"keep_last must be an integer >= 1, got {keep_last!r}")
    if not isinstance(dry_run, bool):
        raise HTTPException(422, f"dry_run must be a boolean, got {dry_run!r}")
    return keep_last, dry_run


def _gc_report(name: str, plan: RetentionPlan, dry_run: bool) -> dict:
    return {
        "name": name,
        "dry_run": dry_run,
        "deleted": list(plan.delete),
        "kept": {str(v): list(reasons) for v, reasons in plan.keep.items()},
    }


@app.post("/adapters/{name}/gc")
def gc_one(name: str, body: dict = Body(default={})) -> dict:
    """Retention for one adapter. Body: ``{"keep_last"?: int, "dry_run"?: bool}``.

    Dry run by default; ``keep_last`` defaults to ``CLASP_REGISTRY_KEEP_LAST``
    (else 5). See ``registry.retention`` for what is always protected.
    """
    with _WRITE_LOCK:
        return _gc_one(name, body)


def _gc_one(name: str, body: dict) -> dict:
    keep_last, dry_run = _parse_gc_body(body)
    plan = gc_adapter(get_store(), name, keep_last=keep_last, dry_run=dry_run)
    return _gc_report(name, plan, dry_run)


@app.post("/gc")
def gc_all(body: dict = Body(default={})) -> dict:
    """Retention across every adapter, with one shared composite-reference scan."""
    with _WRITE_LOCK:
        return _gc_all(body)


def _gc_all(body: dict) -> dict:
    keep_last, dry_run = _parse_gc_body(body)
    store = get_store()
    referenced = collect_referenced(store)
    reports = []
    for name in store.list_adapters():
        if not store.list_versions(name):
            continue
        plan = gc_adapter(store, name, keep_last=keep_last, dry_run=dry_run,
                          referenced=referenced)
        reports.append(_gc_report(name, plan, dry_run))
    return {"dry_run": dry_run, "keep_last": keep_last, "adapters": reports}


@app.get("/adapters/{name}/promotions")
def list_promotions(name: str) -> dict:
    store = get_store()
    if name not in store.list_adapters():
        raise HTTPException(404, f"adapter not found: {name}")
    return {
        "name": name,
        "decisions": [_promotion_decision_to_dict(d) for d in store.list_promotions(name)],
    }
