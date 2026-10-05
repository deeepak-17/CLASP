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
    POST /adapters/{name}/promote                  D5 two-sided rule on the active version
    GET  /adapters/{name}/promotions               promotion/rollback audit trail

Storage errors map to HTTP status in one place (see the exception handlers):
unknown adapter/version -> 404, bad name -> 422, version race -> 409, and a
corrupt on-disk record -> a clean 500 with the reason, never a traceback.
"""
from __future__ import annotations

import json

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
from .lineage import build_lineage
from .promotion import decide
from .storage import (
    AdapterNotFound,
    InvalidAdapterName,
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

_store: RegistryStore | None = None


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


@app.exception_handler(StorageError)
async def _data_error(_: Request, exc: StorageError) -> JSONResponse:
    # L1: e.g. a corrupt/hand-edited `active` pointer — clean 500, not a crash.
    return JSONResponse(status_code=500, content={"detail": f"registry data error: {exc}"})


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
async def save_version(
    name: str,
    file: UploadFile = File(..., description="safetensors payload"),
    meta: str = Form("{}", description="JSON metadata envelope"),
) -> dict:
    """Write a new immutable version and return its metadata."""
    kwargs = _parse_save_meta(meta)
    payload = await file.read()
    try:
        written = get_store().save(name, payload, **kwargs)
    except (VersionExists, InvalidAdapterName):  # M4 race -> 409, bad name -> 422
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


def _parse_promote_body(body: dict) -> tuple[EvalResult, tuple[GuardMetrics, ...]]:
    try:
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

        {"eval": <EvalResult>, "baseline_guard": [<GuardMetrics>, ...]}

    ``baseline_guard`` isn't part of the EvalResult contract — it's an
    API-boundary extension, same pattern as `save`'s ``meta`` envelope.
    """
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

    if decision.action == PromotionAction.ROLLBACK:
        store.set_active(name, decision.active_version_after)
    store.record_promotion(name, decision)
    return _promotion_decision_to_dict(decision)


@app.get("/adapters/{name}/promotions")
def list_promotions(name: str) -> dict:
    store = get_store()
    if name not in store.list_adapters():
        raise HTTPException(404, f"adapter not found: {name}")
    return {
        "name": name,
        "decisions": [_promotion_decision_to_dict(d) for d in store.list_promotions(name)],
    }
