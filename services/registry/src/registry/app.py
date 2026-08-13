"""CLASP State Registry — FastAPI app (P4, Deepak).

Endpoints:

    GET  /healthz                                  liveness
    GET  /adapters                                 list adapter names
    POST /adapters/{name}/versions                 save a new immutable version
    GET  /adapters/{name}/versions                 list versions + metadata
    GET  /adapters/{name}/versions/{v}             metadata for one version
    GET  /adapters/{name}/versions/{v}/file        download the safetensors blob
    GET  /adapters/{name}/active                   metadata for the active version
    POST /adapters/{name}/promote                  D5 two-sided rule on the active version
    GET  /adapters/{name}/promotions               promotion/rollback audit trail

Composite-adapter storage (D6) is the W8 follow-on, still to build.
mTLS in front of these endpoints is a W10 deliverable (D7).
"""
from __future__ import annotations

import json

from contracts import (
    CONTRACTS_VERSION,
    AdapterKind,
    AdapterRef,
    AggregationMethod,
    EvalResult,
    GuardMetrics,
    InProjectMetrics,
    LoRAHyperParams,
    PrivacySpec,
    PromotionAction,
)
from fastapi import Body, FastAPI, File, Form, HTTPException, Response, UploadFile

from . import __version__
from .promotion import decide
from .storage import (
    AdapterNotFound,
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


def _hparams_from(d: dict | None) -> LoRAHyperParams:
    d = d or {}
    base = LoRAHyperParams()
    return LoRAHyperParams(
        rank=d.get("rank", base.rank),
        lora_alpha=d.get("lora_alpha", base.lora_alpha),
        dropout=d.get("dropout", base.dropout),
        target_modules=tuple(d.get("target_modules", base.target_modules)),
        alpha=d.get("alpha", base.alpha),
        beta=d.get("beta", base.beta),
    )


def _in_project_from(d: dict | None) -> InProjectMetrics | None:
    if d is None:
        return None
    return InProjectMetrics(
        edit_similarity=d["edit_similarity"],
        exact_match=d["exact_match"],
        perplexity=d["perplexity"],
        n_examples=d["n_examples"],
    )


def _guard_metrics_from(d: dict) -> GuardMetrics:
    return GuardMetrics(
        benchmark=d["benchmark"],
        pass_at_k={int(k): v for k, v in d["pass_at_k"].items()},
    )


def _eval_result_from(d: dict) -> EvalResult:
    ref = d["adapter"]
    return EvalResult(
        adapter=AdapterRef(
            name=ref["name"],
            version=ref["version"],
            kind=AdapterKind(ref.get("kind", "client")),
            cluster_id=ref.get("cluster_id"),
        ),
        in_project=_in_project_from(d["in_project"]),
        guard=tuple(_guard_metrics_from(g) for g in d.get("guard", ())),
        baseline_in_project=_in_project_from(d.get("baseline_in_project")),
        baseline_noise_band=d.get("baseline_noise_band", 0.0),
        seed=d.get("seed", 0),
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "registry", "contracts": CONTRACTS_VERSION}


@app.get("/adapters")
def list_adapters() -> dict:
    store = get_store()
    return {"adapters": store.list_adapters()}


@app.post("/adapters/{name}/versions", status_code=201)
async def save_version(
    name: str,
    file: UploadFile = File(..., description="safetensors payload"),
    meta: str = Form("{}", description="JSON metadata envelope"),
) -> dict:
    """Write a new immutable version and return its metadata."""
    store = get_store()
    try:
        m = json.loads(meta)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"meta is not valid JSON: {e}") from e

    payload = await file.read()
    try:
        kind = AdapterKind(m.get("kind", "client"))
        aggregation = AggregationMethod(m["aggregation"]) if m.get("aggregation") else None
        written = store.save(
            name,
            payload,
            kind=kind,
            hparams=_hparams_from(m.get("hparams")),
            privacy=PrivacySpec(**m["privacy"]) if m.get("privacy") else None,
            aggregation=aggregation,
            round=m.get("round"),
            seed=m.get("seed", 0),
            cluster_id=m.get("cluster_id"),
            source_clients=tuple(m.get("source_clients", ())),
            set_active=m.get("set_active", True),
        )
    except VersionExists as e:  # M4: concurrent save lost the version race
        raise HTTPException(409, str(e)) from e
    except (StorageError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    return _metadata_to_dict(written)


@app.get("/adapters/{name}/versions")
def list_versions(name: str) -> dict:
    store = get_store()
    try:
        versions = store.list_versions(name)
    except AdapterNotFound as e:
        raise HTTPException(404, f"adapter not found: {name}") from e
    return {
        "name": name,
        "active": store.get_active(name),
        "versions": [_metadata_to_dict(store.get_metadata(name, v)) for v in versions],
    }


@app.get("/adapters/{name}/versions/{version}")
def get_version(name: str, version: int) -> dict:
    try:
        return _metadata_to_dict(get_store().get_metadata(name, version))
    except AdapterNotFound as e:
        raise HTTPException(404, str(e)) from e


@app.get("/adapters/{name}/versions/{version}/file")
def download_version(name: str, version: int) -> Response:
    store = get_store()
    try:
        payload = store.load_payload(name, version)
    except AdapterNotFound as e:
        raise HTTPException(404, str(e)) from e
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


@app.post("/adapters/{name}/promote")
def promote(name: str, body: dict = Body(...)) -> dict:
    """Apply the D5 two-sided rule to the currently-active version.

    Saves auto-activate (D9 groundwork); this endpoint is the checkpoint that
    confirms or reverts that activation once P5's evaluation lands. Body::

        {"eval": <EvalResult>, "baseline_guard": [<GuardMetrics>, ...]}

    ``baseline_guard`` isn't part of the frozen EvalResult contract (v1.0) —
    it's an API-boundary extension, same pattern as `save`'s ``meta`` envelope.
    """
    store = get_store()
    try:
        eval_result = _eval_result_from(body["eval"])
        baseline_guard = tuple(_guard_metrics_from(g) for g in body.get("baseline_guard", ()))
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(422, f"invalid promote payload: {e}") from e

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
