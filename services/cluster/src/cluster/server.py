"""Cluster server: Flower strategy with SVD LoRA aggregation (D2).

``SVDLoRAStrategy`` plugs the Week-3 aggregation core into Flower's FedAvg
skeleton: client updates arrive as flat ndarray lists (see
``LoRAAdapter.to_ndarrays`` for the wire order), are re-factorized through the
exact-average + truncated-SVD path, and the resulting cluster adapter is
broadcast back.

``build_server_app`` / ``start_grpc_server`` expose the same strategy through
Flower's new (ServerApp) and legacy (start_server) entry points; the in-process
driver in ``simulation.py`` uses the strategy math directly so tests need no
network or ray.
"""

from __future__ import annotations

import os as _os

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from pydantic import BaseModel as _BaseModel

from cluster.adapter_format import (
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    TARGET_MODULES,
    AdapterFormatError,
    LoRAAdapter,
)
from cluster.aggregation import (
    aggregate_layerwise,
    aggregate_naive,
    aggregate_svd,
    aggregate_svd_lowrank,
    layerwise_exact_average_error,
)
from cluster.schemas.messages import (
    DEFAULT_CLUSTER_ID as _DEFAULT_CLUSTER_ID,
)
from cluster.schemas.messages import (
    AdapterUpload,
    ClusterAdapterBroadcast,
    TensorPayload,
)


class SVDLoRAStrategy(FedAvg):
    """FedAvg skeleton with delta_W exact-average + SVD re-factorization."""

    def __init__(
        self,
        *args,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
        aggregation: str = "svd",  # "svd" | "naive" (D2 ablation)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if aggregation not in ("svd", "naive"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.aggregation = aggregation
        self.round_log: list[dict[str, Scalar]] = []

    def _to_adapter(self, arrays) -> LoRAAdapter:
        return LoRAAdapter.from_ndarrays(
            arrays, rank=self.rank, alpha=self.alpha, target_modules=self.target_modules
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        adapters = (
            self._to_adapter(parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        )
        weights = [float(fit_res.num_examples) for _, fit_res in results]
        if self.aggregation == "svd":
            merged = aggregate_svd(adapters, weights, rank=self.rank)
        else:
            merged = aggregate_naive(adapters, weights)
        metrics: dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
            "aggregation": self.aggregation,
        }
        losses = [
            fit_res.metrics["loss"]
            for _, fit_res in results
            if fit_res.metrics and "loss" in fit_res.metrics
        ]
        if losses:
            metrics["mean_loss"] = float(sum(losses) / len(losses))
        self.round_log.append(metrics)
        return ndarrays_to_parameters(merged.to_ndarrays()), metrics


def build_strategy(
    initial_adapter: LoRAAdapter,
    aggregation: str = "svd",
    min_clients: int = 3,
) -> SVDLoRAStrategy:
    return SVDLoRAStrategy(
        rank=initial_adapter.rank,
        alpha=initial_adapter.alpha,
        target_modules=initial_adapter.target_modules,
        aggregation=aggregation,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(initial_adapter.to_ndarrays()),
    )


def build_server_app(
    initial_adapter: LoRAAdapter,
    num_rounds: int = 3,
    aggregation: str = "svd",
    min_clients: int = 3,
):
    """Flower ServerApp (new API) around the SVD strategy — used by `flwr run`."""
    from flwr.server import ServerApp, ServerAppComponents

    strategy = build_strategy(initial_adapter, aggregation, min_clients)

    def server_fn(context):
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=num_rounds)
        )

    return ServerApp(server_fn=server_fn)


def start_grpc_server(
    initial_adapter: LoRAAdapter,
    server_address: str = "127.0.0.1:8080",
    num_rounds: int = 3,
    aggregation: str = "svd",
    min_clients: int = 3,
):
    """Legacy blocking entry point (Week 1 skeleton); mTLS lands with P3 in W7."""
    import flwr

    return flwr.server.start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=build_strategy(initial_adapter, aggregation, min_clients),
    )



# =============================================================================
# HTTP service (Integration Sprint B2/B3) — FastAPI on :8002
# =============================================================================
#
# Wraps the exact-average + SVD aggregation above behind a small HTTP surface
# so Edge can reach Cluster over a plain round trip instead of only in-process
# (simulation.py) or over Flower's own wire protocol (SVDLoRAStrategy above,
# used by build_server_app/start_grpc_server for the W7 gRPC path).
#
# NO AGGREGATION MATHEMATICS IS DUPLICATED OR REIMPLEMENTED HERE. Every
# endpoint calls straight into cluster.aggregation and
# cluster.adapter_format.LoRAAdapter.
#
# State is in-memory, keyed by cluster_id, and resets on process restart:
# there is no database here. Durable, versioned storage is services/registry's
# job, and /adapters/{cluster_id}/publish is the seam-B hop that hands it over.

DEFAULT_CLUSTER_ID = _DEFAULT_CLUSTER_ID
#: Guard against a runaway client fan-in; the demo does 3 clients per cluster.
MAX_UPLOADS_PER_CLUSTER = 64


class _StoredUpload:
    __slots__ = ("adapter", "num_examples", "round_id")

    def __init__(self, adapter: LoRAAdapter, num_examples: int, round_id: int) -> None:
        self.adapter = adapter
        self.num_examples = num_examples
        self.round_id = round_id


class _ClusterState:
    """One project cluster's upload buffer and last aggregate.

    Not thread-safe beyond what FastAPI's default single-worker dev server
    already serializes; concurrent multi-worker deployment is out of scope for
    this sprint, same as retry/quorum fault tolerance.
    """

    def __init__(self) -> None:
        self.uploads: dict[str, _StoredUpload] = {}
        self.round_id: int = 0
        self.active: ClusterAdapterBroadcast | None = None
        self.last_manifest: dict[str, object] | None = None


#: cluster_id -> state. Two statically-assigned clusters in Phase II (D1:
#: "web" and "scientific"); a cluster's state is created on its first upload.
_clusters: dict[str, _ClusterState] = {DEFAULT_CLUSTER_ID: _ClusterState()}

#: The default cluster's state, exported under its historical name so callers
#: written against the single-cluster server keep working unchanged.
_state = _clusters[DEFAULT_CLUSTER_ID]


def _cluster(cluster_id: str) -> _ClusterState:
    return _clusters.setdefault(cluster_id, _ClusterState())


def _reset_clusters() -> None:
    """Drop every cluster's buffer, keeping the default cluster's identity."""
    _clusters.clear()
    _clusters[DEFAULT_CLUSTER_ID] = _state
    _state.uploads.clear()
    _state.round_id = 0
    _state.active = None
    _state.last_manifest = None


app = FastAPI(
    title="CLASP Cluster Service",
    description="Edge-facing HTTP surface over the existing SVD aggregation core (P2).",
    version="0.2.0",
)


class AggregateRequest(_BaseModel):
    cluster_id: str = DEFAULT_CLUSTER_ID
    aggregation: str = "svd"  # "svd" | "naive" — same choice SVDLoRAStrategy exposes
    rank: int | None = None  # defaults to the uploaded adapters' own rank
    #: Take the exact low-rank route through the same D2 pipeline
    #: (``aggregate_svd_lowrank``). Same mathematics and — measured on the real
    #: 24-layer web cluster — the same reconstruction error to six decimals
    #: (0.171237 mean / 0.227009 max both ways), but 2.5 s instead of 864.6 s,
    #: and without the 3.2 GB of dense float64 accumulators the reference path
    #: allocates at real width. Set false to force the reference path.
    exact_lowrank: bool = True
    #: Include per-module SVD reconstruction error in the manifest. Opt-in, as
    #: P2 made it: on the reference path it costs a second dense pass over
    #: every module, which is the duplicated work that OOM-killed a
    #: verification run. It is free on the low-rank path (it falls straight out
    #: of the singular values), but the default stays off so one flag means the
    #: same thing on both paths. The demo orchestrator asks for it explicitly —
    #: the sprint wants reconstruction error in the aggregation manifest.
    include_manifest: bool = False
    #: Keep the upload buffer and round counter after aggregating. Off by
    #: default, so the normal path is unchanged: aggregate once, clear, advance.
    #: The D2 ablation needs the SVD aggregate and the naive aggregate built
    #: from the SAME round's uploads to be comparable at all; without this the
    #: orchestrator would have to make every client re-send 25 MB of tensors
    #: between the two, which measures the network rather than the mathematics.
    retain_uploads: bool = False


#: Where seam B posts when the caller does not say. Inside compose the registry
#: answers to its service name, not to localhost, so the default is read from
#: the environment rather than hardcoded to one deployment shape.
DEFAULT_REGISTRY_URL = _os.environ.get("CLASP_REGISTRY_URL", "http://localhost:8004")


class PublishRequest(_BaseModel):
    """Seam B: push this cluster's active aggregate into the State Registry."""

    registry_url: str = DEFAULT_REGISTRY_URL
    #: Registry adapter name; defaults to the cluster_id, which is what makes
    #: ``GET /adapters/cluster-web/versions`` the panel-visible gate.
    adapter_name: str | None = None
    round: int | None = None
    seed: int = 0
    set_active: bool = True
    timeout_s: float = 120.0


def _canonical_safetensors(blob: bytes) -> bytes:
    """Rewrite the safetensors header with sorted, compact JSON.

    ``safetensors`` serializes its header — ``__metadata__`` included — from a
    Rust hash map, so the same tensors and the same metadata come out in a
    different key order from call to call: three ``save`` calls in one process
    were measured producing two distinct sha256 digests. Without this, an
    adapter's published bytes (and the digest the registry records against
    them) change run to run even when nothing about the adapter changed, which
    quietly breaks D9's reproducibility story.

    Header-only: ``data_offsets`` are relative to the buffer that FOLLOWS the
    header, so re-serializing the header JSON canonically leaves them valid.

    Mirrored in ``edge.wire.canonicalize_safetensors`` — duplicated rather than
    imported because ``contracts`` is the only cross-module import path
    (docs/architecture.md §2); ``tests/integration`` asserts the two agree.
    """
    import json

    header_len = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8 + header_len].decode("utf-8"))
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(canonical).to_bytes(8, "little") + canonical + blob[8 + header_len:]


def _peft_bytes(broadcast: ClusterAdapterBroadcast) -> bytes:
    """The aggregate as a safetensors blob, config embedded in ``__metadata__``.

    The registry stores payloads opaquely and serves them back with no config
    beside them (sprint Defect 4). Carrying adapter_config.json inside the
    blob's own metadata header means the round trip through the registry loses
    nothing, and Edge never has to invent a hyperparameter it was not given.
    """
    import json

    from safetensors.numpy import save

    tensors = {t.name: t.to_numpy() for t in broadcast.tensors}
    metadata = {"format": "pt", "cluster_id": broadcast.cluster_id,
                "source_clients": ",".join(broadcast.source_clients)}
    if broadcast.peft_config is not None:
        metadata["adapter_config"] = json.dumps(broadcast.peft_config)
    return _canonical_safetensors(save(tensors, metadata=metadata))


@app.get("/healthz")
def healthz() -> dict[str, object]:
    """Liveness plus per-cluster round bookkeeping.

    The top-level counters describe the DEFAULT cluster (unchanged from the
    single-cluster surface); ``clusters`` breaks them out per cluster_id.
    """
    return {
        "status": "ok",
        "service": "cluster",
        "pending_uploads": len(_state.uploads),
        "round_id": _state.round_id,
        "has_active_adapter": _state.active is not None,
        "clusters": {
            cid: {
                "pending_uploads": len(st.uploads),
                "pending_clients": sorted(st.uploads),
                "round_id": st.round_id,
                "has_active_adapter": st.active is not None,
            }
            for cid, st in sorted(_clusters.items())
        },
    }


@app.post("/uploads", status_code=201)
def upload_adapter(upload: AdapterUpload) -> dict[str, object]:
    """Seam A: Edge -> Cluster. Accepts one client's trained LoRA adapter.

    ``upload.tensors`` may use either key convention ``LoRAAdapter`` parses:
    the short internal ``layers.<i>.<module>.<part>.weight`` form, or a real
    PEFT dump like
    ``base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight`` — both
    go through the same ``LoRAAdapter.from_state_dict``, unchanged.
    """
    state = _cluster(upload.cluster_id)
    # Reject uploads for a stale/wrong round rather than silently folding them
    # into whatever round is currently buffering.
    if upload.round_id != state.round_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"upload round_id={upload.round_id} does not match "
                f"{upload.cluster_id}'s current round_id={state.round_id}"
            ),
        )
    if (upload.client_id not in state.uploads
            and len(state.uploads) >= MAX_UPLOADS_PER_CLUSTER):
        raise HTTPException(
            status_code=429,
            detail=f"{upload.cluster_id} already holds {MAX_UPLOADS_PER_CLUSTER} uploads",
        )

    state_dict = {t.name: t.to_numpy() for t in upload.tensors}
    try:
        adapter = LoRAAdapter.from_state_dict(
            state_dict,
            rank=upload.rank,
            alpha=upload.alpha,
            target_modules=upload.target_modules,
            num_layers=upload.num_layers,
        )
    except AdapterFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    state.uploads[upload.client_id] = _StoredUpload(
        adapter=adapter, num_examples=upload.num_examples, round_id=upload.round_id
    )
    return {
        "client_id": upload.client_id,
        "cluster_id": upload.cluster_id,
        "round_id": upload.round_id,
        "num_layers": upload.num_layers,
        "tensors_received": len(upload.tensors),
        "status": "accepted",
        "pending_uploads": len(state.uploads),
    }


@app.post("/aggregate", response_model=ClusterAdapterBroadcast)
def aggregate(request: AggregateRequest | None = None) -> ClusterAdapterBroadcast:
    """Run the existing aggregation core over one cluster's buffered uploads,
    then clear that buffer and advance its round counter — the same
    one-round-at-a-time semantics ``simulation.run_round`` already uses."""
    req = request or AggregateRequest()
    state = _cluster(req.cluster_id)
    if not state.uploads:
        raise HTTPException(
            status_code=400, detail=f"no uploads to aggregate for {req.cluster_id}"
        )
    if req.aggregation not in ("svd", "naive"):
        raise HTTPException(
            status_code=422, detail=f"unknown aggregation {req.aggregation!r}"
        )

    client_ids = sorted(state.uploads)
    stored = [state.uploads[c] for c in client_ids]
    adapters = [s.adapter for s in stored]
    weights = [float(s.num_examples) for s in stored]
    rank = req.rank or adapters[0].rank

    manifest: dict[str, object] = {
        "cluster_id": req.cluster_id,
        "round_id": state.round_id,
        "num_clients": len(stored),
        "aggregation": req.aggregation,
        "source_clients": client_ids,
        "sample_weights": weights,
        "rank": rank,
    }

    if req.aggregation == "svd" and req.exact_lowrank:
        merged, errors = aggregate_svd_lowrank(adapters, weights, rank=rank)
        manifest["aggregation_path"] = "exact_lowrank"
        if req.include_manifest:
            vals = list(errors.values())
            manifest["svd_reconstruction_error"] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "max": float(np.max(vals)) if vals else 0.0,
                "min": float(np.min(vals)) if vals else 0.0,
                "n_modules": len(vals),
                "definition": (
                    "relative Frobenius ||dW - dW_r|| / ||dW|| of the rank-r "
                    "truncation against the exact weighted-average delta_W"
                ),
            }
    else:
        merged = aggregate_layerwise(
            adapters, weights, rank=rank, aggregation=req.aggregation
        )
        manifest["aggregation_path"] = "reference_layerwise"
        if req.include_manifest:
            # Same keys as the low-rank branch, so a manifest reader does not
            # have to know which path produced it. For the naive branch this
            # number is the D2 ablation: the cross-term bias of averaging the
            # A/B factors instead of the delta_Ws.
            ref = layerwise_exact_average_error(adapters, weights, merged)
            manifest["svd_reconstruction_error"] = {
                "mean": ref["mean_rel_err"],
                "max": ref["max_rel_err"],
                "min": None,
                "n_modules": ref["n_modules"],
                "definition": (
                    "relative Frobenius ||dW_aggregate - dW_exact_average|| / "
                    "||dW_exact_average||, measured per (layer, module)"
                ),
            }

    broadcast = ClusterAdapterBroadcast(
        cluster_id=req.cluster_id,
        round_id=state.round_id,
        rank=merged.rank,
        target_modules=merged.target_modules,
        alpha=merged.alpha,
        num_layers=merged.num_layers,
        num_clients=len(stored),
        source_clients=tuple(client_ids),
        aggregation=req.aggregation,
        peft_config=merged.to_peft_config(),
        tensors=[
            TensorPayload.from_numpy(name, arr)
            for name, arr in merged.to_peft_state_dict().items()
        ],
    )

    state.active = broadcast
    state.last_manifest = manifest
    manifest["retained_uploads"] = req.retain_uploads
    if not req.retain_uploads:
        state.uploads.clear()
        state.round_id += 1
    return broadcast


# NOTE: the literal '/adapters/cluster/active' route is declared BEFORE the
# parameterized one on purpose. FastAPI matches routes in declaration order, so
# this keeps the original single-cluster path meaning "the default cluster"
# instead of resolving cluster_id="cluster" to an empty new cluster.
@app.get("/adapters/cluster/active", response_model=ClusterAdapterBroadcast)
def get_default_active_adapter() -> ClusterAdapterBroadcast:
    """Single-cluster alias for the default cluster's active aggregate."""
    if _state.active is None:
        raise HTTPException(status_code=404, detail="no aggregated cluster adapter yet")
    return _state.active


@app.get("/adapters/{cluster_id}/active", response_model=ClusterAdapterBroadcast)
def get_active_adapter(cluster_id: str) -> ClusterAdapterBroadcast:
    """Where the orchestrator pulls a cluster's aggregate from before the
    registry hop. ``tensors`` carry the full PEFT key convention
    (``to_peft_state_dict``) and ``peft_config`` is a ready adapter_config.json
    — together enough to write a directory ``edge.merge.load_adapter`` reads."""
    state = _clusters.get(cluster_id)
    if state is None or state.active is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregated cluster adapter for {cluster_id!r}"
        )
    return state.active


@app.get("/adapters/{cluster_id}/download")
def download_adapter(cluster_id: str) -> Response:
    """The active aggregate as raw safetensors bytes — byte-for-byte the
    payload seam B stores in the registry, with adapter_config.json embedded in
    the file's own ``__metadata__`` header."""
    state = _clusters.get(cluster_id)
    if state is None or state.active is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregated cluster adapter for {cluster_id!r}"
        )
    return Response(
        content=_peft_bytes(state.active),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{cluster_id}.safetensors"',
            "X-CLASP-Round": str(state.active.round_id),
            "X-CLASP-Source-Clients": ",".join(state.active.source_clients),
        },
    )


@app.get("/aggregate/manifest")
def get_last_manifest() -> dict[str, object]:
    """Single-cluster alias for the default cluster's last aggregation manifest."""
    if _state.last_manifest is None:
        raise HTTPException(status_code=404, detail="no aggregation has run yet")
    return _state.last_manifest


@app.get("/adapters/{cluster_id}/manifest")
def get_cluster_manifest(cluster_id: str) -> dict[str, object]:
    state = _clusters.get(cluster_id)
    if state is None or state.last_manifest is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregation has run for {cluster_id!r}"
        )
    return state.last_manifest


@app.post("/adapters/{cluster_id}/publish", status_code=201)
def publish_to_registry(cluster_id: str, request: PublishRequest | None = None) -> dict:
    """Seam B: Cluster -> Registry. POST the active aggregate as a new version.

    This is the one place Cluster talks to another service. It sends exactly
    what ``/download`` serves, plus the metadata envelope the registry records:
    aggregation method, source clients, round and the LoRA hyperparameters —
    every field read off the aggregate itself, none typed by hand. The registry
    computes and stores its own sha256 over the payload; Edge verifies the
    download against that on the way back out (seam C1).
    """
    import json as _json

    import httpx

    req = request or PublishRequest()
    state = _clusters.get(cluster_id)
    if state is None or state.active is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregated cluster adapter for {cluster_id!r}"
        )
    broadcast = state.active
    name = req.adapter_name or cluster_id
    payload = _peft_bytes(broadcast)
    cfg = broadcast.peft_config or {}
    meta = {
        "kind": "cluster",
        "cluster_id": cluster_id,
        "aggregation": "svd_exact" if broadcast.aggregation == "svd" else "naive_avg",
        "round": req.round if req.round is not None else broadcast.round_id,
        "seed": req.seed,
        "source_clients": list(broadcast.source_clients),
        "set_active": req.set_active,
        "hparams": {
            "rank": broadcast.rank,
            # Pinned to r so PEFT's scaling is exactly 1.0 and the stored
            # tensors mean delta_W itself — the convention edge.aggregate uses.
            "lora_alpha": int(cfg.get("r", broadcast.rank)),
            "dropout": float(cfg.get("lora_dropout", 0.0)),
            "target_modules": list(broadcast.target_modules),
        },
    }

    try:
        with httpx.Client(timeout=req.timeout_s) as http:
            resp = http.post(
                f"{req.registry_url.rstrip('/')}/adapters/{name}/versions",
                files={"file": (f"{name}.safetensors", payload,
                                "application/octet-stream")},
                data={"meta": _json.dumps(meta)},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"registry unreachable at {req.registry_url}: {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"registry rejected the snapshot ({resp.status_code}): {resp.text}",
        )
    return {
        "cluster_id": cluster_id,
        "registry_url": req.registry_url,
        "adapter_name": name,
        "version": resp.json(),
    }
