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

import numpy as np
from fastapi import FastAPI, HTTPException
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
from cluster.aggregation import aggregate_naive, aggregate_svd, exact_average_delta
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
# Wraps the exact-average + SVD aggregation above behind three endpoints so
# Edge can reach Cluster over a plain HTTP round trip instead of only
# in-process (simulation.py) or over Flower's own wire protocol
# (SVDLoRAStrategy above, used by build_server_app/start_grpc_server for the
# W7 gRPC path). No aggregation math is duplicated or reimplemented here —
# every endpoint calls straight into
# cluster.aggregation.{aggregate_svd,aggregate_naive,exact_average_delta} and
# cluster.adapter_format.LoRAAdapter, unchanged.
#
# State is in-memory and resets on process restart: there is no database and
# no registry client here. services/registry is a separate, unmodified
# service — Edge pulling the aggregated adapter back out of the *registry*
# (seam C1 in the integration sprint doc) is registry's endpoint, not this
# one. What this module exposes is Cluster's own upload -> aggregate ->
# download loop, which is the piece Cluster owns end to end.

class _StoredUpload:
    __slots__ = ("adapter", "num_examples", "round_id")

    def __init__(self, adapter: LoRAAdapter, num_examples: int, round_id: int) -> None:
        self.adapter = adapter
        self.num_examples = num_examples
        self.round_id = round_id


class _ClusterState:
    """Everything the HTTP endpoints share. One process, one cluster, one
    upload buffer at a time — matches the integration sprint's single-round,
    single-cluster scope. Not thread-safe beyond what FastAPI's default
    single-worker dev server already serializes; concurrent multi-worker
    deployment is out of scope here, same as retry/quorum fault tolerance."""

    def __init__(self) -> None:
        self.uploads: dict[str, _StoredUpload] = {}
        self.round_id: int = 0
        self.active: ClusterAdapterBroadcast | None = None
        self.last_manifest: dict[str, object] | None = None


_state = _ClusterState()

app = FastAPI(
    title="CLASP Cluster Service",
    description="Edge-facing HTTP surface over the existing SVD aggregation core (P2).",
    version="0.1.0",
)


class AggregateRequest(_BaseModel):
    aggregation: str = "svd"  # "svd" | "naive" — same choice SVDLoRAStrategy exposes
    rank: int | None = None  # defaults to the uploaded adapters' own rank
    # Opt-in: also compute the SVD-reconstruction-error manifest below.
    # Review fix: this recomputed exact_average_delta from scratch on top of
    # the equivalent averaging aggregate_svd() already does internally (and
    # discards) — doubling compute/memory on every /aggregate call on the
    # svd path, the same class of duplicated work that OOM-killed the
    # verification VM at real model width (see test_real_adapter_pipeline.py).
    # Off by default; callers that want the diagnostic ask for it explicitly.
    include_manifest: bool = False


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "cluster",
        "pending_uploads": len(_state.uploads),
        "round_id": _state.round_id,
        "has_active_adapter": _state.active is not None,
    }


@app.post("/uploads", status_code=201)
def upload_adapter(upload: AdapterUpload) -> dict[str, object]:
    """Seam A: Edge -> Cluster. Accepts one client's trained LoRA adapter.

    ``upload.tensors`` may use either key convention ``LoRAAdapter`` already
    parses: the short internal 'layers.<i>.<module>.<part>.weight' form, or
    a real PEFT dump like
    'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight' — both
    go through the same ``LoRAAdapter.from_state_dict``, unchanged.
    """
    # Review fix: reject uploads for a stale/wrong round instead of
    # silently folding them into whatever round is currently buffering (the
    # old code stored upload.round_id but never checked it against
    # _state.round_id, so the 201 response echoed back a round_id that was
    # never actually honored).
    if upload.round_id != _state.round_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"upload round_id={upload.round_id} does not match the "
                f"cluster's current round_id={_state.round_id}"
            ),
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

    _state.uploads[upload.client_id] = _StoredUpload(
        adapter=adapter, num_examples=upload.num_examples, round_id=upload.round_id
    )
    return {
        "client_id": upload.client_id,
        "round_id": upload.round_id,
        "num_layers": upload.num_layers,
        "tensors_received": len(upload.tensors),
        "status": "accepted",
        "pending_uploads": len(_state.uploads),
    }


@app.post("/aggregate", response_model=ClusterAdapterBroadcast)
def aggregate(request: AggregateRequest | None = None) -> ClusterAdapterBroadcast:
    """Runs the existing aggregation core over every upload currently
    buffered, then clears the buffer and advances the round counter — the
    same one-round-at-a-time semantics ``simulation.run_round`` already uses."""
    if not _state.uploads:
        raise HTTPException(status_code=400, detail="no uploads to aggregate")

    req = request or AggregateRequest()
    stored = list(_state.uploads.values())
    adapters = [s.adapter for s in stored]
    weights = [float(s.num_examples) for s in stored]
    rank = req.rank or adapters[0].rank

    if req.aggregation == "svd":
        merged = aggregate_svd(iter(adapters), weights, rank=rank)
    elif req.aggregation == "naive":
        merged = aggregate_naive(iter(adapters), weights)
    else:
        raise HTTPException(status_code=422, detail=f"unknown aggregation {req.aggregation!r}")

    # Manifest: SVD reconstruction error vs the exact weighted average, per
    # layer — the same comparison main.py/demo_all_weeks.py already make at
    # one layer's scale, extended across every layer of a real upload. Reuses
    # exact_average_delta unchanged; no new mathematics.
    #
    # Review fix: this loop recomputes exact_average_delta from scratch for
    # every layer/module on top of the equivalent averaging aggregate_svd()
    # already performed (and discarded) a few lines up, doubling the
    # compute/memory cost of every /aggregate call on the svd path — the same
    # category of duplicated work that OOM-killed the verification VM at real
    # model width. Rather than changing aggregate_svd()'s own internals or
    # return value, the diagnostic is made opt-in: it only runs when the
    # caller explicitly asks for it via AggregateRequest.include_manifest.
    manifest: dict[str, object] = {
        "round_id": _state.round_id,
        "num_clients": len(stored),
        "aggregation": req.aggregation,
        "source_clients": sorted(_state.uploads.keys()),
    }
    if req.include_manifest:
        errors: list[float] = []
        for layer in merged.layer_indices:
            exact = exact_average_delta(iter(adapters), weights, layer=layer)
            for module in merged.target_modules:
                errors.append(
                    float(np.linalg.norm(merged.delta_w(module, layer) - exact[module]))
                )
        manifest["svd_reconstruction_error"] = {
            "mean": float(np.mean(errors)) if errors else 0.0,
            "max": float(np.max(errors)) if errors else 0.0,
        }

    broadcast = ClusterAdapterBroadcast(
        cluster_id="cluster-default",
        round_id=_state.round_id,
        rank=merged.rank,
        target_modules=merged.target_modules,
        alpha=merged.alpha,
        num_layers=merged.num_layers,
        num_clients=len(stored),
        aggregation=req.aggregation,
        peft_config=merged.to_peft_config(),
        tensors=[
            TensorPayload.from_numpy(name, arr)
            for name, arr in merged.to_peft_state_dict().items()
        ],
    )

    _state.active = broadcast
    _state.last_manifest = manifest
    _state.uploads.clear()
    _state.round_id += 1
    return broadcast


@app.get("/adapters/cluster/active", response_model=ClusterAdapterBroadcast)
def get_active_adapter() -> ClusterAdapterBroadcast:
    """Where Edge (or, in the full loop, the registry) pulls the aggregated
    Cluster adapter from. ``tensors`` carry the full PEFT key convention
    (``to_peft_state_dict``) and ``peft_config`` is a ready adapter_config.json
    — together enough to write a directory edge.merge.load_adapter can read."""
    if _state.active is None:
        raise HTTPException(status_code=404, detail="no aggregated cluster adapter yet")
    return _state.active


@app.get("/aggregate/manifest")
def get_last_manifest() -> dict[str, object]:
    if _state.last_manifest is None:
        raise HTTPException(status_code=404, detail="no aggregation has run yet")
    return _state.last_manifest
