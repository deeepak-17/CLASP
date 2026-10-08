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

import re
import time
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from flwr.common import (
    Code,
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
from pydantic import Field as _Field

from cluster.adapter_format import (
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    TARGET_MODULES,
    AdapterFormatError,
    LoRAAdapter,
)
from cluster.aggregation import (
    aggregate_naive,
    aggregate_svd,
    ensure_compatible,
    exact_average_delta,
)
from cluster.clustering import ClusterAssigner
from cluster.integration import SnapshotSink, cluster_adapter_ref
from cluster.redistribution import adapter_from_broadcast, build_broadcast
from cluster.schemas.messages import (
    AdapterUpload,
    ClusterAdapterBroadcast,
)
from cluster.straggler import ClientUpdate, StragglerPolicy, apply_policy


class SVDLoRAStrategy(FedAvg):
    """FedAvg skeleton with delta_W exact-average + SVD re-factorization."""

    def __init__(
        self,
        *args,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
        num_layers: int = 1,
        aggregation: str = "svd",  # "svd" | "naive" (D2 ablation)
        straggler_policy: StragglerPolicy | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if aggregation not in ("svd", "naive"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.straggler_policy = straggler_policy or StragglerPolicy()
        self.round_log: list[dict[str, Scalar]] = []

    def _to_adapter(self, arrays) -> LoRAAdapter:
        return LoRAAdapter.from_ndarrays(
            arrays,
            rank=self.rank,
            alpha=self.alpha,
            target_modules=self.target_modules,
            num_layers=self.num_layers,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Fault-tolerant aggregation (Week 4 Thu / Week 7 Thu).

        1. transport ``failures`` are counted, never aggregated;
        2. results with a non-OK status, or whose payload does not parse as a
           valid adapter, are skipped and counted (one bad client cannot sink
           the round);
        3. the ``StragglerPolicy`` drops over-timeout / empty clients and sets
           the weights (sample-weighted by default);
        4. if the survivors do not reach the policy's quorum the round is
           skipped: ``(None, {})`` is returned and nothing is aggregated.
        """
        t0 = time.monotonic()
        num_failures = len(failures) if failures else 0
        updates: list[ClientUpdate] = []
        skipped: dict[str, str] = {}
        for i, (proxy, fit_res) in enumerate(results):
            metrics = fit_res.metrics or {}
            client_id = str(getattr(proxy, "cid", None) or metrics.get("client_id") or f"client-{i}")
            status = getattr(fit_res, "status", None)
            if status is not None and status.code != Code.OK:
                skipped[client_id] = f"status:{status.code.name}"
                continue
            try:
                adapter = self._to_adapter(parameters_to_ndarrays(fit_res.parameters))
            except (AdapterFormatError, ValueError) as exc:
                skipped[client_id] = f"malformed: {exc}"
                continue
            duration = metrics.get("duration_s")
            loss = metrics.get("loss")
            updates.append(
                ClientUpdate(
                    client_id=client_id,
                    adapter=adapter,
                    num_examples=int(fit_res.num_examples),
                    duration_s=float(duration) if isinstance(duration, (int, float)) else None,
                    loss=float(loss) if isinstance(loss, (int, float)) else None,
                )
            )
        outcome = apply_policy(
            updates, self.straggler_policy, expected=len(results) + num_failures
        )
        skipped.update(outcome.skipped)
        if not outcome.accepted or not outcome.quorum_met:
            self.round_log.append(
                {
                    "round": server_round,
                    "num_clients": 0,
                    "num_failures": num_failures + len(skipped),
                    "aggregation": self.aggregation,
                    "skipped": True,
                }
            )
            return None, {}
        adapters = iter(u.adapter for u in outcome.accepted)
        if self.aggregation == "svd":
            merged = aggregate_svd(adapters, outcome.weights, rank=self.rank)
        else:
            merged = aggregate_naive(adapters, outcome.weights)
        metrics_out: dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(outcome.accepted),
            "num_failures": num_failures + len(skipped),
            "aggregation": self.aggregation,
        }
        losses = [u.loss for u in outcome.accepted if u.loss is not None]
        if losses:
            metrics_out["mean_loss"] = float(sum(losses) / len(losses))
        metrics_out["duration_s"] = time.monotonic() - t0
        self.round_log.append(metrics_out)
        return ndarrays_to_parameters(merged.to_ndarrays()), metrics_out


def build_strategy(
    initial_adapter: LoRAAdapter,
    aggregation: str = "svd",
    min_clients: int = 3,
    straggler_policy: StragglerPolicy | None = None,
) -> SVDLoRAStrategy:
    return SVDLoRAStrategy(
        rank=initial_adapter.rank,
        alpha=initial_adapter.alpha,
        target_modules=initial_adapter.target_modules,
        num_layers=initial_adapter.num_layers,
        aggregation=aggregation,
        straggler_policy=straggler_policy,
        min_fit_clients=min_clients,
        # Flower's default min_evaluate_clients (2) would exceed min_available_clients
        # when min_clients=1 and trigger a misleading startup warning.
        min_evaluate_clients=min(2, min_clients),
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
    __slots__ = ("adapter", "arrived_at", "num_examples", "round_id")

    def __init__(
        self, adapter: LoRAAdapter, num_examples: int, round_id: int, arrived_at: float = 0.0
    ) -> None:
        self.adapter = adapter
        self.num_examples = num_examples
        self.round_id = round_id
        self.arrived_at = arrived_at


DEFAULT_CLUSTER_ID = "cluster-default"
_CLUSTER_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class _ClusterState:
    """One cluster's upload buffer, round counter and active adapter.

    Multi-cluster (Week 8): the service holds one ``_ClusterState`` per cluster
    id in ``_clusters``; the legacy single-cluster endpoints are the same
    handlers bound to ``DEFAULT_CLUSTER_ID``. Buffers never cross clusters —
    every handler takes the state of exactly one cluster. Not thread-safe
    beyond what FastAPI's default single-worker server already serializes.
    """

    def __init__(
        self,
        cluster_id: str = DEFAULT_CLUSTER_ID,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cluster_id = cluster_id
        self.clock = clock  # injectable so straggler timeouts are testable
        self.uploads: dict[str, _StoredUpload] = {}
        self.round_id: int = 0
        self.active: ClusterAdapterBroadcast | None = None
        self.last_manifest: dict[str, object] | None = None
        # Evidence for POST /recluster: client_id -> (adapter it uploaded,
        # adapter it started from) for the contributors of the *last completed*
        # round only. ``None`` start = the client began from the initial adapter
        # (no active adapter existed yet), so its update is its delta_W itself.
        self.last_round: dict[str, tuple[LoRAAdapter, LoRAAdapter | None]] = {}


# The default cluster's state object. Kept as a module-level name (and mutated
# in place, never rebound) because existing tests/clients import it directly.
_state = _ClusterState()
_clusters: dict[str, _ClusterState] = {DEFAULT_CLUSTER_ID: _state}
# client_id -> cluster_id. Explicit assignment only (PUT .../members); a client
# with an assignment may upload only to its own cluster (isolation, W8/W13).
# NOTE: client_id is self-asserted here — binding it to an authenticated
# identity is the mTLS layer's job (P3), not something this service can do.
_membership: dict[str, str] = {}


# ---- client identity hook (the seam where P3's mTLS identity plugs in) -------
#
# ``client_id`` in an upload body is self-asserted. ``configure_identity``
# installs a provider ``(Request) -> str | None`` returning the *authenticated*
# identity of the caller (e.g. the CN of a verified client certificate, relayed
# by whatever terminates TLS). With a provider installed, an upload whose body
# ``client_id`` differs from the authenticated identity is refused (403), and
# with ``require=True`` an unauthenticated upload is refused (401). With no
# provider (the default) behaviour is unchanged. This module does NOT verify
# certificates or issue identities — that is P3 (see docs/INTEGRATION_BOUNDARIES.md).
IdentityProvider = Callable[[Request], "str | None"]


class _IdentityConfig:
    provider: IdentityProvider | None = None
    require: bool = False


_identity = _IdentityConfig()


def configure_identity(provider: IdentityProvider | None, *, require: bool = False) -> None:
    _identity.provider = provider
    _identity.require = require and provider is not None


class _SinkConfig:
    sink: SnapshotSink | None = None


_publishing = _SinkConfig()


def configure_snapshot_sink(sink: SnapshotSink | None) -> None:
    """Publish every aggregated cluster adapter to ``sink`` (the registry seam,
    see ``cluster.integration``). Off by default; a sink error is reported in the
    manifest and never fails the aggregation."""
    _publishing.sink = sink


def _caller_identity(request: Request) -> str | None:
    if _identity.provider is None:
        return None
    try:
        who = _identity.provider(request)
    except Exception as exc:  # a broken provider must fail closed
        raise HTTPException(status_code=401, detail="identity provider failed") from exc
    if who is None and _identity.require:
        raise HTTPException(status_code=401, detail="authenticated client identity required")
    return who

app = FastAPI(
    title="CLASP Cluster Service",
    description="Edge-facing HTTP surface over the existing SVD aggregation core (P2).",
    version="0.2.0",
)


class AggregateRequest(_BaseModel):
    aggregation: str = "svd"  # "svd" | "naive" — same choice SVDLoRAStrategy exposes
    rank: int | None = None  # defaults to the uploaded adapters' own rank
    # Opt-in: also compute the SVD-reconstruction-error manifest below.
    # Off by default: it recomputes exact_average_delta on top of the averaging
    # aggregate_svd() already does, doubling compute/memory per call (the same
    # class of duplicated work that OOM-killed the verification VM at real
    # model width — see test_real_adapter_pipeline.py).
    include_manifest: bool = False
    # Straggler policy (Week 7). A client is a straggler when its upload
    # arrived more than ``straggler_timeout_s`` after the round's first upload.
    straggler_timeout_s: float | None = None
    min_clients: int = 1
    min_fraction: float = 0.0
    weighting: str = "samples"  # "samples" | "uniform"


class MembersRequest(_BaseModel):
    client_ids: list[str]


class ReclusterRequest(_BaseModel):
    """``POST /recluster`` — see the endpoint's docstring for semantics."""

    apply: bool = False  # default is a dry run: report the proposal, change nothing
    min_separation: float = _Field(0.1, ge=0.0)
    min_cluster_size: int = _Field(1, ge=1)
    seed: int = 0
    include_similarity: bool = False  # cosine matrix between clients' updates


def _cluster(cluster_id: str, create: bool = False) -> _ClusterState:
    if not _CLUSTER_ID_RE.match(cluster_id):
        raise HTTPException(status_code=422, detail=f"invalid cluster id {cluster_id!r}")
    state = _clusters.get(cluster_id)
    if state is None:
        if not create:
            raise HTTPException(status_code=404, detail=f"unknown cluster {cluster_id!r}")
        state = _clusters[cluster_id] = _ClusterState(cluster_id)
    return state


def _healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "cluster",
        "pending_uploads": len(_state.uploads),
        "round_id": _state.round_id,
        "has_active_adapter": _state.active is not None,
        "clusters": sorted(_clusters),
    }


def _upload(
    state: _ClusterState, upload: AdapterUpload, authenticated_as: str | None = None
) -> dict[str, object]:
    """Seam A: Edge -> Cluster. Accepts one client's trained LoRA adapter.

    ``upload.tensors`` may use either key convention ``LoRAAdapter`` already
    parses: the short internal 'layers.<i>.<module>.<part>.weight' form, or
    a real PEFT dump like
    'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight'.
    """
    if authenticated_as is not None and authenticated_as != upload.client_id:
        raise HTTPException(
            status_code=403,
            detail=(
                f"authenticated identity {authenticated_as!r} may not upload "
                f"as client {upload.client_id!r}"
            ),
        )
    # Isolation: a client assigned to another cluster may not feed this one.
    owner = _membership.get(upload.client_id)
    if owner is not None and owner != state.cluster_id:
        raise HTTPException(
            status_code=403,
            detail=(
                f"client {upload.client_id!r} is assigned to cluster {owner!r}, "
                f"not {state.cluster_id!r}"
            ),
        )
    if upload.base_cluster_id is not None and upload.base_cluster_id != state.cluster_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"stale_adapter: client {upload.client_id!r} trained from cluster "
                f"{upload.base_cluster_id!r}'s adapter, not {state.cluster_id!r}'s"
            ),
        )
    # Reject uploads for a stale/wrong round instead of silently folding them
    # into whatever round is currently buffering.
    if upload.round_id != state.round_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"upload round_id={upload.round_id} does not match the "
                f"cluster's current round_id={state.round_id}"
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

    # One round = one (rank, alpha, modules, layers) structure: reject a
    # mismatching upload now rather than failing the whole aggregate later.
    for other_id, other in state.uploads.items():
        if other_id == upload.client_id:
            continue
        try:
            ensure_compatible(other.adapter, adapter)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"incompatible with the round's other uploads: {exc}",
            ) from exc
        break  # all buffered uploads already agree with each other

    state.uploads[upload.client_id] = _StoredUpload(
        adapter=adapter,
        num_examples=upload.num_examples,
        round_id=upload.round_id,
        arrived_at=state.clock(),
    )
    return {
        "client_id": upload.client_id,
        "cluster_id": state.cluster_id,
        "round_id": upload.round_id,
        "num_layers": upload.num_layers,
        "tensors_received": len(upload.tensors),
        "status": "accepted",
        "pending_uploads": len(state.uploads),
    }


def _aggregate(state: _ClusterState, request: AggregateRequest | None) -> ClusterAdapterBroadcast:
    """Runs the existing aggregation core over every upload currently
    buffered in ``state``, then clears the buffer and advances that cluster's
    round counter — the same one-round-at-a-time semantics
    ``simulation.run_round`` already uses."""
    if not state.uploads:
        raise HTTPException(status_code=400, detail="no uploads to aggregate")

    req = request or AggregateRequest()
    if req.aggregation not in ("svd", "naive"):
        raise HTTPException(status_code=422, detail=f"unknown aggregation {req.aggregation!r}")
    try:
        policy = StragglerPolicy(
            timeout_s=req.straggler_timeout_s,
            min_clients=req.min_clients,
            min_fraction=req.min_fraction,
            weighting=req.weighting,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    first_arrival = min(u.arrived_at for u in state.uploads.values())
    updates = [
        ClientUpdate(
            client_id=client_id,
            adapter=stored.adapter,
            num_examples=stored.num_examples,
            duration_s=stored.arrived_at - first_arrival,
        )
        for client_id, stored in state.uploads.items()
    ]
    outcome = apply_policy(updates, policy, expected=len(updates))
    if not outcome.accepted or not outcome.quorum_met:
        # keep the buffer so the caller can retry once more clients arrive
        raise HTTPException(
            status_code=409,
            detail=(
                f"quorum not met: {len(outcome.accepted)} usable upload(s), "
                f"{outcome.required} required; skipped={outcome.skipped}"
            ),
        )
    adapters = [u.adapter for u in outcome.accepted]
    weights = outcome.weights
    rank = req.rank or adapters[0].rank

    try:
        if req.aggregation == "svd":
            merged = aggregate_svd(iter(adapters), weights, rank=rank)
        else:
            merged = aggregate_naive(iter(adapters), weights)
    except ValueError as exc:  # incompatible structure / impossible rank
        raise HTTPException(status_code=422, detail=f"cannot aggregate: {exc}") from exc

    manifest: dict[str, object] = {
        "cluster_id": state.cluster_id,
        "round_id": state.round_id,
        "num_clients": len(outcome.accepted),
        "aggregation": req.aggregation,
        "weighting": req.weighting,
        "source_clients": sorted(u.client_id for u in outcome.accepted),
        "skipped_clients": dict(sorted(outcome.skipped.items())),
    }
    if req.include_manifest:
        # SVD reconstruction error vs the exact weighted average, per layer.
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

    broadcast = build_broadcast(
        merged,
        cluster_id=state.cluster_id,
        round_id=state.round_id,
        num_clients=len(outcome.accepted),
        aggregation=req.aggregation,
    )

    start = adapter_from_broadcast(state.active) if state.active is not None else None
    state.last_round = {u.client_id: (u.adapter, start) for u in outcome.accepted}
    if _publishing.sink is not None:
        ref = cluster_adapter_ref(state.cluster_id, state.round_id)
        try:
            _publishing.sink.publish(ref, broadcast)
            manifest["published_as"] = {"name": ref.name, "version": ref.version}
        except Exception as exc:  # noqa: BLE001 - registry trouble must not lose a round
            manifest["publish_error"] = f"{type(exc).__name__}: {exc}"

    state.active = broadcast
    state.last_manifest = manifest
    state.uploads.clear()
    state.round_id += 1
    return broadcast


def _active(state: _ClusterState, request: Request | None = None) -> ClusterAdapterBroadcast:
    """Where Edge (or, in the full loop, the registry) pulls the aggregated
    cluster adapter from. ``tensors`` carry the full PEFT key convention and
    ``peft_config`` is a ready adapter_config.json.

    Retrieval isolation: when an identity provider is installed and the caller
    is a member of a *different* cluster, the request is refused. Callers with
    no membership (registry / evaluation services) are not restricted."""
    caller = _caller_identity(request) if request is not None else None
    owner = _membership.get(caller) if caller is not None else None
    if owner is not None and owner != state.cluster_id:
        raise HTTPException(
            status_code=403,
            detail=(
                f"client {caller!r} belongs to cluster {owner!r} and may not read "
                f"cluster {state.cluster_id!r}'s adapter"
            ),
        )
    if state.active is None:
        raise HTTPException(status_code=404, detail="no aggregated cluster adapter yet")
    return state.active


def _manifest(state: _ClusterState) -> dict[str, object]:
    if state.last_manifest is None:
        raise HTTPException(status_code=404, detail="no aggregation has run yet")
    return state.last_manifest


# ---- legacy single-cluster surface (cluster-default) ----------------------


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return _healthz()


@app.post("/uploads", status_code=201)
def upload_adapter(upload: AdapterUpload, request: Request) -> dict[str, object]:
    return _upload(_state, upload, _caller_identity(request))


@app.post("/aggregate", response_model=ClusterAdapterBroadcast)
def aggregate(request: AggregateRequest | None = None) -> ClusterAdapterBroadcast:
    return _aggregate(_state, request)


@app.get("/adapters/cluster/active", response_model=ClusterAdapterBroadcast)
def get_active_adapter(request: Request) -> ClusterAdapterBroadcast:
    return _active(_state, request)


@app.get("/aggregate/manifest")
def get_last_manifest() -> dict[str, object]:
    return _manifest(_state)


# ---- multi-cluster surface (Week 8) ----------------------------------------


@app.get("/clusters")
def list_clusters() -> dict[str, object]:
    return {
        "clusters": {
            cid: {
                "round_id": st.round_id,
                "pending_uploads": len(st.uploads),
                "has_active_adapter": st.active is not None,
                "members": sorted(c for c, owner in _membership.items() if owner == cid),
            }
            for cid, st in sorted(_clusters.items())
        }
    }


@app.put("/clusters/{cluster_id}/members")
def set_members(cluster_id: str, body: MembersRequest) -> dict[str, object]:
    """Assign ``client_ids`` to this cluster (moving them out of any other).
    Created lazily on first reference. Uploads already buffered by a moved
    client in its old cluster are discarded so they cannot leak into the old
    cluster's aggregate after the move."""
    state = _cluster(cluster_id, create=True)
    for client_id in body.client_ids:
        previous = _membership.get(client_id)
        if previous is not None and previous != cluster_id:
            _clusters[previous].uploads.pop(client_id, None)
        _membership[client_id] = cluster_id
    return {
        "cluster_id": state.cluster_id,
        "members": sorted(c for c, owner in _membership.items() if owner == cluster_id),
    }


@app.post("/clusters/{cluster_id}/uploads", status_code=201)
def upload_adapter_to_cluster(
    cluster_id: str, upload: AdapterUpload, request: Request
) -> dict[str, object]:
    return _upload(_cluster(cluster_id, create=True), upload, _caller_identity(request))


@app.post("/clusters/{cluster_id}/aggregate", response_model=ClusterAdapterBroadcast)
def aggregate_cluster(
    cluster_id: str, request: AggregateRequest | None = None
) -> ClusterAdapterBroadcast:
    return _aggregate(_cluster(cluster_id), request)


@app.get("/clusters/{cluster_id}/adapters/active", response_model=ClusterAdapterBroadcast)
def get_cluster_active_adapter(cluster_id: str, request: Request) -> ClusterAdapterBroadcast:
    return _active(_cluster(cluster_id), request)


@app.get("/clusters/{cluster_id}/aggregate/manifest")
def get_cluster_manifest(cluster_id: str) -> dict[str, object]:
    return _manifest(_cluster(cluster_id))


@app.post("/recluster")
def recluster(body: ReclusterRequest | None = None) -> dict[str, object]:
    """Dynamic re-clustering over HTTP (Week 10, D4) — same algorithm as
    ``MultiClusterFederation``: k-means on the cosine similarity of the clients'
    flattened delta_W updates from the **last completed round**, warm-started
    from the current membership, with the static fallback on any doubt.

    * Evidence is what each cluster's last ``/aggregate`` retained: the adapters
      its accepted contributors uploaded and the cluster adapter they started
      from. Nothing is clustered before a round has been aggregated (409).
    * Default is a **dry run** (``apply=false``): the proposal is returned and
      nothing changes. ``apply=true`` moves clients exactly like
      ``PUT /clusters/{id}/members`` (a moved client's *buffered* upload in its
      old cluster is dropped) and then consumes the evidence, so a second apply
      cannot act on the same round twice.
    * Isolation: the only cross-cluster read is the similarity computation that
      clustering is by definition; the response carries membership and scalar
      statistics, never adapter tensors (the cosine matrix only with
      ``include_similarity=true``). The cluster adapters themselves are still
      only reachable through the per-cluster endpoints. Moved clients must pull
      their new cluster's adapter (HTTP is pull-based) and should send
      ``base_cluster_id`` so a stale adapter is refused (409).
    * Administrative: like ``PUT .../members`` this changes membership and has
      no caller authentication of its own — it belongs behind P3's authorization
      / network policy. Not thread-safe beyond a single worker.
    """
    req = body or ReclusterRequest()
    cluster_ids = sorted({owner for owner in _membership.values()})
    if len(cluster_ids) < 2:
        raise HTTPException(
            status_code=409,
            detail=f"re-clustering needs >= 2 clusters with members, have {cluster_ids}",
        )
    updates: dict[str, LoRAAdapter] = {}
    starts: dict[str, LoRAAdapter | None] = {}
    ignored: dict[str, str] = {}
    for cid in cluster_ids:
        for client_id, (adapter, start) in _clusters[cid].last_round.items():
            if client_id in _membership:
                updates[client_id], starts[client_id] = adapter, start
    for st in _clusters.values():
        for client_id in st.last_round:
            if client_id not in _membership:
                ignored[client_id] = "not assigned to any cluster"
    if not updates:
        raise HTTPException(
            status_code=409,
            detail="no completed aggregation round to cluster: upload and aggregate first",
        )

    completed = max(_clusters[cid].round_id for cid in cluster_ids)
    assigner = ClusterAssigner(
        dict(_membership),
        cluster_ids,
        min_cluster_size=req.min_cluster_size,
        min_separation=req.min_separation,
        seed=req.seed,
    )
    record = assigner.recluster(completed, updates, starts)

    applied = False
    if req.apply and record.method == "dynamic":
        for client_id, (old, new) in record.moved.items():
            _clusters[old].uploads.pop(client_id, None)
            _membership[client_id] = new
        for cid in cluster_ids:
            _clusters[cid].last_round.clear()
        applied = True

    out: dict[str, object] = {
        "applied": applied,
        "method": record.method,  # "dynamic" | "static_fallback"
        "reason": record.reason,
        "completed_rounds": completed,
        "clusters": cluster_ids,
        "clients_clustered": record.clients_clustered,
        "moved": {c: list(v) for c, v in sorted(record.moved.items())},
        "assignment": dict(sorted(record.assignment.items())),
        "cohesion": record.cohesion,
        "separation": record.separation,
        "comparison": record.comparison,
        "ignored": ignored,
    }
    if req.include_similarity:
        out["similarity"] = record.similarity
    return out
