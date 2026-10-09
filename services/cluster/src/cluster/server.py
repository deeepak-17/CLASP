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

import contextlib
import os as _os
import re
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
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
    aggregate_layerwise,
    aggregate_naive,
    aggregate_svd,
    aggregate_svd_lowrank,
    ensure_compatible,
    layerwise_exact_average_error,
)
from cluster.clustering import ClusterAssigner
from cluster.integration import SnapshotSink, cluster_adapter_ref
from cluster.redistribution import adapter_from_broadcast, build_broadcast
from cluster.schemas.messages import (
    DEFAULT_CLUSTER_ID as _DEFAULT_CLUSTER_ID,
)
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
# Wraps the exact-average + SVD aggregation above behind an HTTP surface so
# Edge can reach Cluster over a plain round trip instead of only in-process
# (simulation.py) or over Flower's own wire protocol (SVDLoRAStrategy above,
# used by build_server_app/start_grpc_server for the W7 gRPC path).
#
# NO AGGREGATION MATHEMATICS IS DUPLICATED OR REIMPLEMENTED HERE. Every
# endpoint calls straight into cluster.aggregation and
# cluster.adapter_format.LoRAAdapter.
#
# Two route sets share ONE per-cluster state (``_clusters``), one set of rules
# and one lock per cluster:
#
#   integration surface (sprint, used by demo_ui / Edge / the four-seam test)
#       POST /uploads, POST /aggregate            cluster named in the body
#       GET  /adapters/{id}/active|download|manifest
#       POST /adapters/{id}/publish               seam B: push to the registry
#       GET  /adapters/cluster/active, /aggregate/manifest   default cluster
#
#   cluster-addressed surface (P2: membership, isolation, re-clustering)
#       PUT  /clusters/{id}/members               the only call that *registers*
#       POST /clusters/{id}/uploads|aggregate     404 for an unregistered id
#       GET  /clusters/{id}/adapters/active|aggregate/manifest, GET /clusters
#       POST /recluster
#
# State is in-memory, keyed by cluster_id, and resets on process restart:
# there is no database here. Durable, versioned storage is services/registry's
# job, and /adapters/{cluster_id}/publish is the seam-B hop that hands it over.

DEFAULT_CLUSTER_ID = _DEFAULT_CLUSTER_ID
_CLUSTER_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
#: Guard against a runaway client fan-in; the demo does 3 clients per cluster.
MAX_UPLOADS_PER_CLUSTER = 64
#: Bound on clusters that can exist. ``POST /uploads`` (the integration surface)
#: creates a cluster on its first upload, as the sprint's demo relies on, so
#: without a cap any caller could mint unlimited clusters and fill memory.
MAX_CLUSTERS = 64


class _StoredUpload:
    __slots__ = ("adapter", "arrived_at", "epsilon", "num_examples", "round_id")

    def __init__(
        self,
        adapter: LoRAAdapter,
        num_examples: int,
        round_id: int,
        arrived_at: float = 0.0,
        epsilon: float | None = None,
    ) -> None:
        self.adapter = adapter
        self.num_examples = num_examples
        self.round_id = round_id
        self.arrived_at = arrived_at
        self.epsilon = epsilon


class _ClusterState:
    """One cluster's upload buffer, round counter and active adapter.

    The service holds one ``_ClusterState`` per cluster id in ``_clusters``; the
    legacy single-cluster endpoints are the same handlers bound to
    ``DEFAULT_CLUSTER_ID``. Buffers never cross clusters — every handler takes
    the state of exactly one cluster.

    Concurrency: the handlers are plain ``def`` functions, so FastAPI runs them
    on a thread pool and several can touch one cluster at the same time, even
    with a single worker process. ``lock`` serializes every read-modify-write of
    ``uploads``, ``round_id``, ``active``, ``last_manifest`` and ``last_round``:
    ``_upload`` and ``_aggregate`` hold it for their whole body, so an upload can
    neither be iterated while it is added nor be cleared without having been
    aggregated. An upload that waits behind a running aggregate then sees the
    advanced ``round_id`` and gets an honest 409, never a 201 that is later
    dropped. ``publish_to_registry`` takes it to read ``active`` (and so waits
    for a running aggregate instead of publishing the round before it). Lock
    order (to stay deadlock-free): ``_registry_lock`` first, then at most one
    cluster lock — except ``recluster``, which takes every cluster lock in
    sorted id order. Code holding a cluster lock never takes ``_registry_lock``.

    The lock is per process: with several worker processes each has its own
    state anyway, so run this service with one worker.
    """

    def __init__(
        self,
        cluster_id: str = DEFAULT_CLUSTER_ID,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cluster_id = cluster_id
        self.clock = clock  # injectable so straggler timeouts are testable
        self.lock = threading.Lock()
        self.uploads: dict[str, _StoredUpload] = {}
        self.round_id: int = 0
        self.active: ClusterAdapterBroadcast | None = None
        self.last_manifest: dict[str, object] | None = None
        # Evidence for POST /recluster: client_id -> (adapter it uploaded,
        # adapter it started from) for the contributors of the *last completed*
        # round only. ``None`` start = the client began from the initial adapter
        # (no active adapter existed yet), so its update is its delta_W itself.
        self.last_round: dict[str, tuple[LoRAAdapter, LoRAAdapter | None]] = {}


#: cluster_id -> state. Two statically-assigned clusters in Phase II (D1:
#: "web" and "scientific"); further clusters are registered with
#: ``PUT /clusters/{id}/members`` or created by their first ``POST /uploads``.
_clusters: dict[str, _ClusterState] = {DEFAULT_CLUSTER_ID: _ClusterState()}

#: The default cluster's state, exported under its historical name so callers
#: written against the single-cluster server keep working unchanged. Mutated in
#: place, never rebound.
_state = _clusters[DEFAULT_CLUSTER_ID]

# client_id -> cluster_id. Explicit assignment only (PUT .../members); a client
# with an assignment may upload only to its own cluster (isolation, W8/W13).
# NOTE: client_id is self-asserted here — binding it to an authenticated
# identity is the mTLS layer's job (P3), not something this service can do.
_membership: dict[str, str] = {}
# Guards *mutation* of ``_clusters`` / ``_membership`` (creating a cluster,
# moving clients). Lock-free readers take a ``dict(...)`` snapshot instead of
# iterating the live dict, so they can never see "changed size during iteration".
_registry_lock = threading.RLock()


def _reset_clusters() -> None:
    """Drop every cluster's buffer and every membership, keeping the default
    cluster's identity (used by tests that share one interpreter)."""
    with _registry_lock:
        _clusters.clear()
        _clusters[DEFAULT_CLUSTER_ID] = _state
        _membership.clear()
        with _state.lock:
            _state.uploads.clear()
            _state.round_id = 0
            _state.active = None
            _state.last_manifest = None
            _state.last_round.clear()


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
#
# STATUS: a hook for later. Nothing in the running service calls
# ``configure_identity`` (or ``configure_snapshot_sink`` below) — only tests do —
# so by default uploads are NOT bound to a client certificate and nothing is
# published to a registry. mTLS is not finished; do not read this seam as it.
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
    #: Which cluster ``POST /aggregate`` runs over. Ignored by
    #: ``POST /clusters/{id}/aggregate``, where the path names the cluster (an
    #: explicit, different value there is a 422).
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
    #: A retained run is an ablation, not a completed round: it is not published
    #: to a snapshot sink and is not evidence for ``/recluster``.
    retain_uploads: bool = False
    # Straggler policy (Week 7). A client is a straggler when its upload
    # arrived more than ``straggler_timeout_s`` after the round's first upload.
    straggler_timeout_s: float | None = None
    min_clients: int = 1
    min_fraction: float = 0.0
    # Denominator for ``min_fraction``. Default: the number of clients assigned
    # to the cluster (PUT .../members); with nobody assigned, the uploads that
    # arrived (so min_fraction cannot bite — assign members or set this).
    expected_clients: int | None = _Field(None, ge=1)
    weighting: str = "samples"  # "samples" | "uniform"


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
    """The state of one cluster. ``create=False`` (the default) never creates:
    an unregistered id is a 404, so a typo cannot silently open a new cluster."""
    if not _CLUSTER_ID_RE.match(cluster_id):
        raise HTTPException(status_code=422, detail=f"invalid cluster id {cluster_id!r}")
    state = _clusters.get(cluster_id)
    if state is None:
        if not create:
            raise HTTPException(status_code=404, detail=f"unknown cluster {cluster_id!r}")
        with _registry_lock:  # two racing creators must end up sharing one state
            state = _clusters.get(cluster_id)
            if state is None:
                if len(_clusters) >= MAX_CLUSTERS:
                    raise HTTPException(
                        status_code=429,
                        detail=f"already hosting {MAX_CLUSTERS} clusters; refusing to create "
                        f"{cluster_id!r}",
                    )
                state = _clusters[cluster_id] = _ClusterState(cluster_id)
    return state


def _healthz() -> dict[str, object]:
    """Liveness plus per-cluster round bookkeeping.

    The top-level counters describe the DEFAULT cluster (unchanged from the
    single-cluster surface); ``clusters`` breaks them out per cluster_id. Takes
    no cluster lock, so a probe never queues behind a running aggregate.
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
            for cid, st in sorted(dict(_clusters).items())
        },
    }


def _upload(
    state: _ClusterState, upload: AdapterUpload, authenticated_as: str | None = None
) -> dict[str, object]:
    """Seam A: Edge -> Cluster. Accepts one client's trained LoRA adapter.

    ``upload.tensors`` may use either key convention ``LoRAAdapter`` already
    parses: the short internal 'layers.<i>.<module>.<part>.weight' form, or
    a real PEFT dump like
    'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight'.

    Holds the cluster's lock for the whole check-and-store, so it is atomic with
    respect to ``_aggregate`` and to membership moves (see ``_ClusterState``).
    """
    with state.lock:
        return _upload_locked(state, upload, authenticated_as)


def _upload_locked(
    state: _ClusterState, upload: AdapterUpload, authenticated_as: str | None
) -> dict[str, object]:
    """Body of ``_upload``; the caller must hold ``state.lock``."""
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
                f"upload round_id={upload.round_id} does not match "
                f"{state.cluster_id}'s current round_id={state.round_id}"
            ),
        )
    if upload.client_id not in state.uploads and len(state.uploads) >= MAX_UPLOADS_PER_CLUSTER:
        raise HTTPException(
            status_code=429,
            detail=f"{state.cluster_id} already holds {MAX_UPLOADS_PER_CLUSTER} uploads",
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
        epsilon=upload.epsilon,
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


def _expected_clients(cluster_id: str, requested: int | None) -> int | None:
    """How many clients this round was *expected* to hear from (the denominator
    of ``min_fraction``): an explicit ``expected_clients`` wins, else the number
    of clients assigned to the cluster, else ``None`` (= nobody is assigned, so
    fall back to the uploads that arrived)."""
    if requested is not None:
        return requested
    with _registry_lock:
        members = sum(1 for owner in _membership.values() if owner == cluster_id)
    return members or None


def cluster_epsilon(epsilons: Sequence[float | None]) -> float | None:
    """The privacy budget (D7) to record for an aggregate: the **max** ε over the
    aggregated clients — the loosest guarantee any contribution carries.

    ``None`` as soon as one aggregated client did not report an ε: that client
    was trained without DP (or did not say), so no finite ε describes the
    aggregate and claiming the others' maximum would understate the privacy
    loss. This mirrors the registry's composition rule, where a part trained
    without DP makes the composite's ε ``None``.
    """
    if not epsilons or any(e is None for e in epsilons):
        return None
    return float(max(e for e in epsilons if e is not None))


def _aggregate_target(
    request: AggregateRequest | None, path_cluster_id: str | None
) -> tuple[_ClusterState, AggregateRequest]:
    """Resolve which cluster an aggregate runs over (the body's ``cluster_id`` on
    ``POST /aggregate``, the path on ``POST /clusters/{id}/aggregate``)."""
    req = request or AggregateRequest()
    if path_cluster_id is None:
        return _cluster(req.cluster_id), req
    if "cluster_id" in req.model_fields_set and req.cluster_id != path_cluster_id:
        raise HTTPException(
            status_code=422,
            detail=(
                f"body cluster_id {req.cluster_id!r} does not match the path's "
                f"cluster {path_cluster_id!r}"
            ),
        )
    return _cluster(path_cluster_id), req


def _aggregate(state: _ClusterState, request: AggregateRequest | None) -> ClusterAdapterBroadcast:
    """Runs the existing aggregation core over every upload currently
    buffered in ``state``, then clears the buffer and advances that cluster's
    round counter — the same one-round-at-a-time semantics
    ``simulation.run_round`` already uses.

    The cluster lock is held for the whole call. ``aggregate_svd`` can take
    seconds on real adapters; an upload arriving meanwhile waits and is then
    judged against the *new* round (409 if it was for the one just aggregated),
    instead of being accepted into a buffer that is about to be cleared. Reads
    that do not need a consistent buffer (``/healthz``, ``/clusters``, the
    active-adapter and manifest GETs) take no lock and are not blocked.
    """
    expected = _expected_clients(
        state.cluster_id, request.expected_clients if request is not None else None
    )  # before the cluster lock: lock order is registry -> cluster, never reverse
    with state.lock:
        return _aggregate_locked(state, request, expected)


def _aggregate_locked(
    state: _ClusterState, request: AggregateRequest | None, expected: int | None
) -> ClusterAdapterBroadcast:
    """Body of ``_aggregate``; the caller must hold ``state.lock``."""
    if not state.uploads:
        raise HTTPException(
            status_code=400, detail=f"no uploads to aggregate for {state.cluster_id}"
        )

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

    client_ids = sorted(state.uploads)  # deterministic order => reproducible floats
    first_arrival = min(u.arrived_at for u in state.uploads.values())
    updates = [
        ClientUpdate(
            client_id=client_id,
            adapter=state.uploads[client_id].adapter,
            num_examples=state.uploads[client_id].num_examples,
            duration_s=state.uploads[client_id].arrived_at - first_arrival,
        )
        for client_id in client_ids
    ]
    # ``min_fraction`` is a fraction of the clients the round *expected*, not of
    # the uploads that happened to arrive (that would always be 100%). Never
    # below what actually arrived.
    expected_n = max(len(updates), expected or 0)
    outcome = apply_policy(updates, policy, expected=expected_n)
    if not outcome.accepted or not outcome.quorum_met:
        # keep the buffer so the caller can retry once more clients arrive
        raise HTTPException(
            status_code=409,
            detail=(
                f"quorum not met: {len(outcome.accepted)} usable upload(s), "
                f"{outcome.required} required (of {expected_n} expected); "
                f"skipped={outcome.skipped}"
            ),
        )
    accepted = outcome.accepted
    adapters = [u.adapter for u in accepted]
    weights = [float(w) for w in outcome.weights]
    used_ids = [u.client_id for u in accepted]
    rank = req.rank or adapters[0].rank

    epsilon = cluster_epsilon([state.uploads[c].epsilon for c in used_ids])

    manifest: dict[str, object] = {
        "cluster_id": state.cluster_id,
        "round_id": state.round_id,
        "num_clients": len(accepted),
        "expected_clients": expected_n,
        "aggregation": req.aggregation,
        "weighting": req.weighting,
        "source_clients": used_ids,
        "skipped_clients": dict(sorted(outcome.skipped.items())),
        "sample_weights": [float(u.num_examples) for u in accepted],
        "rank": rank,
        "epsilon": epsilon,
    }
    if epsilon is None:
        manifest["epsilon_not_reported_by"] = sorted(
            c for c in used_ids if state.uploads[c].epsilon is None
        )

    try:
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
    except ValueError as exc:  # incompatible structure / impossible rank
        raise HTTPException(status_code=422, detail=f"cannot aggregate: {exc}") from exc

    broadcast = build_broadcast(
        merged,
        cluster_id=state.cluster_id,
        round_id=state.round_id,
        num_clients=len(accepted),
        aggregation=req.aggregation,
        source_clients=used_ids,
        epsilon=epsilon,
    )

    completed = not req.retain_uploads
    if completed:
        # Evidence for POST /recluster and the optional snapshot sink: only a
        # completed round counts, not a retained ablation run.
        start = adapter_from_broadcast(state.active) if state.active is not None else None
        state.last_round = {u.client_id: (u.adapter, start) for u in accepted}
        if _publishing.sink is not None:
            ref = cluster_adapter_ref(state.cluster_id, state.round_id)
            try:
                _publishing.sink.publish(ref, broadcast)
                manifest["published_as"] = {"name": ref.name, "version": ref.version}
            except Exception as exc:  # noqa: BLE001 - registry trouble must not lose a round
                manifest["publish_error"] = f"{type(exc).__name__}: {exc}"

    state.active = broadcast
    manifest["retained_uploads"] = req.retain_uploads
    state.last_manifest = manifest
    if completed:
        state.uploads.clear()
        state.round_id += 1
    return broadcast


def _read_access(state: _ClusterState, request: Request | None) -> None:
    """Retrieval isolation: when an identity provider is installed and the caller
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


def _active(state: _ClusterState | None, cluster_id: str, request: Request | None = None):
    """Where Edge (or, in the full loop, the registry) pulls the aggregated
    cluster adapter from. ``tensors`` carry the full PEFT key convention and
    ``peft_config`` is a ready adapter_config.json. Never creates a cluster."""
    if state is not None:
        _read_access(state, request)
    active = state.active if state is not None else None  # one atomic read
    if active is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregated cluster adapter for {cluster_id!r}"
        )
    return active


def _manifest(state: _ClusterState | None, cluster_id: str) -> dict[str, object]:
    manifest = state.last_manifest if state is not None else None
    if manifest is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregation has run for {cluster_id!r}"
        )
    return manifest


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


# ---- integration surface: cluster named in the body (sprint B2/B3) ----------


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return _healthz()


@app.post("/uploads", status_code=201)
def upload_adapter(upload: AdapterUpload, request: Request) -> dict[str, object]:
    """Seam A, cluster named by ``upload.cluster_id`` (default cluster when
    omitted). The cluster is created on its first upload here — the sprint demo
    relies on it — bounded by ``MAX_CLUSTERS``; use ``PUT /clusters/{id}/members``
    plus ``POST /clusters/{id}/uploads`` where a mistyped id must be a 404."""
    return _upload(_cluster(upload.cluster_id, create=True), upload, _caller_identity(request))


@app.post("/aggregate", response_model=ClusterAdapterBroadcast)
def aggregate(request: AggregateRequest | None = None) -> ClusterAdapterBroadcast:
    state, req = _aggregate_target(request, None)
    return _aggregate(state, req)


# NOTE: the literal '/adapters/cluster/active' route is declared BEFORE the
# parameterized one on purpose. FastAPI matches routes in declaration order, so
# this keeps the original single-cluster path meaning "the default cluster"
# instead of resolving cluster_id="cluster" to an empty new cluster.
@app.get("/adapters/cluster/active", response_model=ClusterAdapterBroadcast)
def get_default_active_adapter(request: Request) -> ClusterAdapterBroadcast:
    """Single-cluster alias for the default cluster's active aggregate."""
    return _active(_clusters.get(DEFAULT_CLUSTER_ID), DEFAULT_CLUSTER_ID, request)


@app.get("/adapters/{cluster_id}/active", response_model=ClusterAdapterBroadcast)
def get_active_adapter(cluster_id: str, request: Request) -> ClusterAdapterBroadcast:
    """Where the orchestrator pulls a cluster's aggregate from before the
    registry hop. ``tensors`` carry the full PEFT key convention
    (``to_peft_state_dict``) and ``peft_config`` is a ready adapter_config.json
    — together enough to write a directory ``edge.merge.load_adapter`` reads."""
    return _active(_clusters.get(cluster_id), cluster_id, request)


@app.get("/adapters/{cluster_id}/download")
def download_adapter(cluster_id: str, request: Request) -> Response:
    """The active aggregate as raw safetensors bytes — byte-for-byte the
    payload seam B stores in the registry, with adapter_config.json embedded in
    the file's own ``__metadata__`` header."""
    active = _active(_clusters.get(cluster_id), cluster_id, request)
    return Response(
        content=_peft_bytes(active),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{cluster_id}.safetensors"',
            "X-CLASP-Round": str(active.round_id),
            "X-CLASP-Source-Clients": ",".join(active.source_clients),
        },
    )


@app.get("/aggregate/manifest")
def get_last_manifest() -> dict[str, object]:
    """Single-cluster alias for the default cluster's last aggregation manifest."""
    return _manifest(_clusters.get(DEFAULT_CLUSTER_ID), DEFAULT_CLUSTER_ID)


@app.get("/adapters/{cluster_id}/manifest")
def get_cluster_manifest(cluster_id: str) -> dict[str, object]:
    return _manifest(_clusters.get(cluster_id), cluster_id)


@app.post("/adapters/{cluster_id}/publish", status_code=201)
def publish_to_registry(cluster_id: str, request: PublishRequest | None = None) -> dict:
    """Seam B: Cluster -> Registry. POST the active aggregate as a new version.

    This is the one place Cluster talks to another service. It sends exactly
    what ``/download`` serves, plus the metadata envelope the registry records:
    aggregation method, source clients, round, the LoRA hyperparameters and —
    when every aggregated client reported one — the cluster's privacy budget
    (D7: the max ε over the aggregated clients). Every field is read off the
    aggregate itself, none typed by hand. The registry computes and stores its
    own sha256 over the payload; Edge verifies the download against that on the
    way back out (seam C1).

    The cluster lock is taken to read ``active`` (so a publish that arrives
    during an aggregate waits for it and sends the new round, never a torn
    one) and released before the network call.
    """
    import json as _json

    import httpx

    req = request or PublishRequest()
    state = _clusters.get(cluster_id)
    broadcast = None
    if state is not None:
        with state.lock:
            broadcast = state.active
    if broadcast is None:
        raise HTTPException(
            status_code=404, detail=f"no aggregated cluster adapter for {cluster_id!r}"
        )
    name = req.adapter_name or cluster_id
    payload = _peft_bytes(broadcast)
    cfg = broadcast.peft_config or {}
    meta: dict[str, object] = {
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
    if broadcast.epsilon is not None:
        # D7: without this every cluster (and every composite built from it) is
        # recorded in the registry with epsilon = null.
        meta["privacy"] = {"epsilon": broadcast.epsilon}

    from cluster.tls import outbound_ssl_context

    # Under mTLS (CLASP_TLS_* set, see cluster.serve) the registry gets the
    # cluster's certificate and must present one the CLASP CA signed.
    tls = outbound_ssl_context(_os.environ) if req.registry_url.startswith("https") else None
    try:
        with httpx.Client(timeout=req.timeout_s, verify=tls if tls is not None else True) as http:
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


# ---- cluster-addressed surface (P2: membership, isolation, re-clustering) ----


@app.get("/clusters")
def list_clusters() -> dict[str, object]:
    # Snapshots, not the live dicts: another request may be creating a cluster
    # or moving a client while this one iterates. Takes no cluster lock, so it
    # answers even while an aggregate is running.
    clusters = dict(_clusters)
    membership = dict(_membership)
    out = {}
    for cid, st in sorted(clusters.items()):
        uploaded = sorted(st.uploads)  # one snapshot; len() is taken from it too
        active = st.active  # one atomic read
        out[cid] = {
            "round_id": st.round_id,
            "pending_uploads": len(uploaded),
            "has_active_adapter": active is not None,
            "members": sorted(c for c, owner in membership.items() if owner == cid),
            # Who has uploaded into the round that is buffering now — what the
            # demo panel shows arriving, as opposed to ``members`` (assigned).
            "uploaded_clients": uploaded,
            # Who the active aggregate was built from (empty before the first).
            "aggregated_clients": list(active.source_clients) if active is not None else [],
            "aggregated_round_id": active.round_id if active is not None else None,
        }
    return {"clusters": out}


@app.put("/clusters/{cluster_id}/members")
def set_members(cluster_id: str, body: MembersRequest) -> dict[str, object]:
    """Assign ``client_ids`` to this cluster (moving them out of any other).
    Created if new. Uploads already buffered by a moved client in its old
    cluster are discarded so they cannot leak into the old cluster's aggregate
    after the move.

    This is the call that *registers* a cluster for the cluster-addressed
    surface: ``POST /clusters/{id}/uploads`` and ``.../aggregate`` answer 404
    for an id nobody registered, so a typo cannot silently open a new cluster.

    Order matters for isolation: membership is changed first (under the registry
    lock), the stale buffered uploads are dropped afterwards under each old
    cluster's lock. An upload that checked the old membership either finished
    before that drop (and is removed by it) or is refused with 403."""
    state = _cluster(cluster_id, create=True)
    moved_from: list[tuple[str, str]] = []
    with _registry_lock:
        for client_id in body.client_ids:
            previous = _membership.get(client_id)
            if previous is not None and previous != cluster_id:
                moved_from.append((previous, client_id))
            _membership[client_id] = cluster_id
        members = sorted(c for c, owner in _membership.items() if owner == cluster_id)
    for previous, client_id in moved_from:
        old = _clusters[previous]
        with old.lock:  # may wait for a running aggregate of the old cluster
            old.uploads.pop(client_id, None)
    return {"cluster_id": state.cluster_id, "members": members}


def _path_matches_body(cluster_id: str, upload: AdapterUpload) -> None:
    if "cluster_id" in upload.model_fields_set and upload.cluster_id != cluster_id:
        raise HTTPException(
            status_code=422,
            detail=(
                f"body cluster_id {upload.cluster_id!r} does not match the path's "
                f"cluster {cluster_id!r}"
            ),
        )


@app.post("/clusters/{cluster_id}/uploads", status_code=201)
def upload_adapter_to_cluster(
    cluster_id: str, upload: AdapterUpload, request: Request
) -> dict[str, object]:
    # create=False: only ``PUT /clusters/{id}/members`` registers a cluster, so a
    # mistyped id is a 404 and callers cannot mint unlimited clusters.
    state = _cluster(cluster_id)
    _path_matches_body(cluster_id, upload)
    return _upload(state, upload, _caller_identity(request))


@app.post("/clusters/{cluster_id}/aggregate", response_model=ClusterAdapterBroadcast)
def aggregate_cluster(
    cluster_id: str, request: AggregateRequest | None = None
) -> ClusterAdapterBroadcast:
    state, req = _aggregate_target(request, cluster_id)
    return _aggregate(state, req)


@app.get("/clusters/{cluster_id}/adapters/active", response_model=ClusterAdapterBroadcast)
def get_cluster_active_adapter(cluster_id: str, request: Request) -> ClusterAdapterBroadcast:
    return _active(_cluster(cluster_id), cluster_id, request)


@app.get("/clusters/{cluster_id}/aggregate/manifest")
def get_cluster_manifest_by_path(cluster_id: str) -> dict[str, object]:
    return _manifest(_cluster(cluster_id), cluster_id)


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
      / network policy.
    * Concurrency: holds the registry lock and *every* cluster lock (sorted id
      order) for its whole run, so the membership, the retained evidence and the
      buffers it reads and edits cannot change underneath it. Uploads and
      aggregates for any cluster wait until it is done; it in turn waits for a
      running aggregate. Run it between rounds.
    """
    req = body or ReclusterRequest()
    with _registry_lock:
        states = sorted(_clusters.items())
        with contextlib.ExitStack() as held:
            for _cid, st in states:
                held.enter_context(st.lock)
            return _recluster_locked(req)


def _recluster_locked(req: ReclusterRequest) -> dict[str, object]:
    """Body of ``recluster``; the caller holds the registry lock and every cluster lock."""
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
