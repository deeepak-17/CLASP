"""Multi-cluster federated rounds (Week 8 + Week 10, P2).

``MultiClusterFederation`` is the piece that ties the P2 parts into one round
loop, reusing — not re-implementing — each of them:

    clients (Flower NumPyClient.fit, FedProx)           cluster.client / simulation
      -> per-cluster straggler policy                   cluster.straggler
      -> per-cluster delta_W exact-average + SVD        cluster.aggregation
      -> dynamic re-clustering / static fallback        cluster.clustering
      -> per-cluster redistribution with retry          cluster.redistribution

Cluster isolation is a structural rule, not a convention: a client only ever
trains from, contributes to, and receives the adapter of the cluster it is
*currently assigned* to. ``isolation_violations()`` re-checks that rule against
the recorded history, and the tests assert it is empty (and non-empty when an
adversarial delivery is injected, so the check itself is proven to bite).

Order inside a round (documented because it decides what "isolated" means):
every cluster aggregates the updates of the clients that *trained from its
adapter this round*; only then is the assignment re-evaluated, and the new
members receive their (possibly new) cluster's adapter for the next round.
A client moved by re-clustering therefore never contributes to — or trains
from — a cluster it did not belong to at the start of that round.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from contracts.types import AdapterRef
from flwr.client import NumPyClient

from cluster.adapter_format import AdapterFormatError, LoRAAdapter
from cluster.aggregation import aggregate_naive, aggregate_svd
from cluster.clustering import AssignmentRecord, ClusterAssigner
from cluster.integration import SnapshotSink, cluster_adapter_ref
from cluster.redistribution import (
    DeliveryReport,
    adapter_from_broadcast,
    build_broadcast,
    redistribute,
)
from cluster.schemas.messages import ClusterAdapterBroadcast, RoundMetrics
from cluster.straggler import ClientUpdate, StragglerPolicy, apply_policy


@dataclass
class ClusterRoundResult:
    cluster_id: str
    round_id: int
    members: list[str]  # assigned to this cluster when the round started
    contributors: list[str]  # accepted into the aggregate
    skipped: dict[str, str]  # client_id -> reason (timeout / no_examples / offline / ...)
    aggregated: bool  # False => quorum not met, previous adapter kept
    adapter: LoRAAdapter
    metrics: RoundMetrics
    delivery: DeliveryReport | None = None
    # registry hand-off (only when a SnapshotSink is configured and the cluster aggregated)
    published: AdapterRef | None = None
    publish_error: str | None = None


@dataclass
class FederatedRound:
    round_id: int
    clusters: dict[str, ClusterRoundResult]
    assignment_before: dict[str, str]
    assignment_after: dict[str, str]
    recluster: AssignmentRecord | None = None
    # accepted client updates of this round and the adapters they started from
    # (kept so re-clustering decisions can be re-derived / audited offline)
    updates: dict[str, LoRAAdapter] = field(default_factory=dict)
    starts: dict[str, LoRAAdapter] = field(default_factory=dict)


@dataclass
class _Contribution:
    round_id: int
    cluster_id: str
    members: tuple[str, ...]
    contributors: tuple[str, ...]


class MultiClusterFederation:
    """Round driver over several clusters with a shared client pool."""

    def __init__(
        self,
        clients: Mapping[str, NumPyClient],
        assigner: ClusterAssigner,
        initial_adapter: LoRAAdapter,
        *,
        aggregation: str = "svd",
        policy: StragglerPolicy | None = None,
        send: Callable[[str, ClusterAdapterBroadcast], None] | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.0,
        sleep: Callable[[float], None] | None = None,
        fit_config: Mapping[str, object] | None = None,
        sink: SnapshotSink | None = None,
    ) -> None:
        if aggregation not in ("svd", "naive"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        missing = sorted(set(clients) - set(assigner.assignment))
        if missing:
            raise ValueError(f"clients without a cluster assignment: {missing}")
        initial_adapter.validate()
        self.clients = dict(clients)
        self.assigner = assigner
        self.aggregation = aggregation
        self.policy = policy or StragglerPolicy()
        self.fit_config = dict(fit_config or {})
        self._send = send
        self._sink = sink
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep or (lambda _s: None)

        self.rank = initial_adapter.rank
        self.completed_rounds = 0
        # cluster_id -> current cluster adapter (server side)
        self.cluster_adapters: dict[str, LoRAAdapter] = {
            cid: _copy_adapter(initial_adapter) for cid in assigner.cluster_ids
        }
        # client_id -> adapter it last *received* (decoded from the wire message)
        self.client_adapters: dict[str, LoRAAdapter] = {
            cid: self.cluster_adapters[assigner.assignment[cid]] for cid in self.clients
        }
        # client_id -> (cluster_id, round_id) of the last broadcast it received
        self.received: dict[str, tuple[str, int]] = {}
        self.rounds: list[FederatedRound] = []
        self._contributions: list[_Contribution] = []

    # ------------------------------------------------------------------

    def run_round(self, offline: Iterable[str] = ()) -> FederatedRound:
        """One federated round. ``offline`` clients never answer (dropout)."""
        round_id = self.completed_rounds
        offline = set(offline)
        t_round = time.monotonic()
        before = dict(self.assigner.assignment)

        members: dict[str, list[str]] = {cid: [] for cid in self.assigner.cluster_ids}
        for client_id in sorted(self.clients):
            members[before[client_id]].append(client_id)

        updates: dict[str, list[ClientUpdate]] = {cid: [] for cid in members}
        skipped: dict[str, dict[str, str]] = {cid: {} for cid in members}
        starts: dict[str, LoRAAdapter] = {}
        for cluster_id, ids in members.items():
            for client_id in ids:
                if client_id in offline:
                    skipped[cluster_id][client_id] = "offline"
                    continue
                held = self.received.get(client_id)
                if held is not None and held[0] != cluster_id:
                    # redistribution to this client failed after its move: it still
                    # holds another cluster's adapter, so it must not train this round
                    skipped[cluster_id][client_id] = f"stale_adapter (holds {held[0]})"
                    continue
                start = self.client_adapters[client_id]
                template = self.cluster_adapters[cluster_id]
                try:
                    arrays, num_examples, metrics = self.clients[client_id].fit(
                        [a.copy() for a in start.to_ndarrays()], dict(self.fit_config)
                    )
                    adapter = LoRAAdapter.from_ndarrays(
                        arrays,
                        rank=template.rank,
                        alpha=template.alpha,
                        target_modules=template.target_modules,
                        num_layers=template.num_layers,
                    )
                except AdapterFormatError as exc:
                    skipped[cluster_id][client_id] = f"malformed: {exc}"
                    continue
                except Exception as exc:  # noqa: BLE001 - a crashing client is a dropout
                    skipped[cluster_id][client_id] = f"failed: {type(exc).__name__}: {exc}"
                    continue
                starts[client_id] = start
                loss = metrics.get("loss") if metrics else None
                duration = metrics.get("duration_s") if metrics else None
                updates[cluster_id].append(
                    ClientUpdate(
                        client_id=client_id,
                        adapter=adapter,
                        num_examples=int(num_examples),
                        duration_s=float(duration) if duration is not None else None,
                        loss=float(loss) if loss is not None else None,
                    )
                )

        # ---- per-cluster aggregation (old assignment) ------------------
        results: dict[str, ClusterRoundResult] = {}
        accepted_adapters: dict[str, LoRAAdapter] = {}
        for cluster_id, ids in members.items():
            outcome = apply_policy(updates[cluster_id], self.policy, expected=len(ids))
            skipped[cluster_id].update(outcome.skipped)
            prev = self.cluster_adapters[cluster_id]
            aggregated = bool(outcome.accepted) and outcome.quorum_met
            if aggregated:
                adapters_iter = iter(u.adapter for u in outcome.accepted)
                if self.aggregation == "svd":
                    merged = aggregate_svd(adapters_iter, outcome.weights, rank=self.rank)
                else:
                    merged = aggregate_naive(adapters_iter, outcome.weights)
                self.cluster_adapters[cluster_id] = merged
                for u in outcome.accepted:
                    accepted_adapters[u.client_id] = u.adapter
            else:
                merged = prev
                if ids:
                    reason = "quorum_not_met" if outcome.accepted else "no_updates"
                    for u in outcome.accepted:
                        skipped[cluster_id].setdefault(u.client_id, reason)
            losses = [u.loss for u in outcome.accepted if u.loss is not None]
            contributors = [u.client_id for u in outcome.accepted] if aggregated else []
            self._contributions.append(
                _Contribution(round_id, cluster_id, tuple(ids), tuple(contributors))
            )
            results[cluster_id] = ClusterRoundResult(
                cluster_id=cluster_id,
                round_id=round_id,
                members=list(ids),
                contributors=contributors,
                skipped=dict(skipped[cluster_id]),
                aggregated=aggregated,
                adapter=merged,
                metrics=RoundMetrics(
                    round_id=round_id,
                    num_clients=len(contributors),
                    mean_loss=(sum(losses) / len(losses)) if losses and aggregated else None,
                    duration_s=None,  # filled below once the round is complete
                    aggregation=self.aggregation,
                ),
            )

        # ---- dynamic re-clustering (after this round's aggregation) ----
        self.completed_rounds += 1
        record = self.assigner.maybe_recluster(
            self.completed_rounds,
            {cid: accepted_adapters[cid] for cid in accepted_adapters},
            {cid: starts[cid] for cid in accepted_adapters},
        )
        after = dict(self.assigner.assignment)

        # ---- redistribution to the *current* members of every cluster ---
        for cluster_id, result in results.items():
            new_members = sorted(c for c, cl in after.items() if cl == cluster_id and c in self.clients)
            broadcast = build_broadcast(
                result.adapter,
                cluster_id=cluster_id,
                round_id=round_id,
                num_clients=max(len(result.contributors), 1),
                aggregation=self.aggregation,
            )
            decoded = adapter_from_broadcast(broadcast)
            if self._sink is not None and result.aggregated:
                ref = cluster_adapter_ref(cluster_id, round_id)
                try:
                    self._sink.publish(ref, broadcast)
                    result.published = ref
                except Exception as exc:  # noqa: BLE001 - registry trouble must not abort a round
                    result.publish_error = f"{type(exc).__name__}: {exc}"
            result.delivery = redistribute(
                broadcast,
                new_members,
                self._deliver(decoded),
                max_retries=self._max_retries,
                backoff_s=self._backoff_s,
                sleep=self._sleep,
            )

        elapsed = time.monotonic() - t_round
        for result in results.values():
            result.metrics = result.metrics.model_copy(update={"duration_s": elapsed})
        fed_round = FederatedRound(
            round_id,
            results,
            before,
            after,
            record,
            updates=dict(accepted_adapters),
            starts={cid: starts[cid] for cid in accepted_adapters},
        )
        self.rounds.append(fed_round)
        return fed_round

    def run(self, num_rounds: int, offline: Iterable[str] = ()) -> list[FederatedRound]:
        return [self.run_round(offline) for _ in range(num_rounds)]

    # ------------------------------------------------------------------

    def _deliver(self, decoded: LoRAAdapter) -> Callable[[str, object], None]:
        def deliver(client_id: str, payload: object) -> None:
            assert isinstance(payload, ClusterAdapterBroadcast)
            if self._send is not None:
                self._send(client_id, payload)  # may raise -> redistribute retries
            self.received[client_id] = (payload.cluster_id, payload.round_id)
            self.client_adapters[client_id] = decoded

        return deliver

    def isolation_violations(self) -> list[str]:
        """Every breach of the isolation rule found in the recorded history.

        * a contributor to cluster C's aggregate was not a member of C that round;
        * a client's last received broadcast is from a cluster it is not in now.
        """
        problems: list[str] = []
        for c in self._contributions:
            stray = sorted(set(c.contributors) - set(c.members))
            if stray:
                problems.append(
                    f"round {c.round_id}: cluster {c.cluster_id} aggregated non-members {stray}"
                )
        for client_id, (cluster_id, round_id) in sorted(self.received.items()):
            assigned = self.assigner.assignment.get(client_id)
            if assigned != cluster_id:
                problems.append(
                    f"client {client_id} holds cluster {cluster_id}'s adapter "
                    f"(round {round_id}) but is assigned to {assigned}"
                )
        return problems


def _copy_adapter(adapter: LoRAAdapter) -> LoRAAdapter:
    return LoRAAdapter(
        rank=adapter.rank,
        alpha=adapter.alpha,
        target_modules=adapter.target_modules,
        num_layers=adapter.num_layers,
        modules={
            layer: {m: {k: v.copy() for k, v in pair.items()} for m, pair in mods.items()}
            for layer, mods in adapter.modules.items()
        },
    )
