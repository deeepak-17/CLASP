"""P2-side seam to the registry (P4): publish each aggregated cluster adapter.

Cluster does not own the registry (``services/registry`` is P4's service), so this
module defines only what Cluster *needs* from it for the automatic, per-aggregate
publish hook and ships an in-memory stand-in so that path is testable. (Publishing
on request goes through ``POST /adapters/{id}/publish`` in ``cluster.server``, which
does talk to the registry's HTTP API.)

  * ``SnapshotSink``           what Cluster calls after a cluster aggregates.
  * ``cluster_adapter_ref``    Cluster's mapping onto ``contracts.AdapterRef``
                               (name = cluster id, version = round_id + 1,
                               kind = CLUSTER). This naming is **Cluster's
                               proposal**; P4 has not defined one yet.
  * ``InMemorySnapshotSink``   a test double. It is NOT a registry.

Nothing here talks to a network. When P4 publishes a real client/endpoint the
only code to write is a ``SnapshotSink`` that adapts it (see
docs/INTEGRATION_BOUNDARIES.md).
"""

from __future__ import annotations

from typing import Protocol

from contracts.types import AdapterKind, AdapterRef

from cluster.schemas.messages import ClusterAdapterBroadcast


class SnapshotSink(Protocol):
    """Receives one aggregated cluster adapter (PEFT tensors + adapter_config)."""

    def publish(self, ref: AdapterRef, broadcast: ClusterAdapterBroadcast) -> None: ...


def cluster_adapter_ref(cluster_id: str, round_id: int) -> AdapterRef:
    """Round ``r`` (0-based) is registry version ``r + 1`` of that cluster's adapter."""
    return AdapterRef(
        name=cluster_id,
        version=round_id + 1,
        kind=AdapterKind.CLUSTER,
        cluster_id=cluster_id,
    )


class InMemorySnapshotSink:
    """Test double for the registry: remembers every publish, in order."""

    def __init__(self) -> None:
        self.snapshots: list[tuple[AdapterRef, ClusterAdapterBroadcast]] = []

    def publish(self, ref: AdapterRef, broadcast: ClusterAdapterBroadcast) -> None:
        self.snapshots.append((ref, broadcast))

    def latest(self, cluster_id: str) -> tuple[AdapterRef, ClusterAdapterBroadcast] | None:
        for ref, broadcast in reversed(self.snapshots):
            if ref.cluster_id == cluster_id:
                return ref, broadcast
        return None
