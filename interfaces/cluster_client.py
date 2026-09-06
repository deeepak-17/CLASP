"""P2 (Cluster Layer) boundary — the partition-plan handshake.

**P5 does not implement federated aggregation.** P2 (Prasanth) owns Flower,
FedProx and cluster-LoRA redistribution. The Week-1/2 coupling runs the other
way: P5's partition manifest determines *how many Flower clients exist and
what each one trains on*, so P2 needs to consume it.

This module defines the handshake:

* :class:`ClusterPlanConsumer` — the ``Protocol`` P2 implements to accept a
  :class:`~interfaces.contracts.PartitionManifest`.
* :class:`MockClusterClient` — records what it was handed and applies the same
  admission checks P2's real loop will (client count within simulator limits,
  every shard non-empty). This is what makes Week-2 Friday's cross-check
  executable rather than a conversation.

Week-2 note: P2's Week-2 target is "simulate 3 dummy clients", so the default
``expected_client_count`` is 3. Week 8 scales this to 5 per cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from interfaces.contracts import PartitionManifest
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)

#: P2's Week-2 simulated-client count (raised to 5 in Week 8).
DEFAULT_EXPECTED_CLIENTS = 3


@dataclass(frozen=True)
class PlanAcceptance:
    """Outcome of offering a partition manifest to the Cluster Layer."""

    accepted: bool
    client_ids: list[str] = field(default_factory=list)
    cluster_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "accepted" if self.accepted else "rejected"
        return (
            f"Cluster Layer {state} the plan: {len(self.client_ids)} client(s), "
            f"{len(self.cluster_ids)} cluster(s), {len(self.warnings)} warning(s), "
            f"{len(self.errors)} error(s)"
        )


@runtime_checkable
class ClusterPlanConsumer(Protocol):
    """Contract P2's Cluster Layer satisfies to receive P5's partition plan."""

    def register_partition_plan(self, manifest: PartitionManifest) -> PlanAcceptance:
        """Accept or reject a partition manifest as a federated round plan."""
        ...


class MockClusterClient:
    """Offline stand-in for P2's Flower server.

    Applies the admission rules P2's aggregation loop needs to hold, without
    running any aggregation:

    * at least two clients (federation of one is not a federation);
    * no empty shards (a Flower client with no data stalls the round);
    * client ids unique;
    * a *warning* — not an error — when the client count differs from what P2
      currently simulates, since scaling client counts is expected over time.
    """

    def __init__(
        self,
        expected_client_count: int = DEFAULT_EXPECTED_CLIENTS,
        max_client_count: int = 16,
    ) -> None:
        self.expected_client_count = expected_client_count
        self.max_client_count = max_client_count
        self.last_manifest: PartitionManifest | None = None

    def register_partition_plan(self, manifest: PartitionManifest) -> PlanAcceptance:
        errors: list[str] = []
        warnings: list[str] = []

        client_ids = [shard.client_id for shard in manifest.shards]

        if manifest.num_clients < 2:
            errors.append(
                f"Federated aggregation needs >= 2 clients; manifest declares {manifest.num_clients}"
            )
        if manifest.num_clients > self.max_client_count:
            errors.append(
                f"Manifest declares {manifest.num_clients} clients, simulator caps at {self.max_client_count}"
            )
        if len(set(client_ids)) != len(client_ids):
            errors.append("Duplicate client_id values in manifest")

        empty = [shard.client_id for shard in manifest.shards if shard.num_files == 0]
        if empty:
            errors.append(f"Empty shard(s) would stall a Flower round: {', '.join(empty)}")

        if manifest.num_clients != self.expected_client_count:
            warnings.append(
                f"Cluster Layer currently simulates {self.expected_client_count} client(s) "
                f"but the manifest declares {manifest.num_clients}; P2 must widen the simulation"
            )

        accepted = not errors
        if accepted:
            self.last_manifest = manifest

        _LOG.info(
            "MockClusterClient: %s plan (clients=%d, clusters=%d)",
            "accepted" if accepted else "rejected",
            manifest.num_clients,
            manifest.num_clusters,
        )
        return PlanAcceptance(
            accepted=accepted,
            client_ids=client_ids,
            cluster_ids=list(manifest.cluster_ids),
            warnings=warnings,
            errors=errors,
        )
