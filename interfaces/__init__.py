"""Integration seam between CLASP-P5 and the rest of Team 102's modules.

Everything in this package is either

* a **contract** — a data shape crossing a module boundary
  (:mod:`interfaces.contracts`), or
* a **protocol + mock** — the call signature another member's module will
  satisfy, paired with an offline stand-in so P5 can be developed and tested
  before that module exists.

No production logic belonging to P1–P4 lives here, and none should be added.
When a teammate ships the real implementation, the mock is swapped out at the
composition root (``scripts/*.py``); no P5 business logic changes.

Boundary map
------------
=========================  =====  ==============================================
Module                     Owner  P5 relationship
=========================  =====  ==============================================
:mod:`.edge_client`        P1     P5 calls it (Week 3+) to generate completions.
:mod:`.cluster_client`     P2     P5 hands it the partition plan (Week 2).
:mod:`.security_client`    P3     P5 reads epsilon to annotate results (Week 7+).
:mod:`.registry_client`    P4     P5 reads snapshots to know what to evaluate.
=========================  =====  ==============================================
"""

from interfaces.cluster_client import ClusterPlanConsumer, MockClusterClient, PlanAcceptance
from interfaces.contracts import (
    CONTRACT_VERSION,
    AdapterKind,
    AdapterRef,
    BenchmarkName,
    EvalResult,
    PartitionManifest,
    PartitionShard,
    PartitionStrategyName,
    SnapshotMetadata,
)
from interfaces.edge_client import (
    EdgeInferenceClient,
    GenerationRequest,
    GenerationResult,
    MockEdgeInferenceClient,
)
from interfaces.registry_client import (
    MockRegistryClient,
    RegistryReadClient,
    SnapshotNotFoundError,
)
from interfaces.security_client import NullPrivacyAccountant, PrivacyAccountant, PrivacyBudget

__all__ = [
    "CONTRACT_VERSION",
    "AdapterKind",
    "AdapterRef",
    "BenchmarkName",
    "ClusterPlanConsumer",
    "EdgeInferenceClient",
    "EvalResult",
    "GenerationRequest",
    "GenerationResult",
    "MockClusterClient",
    "MockEdgeInferenceClient",
    "MockRegistryClient",
    "NullPrivacyAccountant",
    "PartitionManifest",
    "PartitionShard",
    "PartitionStrategyName",
    "PlanAcceptance",
    "PrivacyAccountant",
    "PrivacyBudget",
    "RegistryReadClient",
    "SnapshotMetadata",
    "SnapshotNotFoundError",
]
