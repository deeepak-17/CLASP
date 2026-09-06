"""Federated data partitioning — the Week-2 deliverable.

Turns the collected D1 corpus into the federated client topology the rest of
CLASP trains against:

* :mod:`partitions.strategies` — project-level (non-IID by project) assignment.
* :mod:`partitions.partitioner` — shard writing and contract-manifest construction.
* :mod:`partitions.validation` — non-overlap, coverage, purity and balance checks.
* :mod:`partitions.metadata` — descriptive statistics written alongside the manifest.

Shards are written to ``datasets/partitions/``; the contract artefact other
modules read is ``datasets/partitions/manifest.json``.
"""

from partitions.metadata import (
    PartitionMetadata,
    build_partition_metadata,
    render_metadata_report,
    write_partition_metadata,
)
from partitions.models import PartitionConfig, ValidationConfig
from partitions.partitioner import (
    PartitionResult,
    Partitioner,
    load_partition_config,
    load_partition_manifest,
    load_shard_records,
)
from partitions.strategies import (
    ClientAssignment,
    PartitionStrategy,
    ProjectLevelStrategy,
    build_strategy,
)
from partitions.validation import (
    BalanceMetrics,
    PartitionValidationResult,
    PartitionValidator,
    compute_balance,
    render_validation_report,
)

__all__ = [
    "BalanceMetrics",
    "ClientAssignment",
    "PartitionConfig",
    "PartitionMetadata",
    "PartitionResult",
    "PartitionStrategy",
    "PartitionValidationResult",
    "PartitionValidator",
    "Partitioner",
    "ProjectLevelStrategy",
    "ValidationConfig",
    "build_partition_metadata",
    "build_strategy",
    "compute_balance",
    "load_partition_config",
    "load_partition_manifest",
    "load_shard_records",
    "render_metadata_report",
    "render_validation_report",
    "write_partition_metadata",
]
