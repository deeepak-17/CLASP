"""CLASP cluster service (P2): federated LoRA aggregation over Flower."""

from cluster.adapter_format import (
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    TARGET_MODULES,
    AdapterFormatError,
    LoRAAdapter,
    random_adapter,
)
from cluster.aggregation import (
    StreamingWeightedMean,
    aggregate_layerwise,
    aggregate_naive,
    aggregate_svd,
    aggregate_svd_lowrank,
    exact_average_delta,
    layerwise_exact_average_error,
    truncated_svd_refactor,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RANK",
    "TARGET_MODULES",
    "AdapterFormatError",
    "LoRAAdapter",
    "StreamingWeightedMean",
    "__version__",
    "aggregate_layerwise",
    "aggregate_naive",
    "aggregate_svd",
    "aggregate_svd_lowrank",
    "exact_average_delta",
    "layerwise_exact_average_error",
    "random_adapter",
    "truncated_svd_refactor",
]
