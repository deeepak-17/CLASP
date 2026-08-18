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
    aggregate_naive,
    aggregate_svd,
    exact_average_delta,
    truncated_svd_refactor,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_ALPHA",
    "DEFAULT_RANK",
    "TARGET_MODULES",
    "AdapterFormatError",
    "LoRAAdapter",
    "random_adapter",
    "StreamingWeightedMean",
    "aggregate_naive",
    "aggregate_svd",
    "exact_average_delta",
    "truncated_svd_refactor",
]
