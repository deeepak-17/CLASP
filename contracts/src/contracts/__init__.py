"""CLASP shared interface contracts.

This package is the integration seam between modules (edge, cluster,
registry, evaluation, security). Changes here ripple across services, so
keep it backward-compatible where possible and version it deliberately.

**Frozen at v1.0.0 (Phase II W2).** Post-freeze changes need a semver bump +
all-hands sign-off (MASTER_PLAN D9).
"""
from .types import (
    CONTRACTS_VERSION,
    AdapterKind,
    AdapterMetadata,
    AdapterRef,
    AdapterUpload,
    AggregationMethod,
    ClusterSnapshot,
    EvalResult,
    GuardMetrics,
    InProjectMetrics,
    LoRAHyperParams,
    PrivacySpec,
    PromotionAction,
    PromotionDecision,
    RunManifest,
    utcnow_iso,
)

__version__ = "1.0.0"
__all__ = [
    "CONTRACTS_VERSION",
    "AdapterKind",
    "AdapterMetadata",
    "AdapterRef",
    "AdapterUpload",
    "AggregationMethod",
    "ClusterSnapshot",
    "EvalResult",
    "GuardMetrics",
    "InProjectMetrics",
    "LoRAHyperParams",
    "PrivacySpec",
    "PromotionAction",
    "PromotionDecision",
    "RunManifest",
    "utcnow_iso",
    "__version__",
]
