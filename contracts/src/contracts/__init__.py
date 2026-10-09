"""CLASP shared interface contracts.

This package is the integration seam between modules (edge, cluster,
registry, evaluation, security). Changes here ripple across services, so
keep it backward-compatible where possible and version it deliberately.

Frozen at v1.0.0; v1.1.0 adds composite adapters (D6) and lossless JSON
(``to_json`` / ``from_json``) on every value object, backward-compatibly.
Changes need a semver bump + all-hands sign-off.
"""
from .types import (
    CONTRACTS_VERSION,
    DEFAULT_BASE_MODEL,
    AdapterKind,
    AdapterMetadata,
    AdapterRef,
    AdapterUpload,
    AggregationMethod,
    ClusterSnapshot,
    CompositeProvenance,
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

__version__ = "1.1.0"
__all__ = [
    "CONTRACTS_VERSION",
    "DEFAULT_BASE_MODEL",
    "AdapterKind",
    "AdapterMetadata",
    "AdapterRef",
    "AdapterUpload",
    "AggregationMethod",
    "ClusterSnapshot",
    "CompositeProvenance",
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
