"""CLASP shared interface contracts.

This package is the integration seam between modules (edge, cluster,
registry, evaluation, security). Changes here ripple across services, so
keep it backward-compatible where possible and version it deliberately.
"""
from .types import AdapterRef, EvalResult, AdapterKind

__version__ = "0.1.0"
__all__ = ["AdapterRef", "EvalResult", "AdapterKind", "__version__"]
