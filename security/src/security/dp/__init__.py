from __future__ import annotations

from .accountant import BudgetExhaustedError, PrivacyAccountant
from .config import DPConfig
from .engine import get_privacy_engine, make_private

# Backward-compat alias: PR #17 (edge) used EpsilonTracker
EpsilonTracker = PrivacyAccountant

__all__ = [
    "DPConfig",
    "make_private",
    "PrivacyAccountant",
    "EpsilonTracker",
    "BudgetExhaustedError",
    "get_privacy_engine",
]
