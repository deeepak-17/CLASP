"""P3 (Security) boundary — privacy-accounting placeholder.

**P5 does not implement DP-SGD or mTLS.** P3 (Kapilan) owns Opacus
integration, the moments accountant and transport security.

P5's only structural dependency is *reporting*: from Week 7 the epsilon spent
on a training run is displayed alongside its Pass@k number, so the
privacy-utility trade-off can be read off a single chart. Fixing the shape now
means the eval result schema does not have to change later.

Weeks 1-2 need nothing more than the protocol and a null implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from utils.errors import ClaspP5Error


@dataclass(frozen=True)
class PrivacyBudget:
    """A point-in-time reading of the DP-SGD privacy accountant.

    Attributes:
        epsilon: Cumulative privacy loss. ``None`` when DP is disabled — the
            no-DP baseline arm of the experiment.
        delta: Failure probability of the (epsilon, delta) guarantee.
        noise_multiplier: Gaussian noise multiplier used by DP-SGD.
        max_grad_norm: Per-sample gradient clipping threshold.
        steps: Number of optimiser steps accounted for.
        enabled: Whether differential privacy was active for the run.
    """

    epsilon: float | None = None
    delta: float | None = None
    noise_multiplier: float | None = None
    max_grad_norm: float | None = None
    steps: int = 0
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.enabled and self.epsilon is None:
            raise ClaspP5Error("PrivacyBudget marked enabled but carries no epsilon")
        if self.epsilon is not None and self.epsilon < 0:
            raise ClaspP5Error(f"epsilon must be non-negative, got {self.epsilon}")

    def label(self) -> str:
        """Short human-readable label for charts and reports."""
        if not self.enabled or self.epsilon is None:
            return "DP off (baseline)"
        return f"ε={self.epsilon:.2f}, δ={self.delta:g}" if self.delta else f"ε={self.epsilon:.2f}"


@runtime_checkable
class PrivacyAccountant(Protocol):
    """Contract P3's DP module satisfies for P5 to annotate eval results."""

    def current_budget(self) -> PrivacyBudget:
        """Return the privacy budget spent so far."""
        ...


class NullPrivacyAccountant:
    """No-DP accountant. The correct stand-in for the Week-1 baseline arm.

    Returns a disabled :class:`PrivacyBudget` rather than raising, so eval
    artefacts have a uniform shape whether or not DP is in play.
    """

    def current_budget(self) -> PrivacyBudget:
        return PrivacyBudget(enabled=False)
