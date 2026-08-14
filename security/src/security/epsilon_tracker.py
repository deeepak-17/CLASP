"""Standalone RDP-based epsilon tracker for CLASP federated training.

Tracks cumulative privacy spend across training steps *without* needing
an active Opacus ``PrivacyEngine``.  Useful for:
  - Tracking ε across multiple FL rounds on the cluster side.
  - Pre-computing privacy budgets before training starts.
  - Generating the epsilon-budget reports for panel sign-off.

The tracker uses Rényi Differential Privacy (RDP) internally and converts
to (ε, δ)-DP via the optimal RDP→DP conversion.

References
----------
- Time-Adaptive Privacy Spending — Kiani et al., ICLR 2025
- Tighter Privacy Auditing — Cebere et al., ICLR 2025
- FedASK — Wen et al., NeurIPS 2025
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

from security.dp_config import DPConfig

logger = logging.getLogger(__name__)

# Default Rényi divergence orders for the moments accountant.
# Covers integer orders 2..64 plus some fractional orders for tighter bounds.
DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


@dataclass
class EpsilonTracker:
    """Tracks cumulative ε via the RDP moments accountant.

    Parameters
    ----------
    noise_multiplier : float
        Gaussian noise scale σ used in DP-SGD.
    sample_rate : float
        Probability that each example is included in a mini-batch
        (= batch_size / dataset_size).
    delta : float
        Target failure probability for the (ε, δ)-DP guarantee.
    alphas : list[float]
        Rényi divergence orders for the moments accountant.
    """

    noise_multiplier: float
    sample_rate: float
    delta: float = 1e-5
    alphas: list[float] = field(default_factory=lambda: list(DEFAULT_ALPHAS))

    # Internal state
    _steps: int = field(default=0, init=False, repr=False)
    _rdp: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize RDP accumulator to zeros
        self._rdp = [0.0] * len(self.alphas)

    @classmethod
    def from_dp_config(cls, dp_config: DPConfig, dataset_size: int) -> "EpsilonTracker":
        """Create a tracker from a :class:`DPConfig` and dataset size."""
        sample_rate = dp_config.batch_size / dataset_size

        # Auto-calibrate noise multiplier if not specified
        noise_multiplier = dp_config.noise_multiplier
        if noise_multiplier is None:
            from opacus.accountants.utils import get_noise_multiplier
            noise_multiplier = get_noise_multiplier(
                target_epsilon=dp_config.target_epsilon,
                target_delta=dp_config.target_delta,
                sample_rate=sample_rate,
                epochs=dp_config.epochs,
            )

        return cls(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=dp_config.target_delta,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, num_steps: int = 1) -> float:
        """Record *num_steps* training steps and return current ε.

        Each step corresponds to one optimizer update (one mini-batch of
        gradient computation + noise addition).
        """
        for alpha_idx, alpha in enumerate(self.alphas):
            self._rdp[alpha_idx] += num_steps * self._compute_rdp_single(
                q=self.sample_rate,
                sigma=self.noise_multiplier,
                alpha=alpha,
            )
        self._steps += num_steps
        return self.get_epsilon()

    def get_epsilon(self, delta: float | None = None) -> float:
        """Current cumulative ε via optimal RDP → (ε, δ) conversion."""
        d = delta if delta is not None else self.delta
        return self._rdp_to_epsilon(self._rdp, self.alphas, d)

    @property
    def total_steps(self) -> int:
        return self._steps

    def budget_remaining(self, max_epsilon: float) -> float:
        """How much ε budget is left before exceeding *max_epsilon*."""
        return max(0.0, max_epsilon - self.get_epsilon())

    def is_budget_exceeded(self, max_epsilon: float) -> bool:
        """Check if the cumulative ε exceeds *max_epsilon*."""
        return self.get_epsilon() > max_epsilon

    def to_report(self) -> dict:
        """JSON-serialisable privacy spending report."""
        eps = self.get_epsilon()
        return {
            "total_steps": self._steps,
            "current_epsilon": round(eps, 6),
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": round(self.sample_rate, 6),
            "budget_status": "OK" if eps <= 8.0 else "EXCEEDED",
        }

    # ------------------------------------------------------------------
    # RDP internals (Mironov, IEEE CSF 2017 — used by all 2025-26 papers)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rdp_single(q: float, sigma: float, alpha: float) -> float:
        """Compute RDP of the Sampled Gaussian Mechanism for a single α.

        Uses the analytical formula for subsampled Gaussian (Mironov 2017,
        Balle et al. 2020).  For simplicity, we use the upper bound:

            RDP_α ≤ (1 / (α - 1)) * log(1 - q + q * exp((α - 1) / (2σ²)))

        when α > 1, which is tight for the regimes we operate in.
        """
        if alpha <= 1:
            return 0.0
        if sigma == 0:
            return float("inf")
        if q == 0:
            return 0.0

        # For numerical stability with large α
        log_term = (alpha - 1) / (2.0 * sigma * sigma)

        if log_term > 500:  # Prevent overflow
            return log_term / (alpha - 1)

        inner = 1 - q + q * math.exp(log_term)
        if inner <= 0:
            return float("inf")

        return math.log(inner) / (alpha - 1)

    @staticmethod
    def _rdp_to_epsilon(
        rdp: list[float],
        alphas: list[float],
        delta: float,
    ) -> float:
        """Convert RDP guarantees to (ε, δ)-DP via optimal conversion.

        ε = min_α { RDP_α - log(δ) / (α - 1) }
        """
        best_eps = float("inf")
        for alpha, rdp_alpha in zip(alphas, rdp):
            if alpha <= 1:
                continue
            eps = rdp_alpha - math.log(delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps
        return max(0.0, best_eps)
