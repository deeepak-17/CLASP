"""Lightweight wrapper around Opacus's built-in privacy accountants.

Provides CLASP-specific helpers for querying the current ε, checking
whether the privacy budget is exhausted, and converting between RDP
and (ε, δ)-DP.

References
----------
- Tighter Privacy Auditing of DP-SGD — Cebere et al., ICLR 2025
- Time-Adaptive Privacy Spending — Kiani et al., ICLR 2025
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_current_epsilon(
    privacy_engine,
    delta: float | None = None,
) -> float:
    """Return the cumulative ε spent so far.

    Parameters
    ----------
    privacy_engine : opacus.PrivacyEngine
        A privacy engine that has been used for at least one training step.
    delta : float, optional
        Failure probability.  If ``None``, uses the delta configured at
        engine creation time (typically 1e-5).

    Returns
    -------
    float
        Current (ε, δ)-DP guarantee.
    """
    kwargs = {}
    if delta is not None:
        kwargs["delta"] = delta

    eps = privacy_engine.get_epsilon(**kwargs)
    return eps


def check_budget_exceeded(
    privacy_engine,
    max_epsilon: float,
    delta: float | None = None,
) -> bool:
    """Check whether the cumulative privacy spend exceeds *max_epsilon*.

    Returns ``True`` if the budget is blown — training should stop.
    """
    current = get_current_epsilon(privacy_engine, delta=delta)
    exceeded = current > max_epsilon
    if exceeded:
        logger.warning(
            "Privacy budget EXCEEDED: ε=%.4f > max_ε=%.4f.  "
            "Stop training to preserve differential privacy guarantee.",
            current,
            max_epsilon,
        )
    return exceeded
