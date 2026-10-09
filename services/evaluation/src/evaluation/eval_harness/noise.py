"""Evaluation noise — how large a change has to be before it is a change.

D5 promotes a candidate when its in-project gain clears a *noise band* and
its HumanEval pass@1 does not drop by more than a tolerance. Both numbers
only mean something relative to the measurement noise of the metric they
gate, so this module quantifies that noise from real per-item outcomes:

* :func:`bootstrap_mean_ci` — percentile bootstrap interval of a mean over
  per-item scores (per-task pass/fail, per-example edit similarity);
* :func:`binomial_standard_error` — the closed form for a proportion;
* :func:`paired_min_detectable_drop` / :func:`unpaired_min_detectable_drop`
  — the smallest change a two-sided test at ``z`` can tell from zero when
  candidate and baseline are scored on the same items (paired; noise comes
  only from items whose outcome flips) or on independent items;
* :func:`classify_guard_drop` — the noise-aware reading of a guard result:
  a drop beyond the tolerance *and* beyond the noise floor is a regression,
  one beyond the tolerance but inside the noise floor is ``within_noise``
  (not evidence either way), anything else passes.

Every function is deterministic (the bootstrap takes an explicit seed).
"""

from __future__ import annotations

import math
import random
from statistics import fmean
from typing import Any, Sequence

from evaluation.utils.errors import EvaluationError

#: Two-sided 95 % normal quantile.
Z_95 = 1.959963984540054


def binomial_standard_error(p: float, n: int) -> float:
    """Standard error of a proportion ``p`` estimated from ``n`` items."""
    if n <= 0:
        raise EvaluationError("n must be positive")
    if not 0.0 <= p <= 1.0:
        raise EvaluationError(f"p must be in [0, 1], got {p}")
    return math.sqrt(p * (1.0 - p) / n)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``mean(values)``."""
    if not values:
        raise EvaluationError("bootstrap needs at least one value")
    if not 0.0 < confidence < 1.0:
        raise EvaluationError("confidence must be in (0, 1)")
    # Resample the values with replacement many times; the spread of the resampled means is the interval.
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    tail = (1.0 - confidence) / 2.0
    lo = means[int(math.floor(tail * (n_resamples - 1)))]
    hi = means[int(math.ceil((1.0 - tail) * (n_resamples - 1)))]
    return lo, hi


def paired_bootstrap_diff_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval of ``mean(candidate - baseline)`` over the same items.

    This is the in-project noise band D5 needs: score both versions on the
    same held-out examples (``evaluation.completion.per_example_rows``), and a
    gain whose interval excludes zero is beyond noise.
    """
    if len(baseline) != len(candidate):
        raise EvaluationError("paired bootstrap needs the same items on both sides")
    return bootstrap_mean_ci(
        [c - b for b, c in zip(baseline, candidate)],
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def unpaired_min_detectable_drop(p: float, n: int, *, z: float = Z_95) -> float:
    """Smallest difference of two proportions near ``p``, each over ``n`` independent items."""
    return z * math.sqrt(2.0) * binomial_standard_error(p, n)


def paired_min_detectable_drop(discordance: float, n: int, *, z: float = Z_95) -> float:
    """Smallest pass-rate difference detectable on the same ``n`` items.

    ``discordance`` is the fraction of items whose outcome differs between the
    two models (McNemar). Under no true difference, the paired difference has
    standard error ``sqrt(discordance / n)``.
    """
    if n <= 0:
        raise EvaluationError("n must be positive")
    if not 0.0 <= discordance <= 1.0:
        raise EvaluationError(f"discordance must be in [0, 1], got {discordance}")
    # Paired test: only items whose pass/fail flips between the two models add noise.
    return z * math.sqrt(discordance / n)


def required_items_paired(drop: float, discordance: float, *, z: float = Z_95) -> int:
    """Items needed before a paired comparison can resolve a ``drop``."""
    if drop <= 0:
        raise EvaluationError("drop must be positive")
    return math.ceil(z * z * discordance / (drop * drop))


def classify_guard_drop(drop: float, *, tolerance: float, noise_floor: float) -> str:
    """``regression`` | ``within_noise`` | ``pass`` for a baseline-minus-candidate drop."""
    # Within D5's tolerance -> pass; beyond it but inside the noise floor -> not enough evidence either way.
    if drop <= tolerance:
        return "pass"
    if drop <= noise_floor:
        return "within_noise"
    return "regression"


def guard_noise_report(
    per_task_passed: Sequence[bool],
    *,
    tolerance: float,
    full_benchmark_size: int,
    discordance_grid: Sequence[float] = (0.05, 0.10, 0.20),
    seed: int = 0,
) -> dict[str, Any]:
    """Noise floor of a HumanEval-style guard, from one scored anchor's per-task outcomes."""
    n = len(per_task_passed)
    if n == 0:
        raise EvaluationError("guard noise report needs at least one scored task")
    scores = [1.0 if passed else 0.0 for passed in per_task_passed]
    p = fmean(scores)
    lo, hi = bootstrap_mean_ci(scores, seed=seed)
    rows = []
    # Discordance is unknown until a candidate is scored, so report the floor for a few plausible values.
    for d in discordance_grid:
        rows.append(
            {
                "discordance": d,
                "min_detectable_drop_at_n": round(paired_min_detectable_drop(d, n), 4),
                "min_detectable_drop_full_benchmark": round(paired_min_detectable_drop(d, full_benchmark_size), 4),
                "items_needed_for_tolerance": required_items_paired(tolerance, d),
            }
        )
    return {
        "n_tasks": n,
        "pass_at_1": round(p, 4),
        "standard_error": round(binomial_standard_error(p, n), 4),
        "bootstrap_ci_95": [round(lo, 4), round(hi, 4)],
        "tolerance": tolerance,
        "full_benchmark_size": full_benchmark_size,
        "unpaired_min_detectable_drop_at_n": round(unpaired_min_detectable_drop(p, n), 4),
        "paired": rows,
    }
