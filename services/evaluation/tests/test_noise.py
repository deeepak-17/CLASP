"""Tests for eval_harness/noise.py — evaluation noise and noise-aware guard verdicts."""

from __future__ import annotations

import math

import pytest

from evaluation.eval_harness.noise import (
    Z_95,
    binomial_standard_error,
    bootstrap_mean_ci,
    classify_guard_drop,
    guard_noise_report,
    paired_bootstrap_diff_ci,
    paired_min_detectable_drop,
    required_items_paired,
    unpaired_min_detectable_drop,
)
from evaluation.utils.errors import EvaluationError


def test_binomial_standard_error_closed_form() -> None:
    assert binomial_standard_error(0.5, 100) == pytest.approx(0.05)
    with pytest.raises(EvaluationError):
        binomial_standard_error(1.5, 10)


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0] * 10 + [0.0] * 10
    lo, hi = bootstrap_mean_ci(values, n_resamples=2000, seed=7)
    assert (lo, hi) == bootstrap_mean_ci(values, n_resamples=2000, seed=7)
    assert lo < 0.5 < hi


def test_constant_values_have_a_zero_width_interval() -> None:
    assert bootstrap_mean_ci([0.3] * 12, n_resamples=500) == (pytest.approx(0.3), pytest.approx(0.3))


def test_paired_diff_needs_matching_items() -> None:
    with pytest.raises(EvaluationError):
        paired_bootstrap_diff_ci([1.0, 0.0], [1.0])
    lo, hi = paired_bootstrap_diff_ci([0.5] * 30, [0.6] * 30, n_resamples=500)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_detectable_drop_formulas() -> None:
    assert paired_min_detectable_drop(0.1, 20) == pytest.approx(Z_95 * math.sqrt(0.1 / 20))
    assert unpaired_min_detectable_drop(0.5, 20) == pytest.approx(Z_95 * math.sqrt(2 * 0.25 / 20))
    # Inverting the paired floor recovers the item count.
    n = required_items_paired(0.02, 0.1)
    assert paired_min_detectable_drop(0.1, n) <= 0.02 < paired_min_detectable_drop(0.1, n - 1)


@pytest.mark.parametrize(
    ("drop", "expected"),
    [(0.0, "pass"), (0.02, "pass"), (0.05, "within_noise"), (0.2, "regression")],
)
def test_classify_guard_drop(drop: float, expected: str) -> None:
    assert classify_guard_drop(drop, tolerance=0.02, noise_floor=0.1) == expected


def test_guard_report_on_twenty_tasks_half_solved() -> None:
    report = guard_noise_report([True] * 10 + [False] * 10, tolerance=0.02, full_benchmark_size=164)
    assert report["pass_at_1"] == 0.5
    assert report["standard_error"] == pytest.approx(0.1118, abs=1e-4)
    # The finding the noise report states: 2 points is below the floor even at 164 tasks.
    assert all(row["min_detectable_drop_full_benchmark"] > 0.02 for row in report["paired"])
