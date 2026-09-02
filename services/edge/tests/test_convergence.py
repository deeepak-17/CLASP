"""Convergence-summary tests (P1 Edge, W3 · E3.2).

`convergence` decides whether a training run counts as "loss went down", which
is half of E3.2's stability gate. It got these wrong once already — the first
sweep cell reported 0.669 -> 1.188 and was marked unstable while held-out
perplexity had actually improved, because the endpoints were single noisy
micro-batches rather than a trend. These tests pin the fixed behaviour.
"""
import pytest

from edge.train_client import convergence


def test_clean_descent_is_a_decrease():
    curve = [2.0 - 0.01 * i for i in range(100)]
    out = convergence(curve)
    assert out["loss_decreased"] is True
    assert out["loss_slope"] < 0
    assert out["last_window"] < out["first_window"]


def test_clean_ascent_is_not_a_decrease():
    curve = [1.0 + 0.01 * i for i in range(100)]
    out = convergence(curve)
    assert out["loss_decreased"] is False
    assert out["loss_slope"] > 0


def test_noisy_endpoints_do_not_flip_the_verdict():
    """The bug this function was rewritten for: a descending curve whose FIRST
    point is unusually low and LAST point unusually high still counts as a
    decrease, because windows and slope both see the trend."""
    curve = [2.0 - 0.01 * i for i in range(100)]
    curve[0] = 0.669      # freak easy block
    curve[-1] = 1.188     # freak hard block
    assert curve[-1] > curve[0]                 # endpoints say "went up"
    assert convergence(curve)["loss_decreased"] is True


def test_plunge_then_rebound_fails_on_slope():
    """Ends lower than it started, but spent the back half climbing. The window
    test alone would pass this; the slope is what rejects it."""
    curve = [3.0 - 0.2 * i for i in range(10)] + [1.0 + 0.05 * i for i in range(90)]
    out = convergence(curve)
    assert out["last_window"] > out["first_window"] or out["loss_slope"] > 0
    assert out["loss_decreased"] is False


def test_flat_curve_is_not_a_decrease():
    out = convergence([1.5] * 50)
    assert out["loss_decreased"] is False
    assert out["loss_slope"] == pytest.approx(0.0, abs=1e-9)


def test_window_is_twenty_percent_of_steps():
    out = convergence([1.0] * 100)
    assert out["window_steps"] == 20
    assert out["n_steps_recorded"] == 100


def test_window_never_collapses_to_zero_on_short_runs():
    out = convergence([2.0, 1.0])
    assert out["window_steps"] >= 1
    assert out["loss_decreased"] is True


def test_single_step_run_is_handled():
    out = convergence([1.0])
    assert out["n_steps_recorded"] == 1
    assert out["loss_decreased"] is False   # one point has no trend


def test_empty_curve_does_not_crash():
    out = convergence([])
    assert out["loss_decreased"] is False
    assert out["first_window"] is None
    assert out["n_steps_recorded"] == 0


def test_output_is_json_safe():
    import json
    json.dumps(convergence([2.0 - 0.01 * i for i in range(30)]))
