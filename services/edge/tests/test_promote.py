"""Edge's D5 prediction defers to the registry's own rule.

``edge.promote.predict_decision`` records what D5 implies next to the live
registry's answer. It must not carry a copy of the rule: it calls
``registry.promotion.decide`` (P4). These tests run when the registry package
is installed beside the edge and check the prediction against the rule's four
outcomes; without it, the prediction says it is unavailable instead of guessing.
"""
from __future__ import annotations

import builtins

import pytest

from edge.promote import GuardStatus, _previous_version, build_eval_result, predict_decision


def metrics(es: float) -> dict:
    return {"edit_similarity": es, "exact_match": 0.1, "perplexity": 2.5, "n_examples": 60}


def guard(candidate: float, baseline: float) -> GuardStatus:
    return GuardStatus(available=True, reason="test anchors",
                       candidate={"benchmark": "HumanEval", "pass_at_k": {"1": candidate}},
                       baseline={"benchmark": "HumanEval", "pass_at_k": {"1": baseline}})


NO_GUARD = GuardStatus(available=False, reason="no anchors — test")


def predict(cand: float, base: float, noise_band: float, guard_status: GuardStatus,
            *, version: int = 3, previous_version=2):
    """Predict on the same C2 body ``promote_candidate`` sends."""
    eval_result = build_eval_result(
        adapter_name="cluster-web", version=version, kind="cluster",
        in_project=metrics(cand), baseline_in_project=metrics(base), cluster_id="web",
        guard=[guard_status.candidate] if guard_status.available else [],
        baseline_noise_band=noise_band)
    baseline_guard = [guard_status.baseline] if guard_status.available else []
    return predict_decision(eval_result, baseline_guard, previous_version)


@pytest.fixture
def registry_rule():
    return pytest.importorskip("registry.promotion")


@pytest.mark.parametrize("cand,base,guard_status,expected", [
    (0.55, 0.50, guard(0.30, 0.31), "promote"),       # both halves hold
    (0.50, 0.55, guard(0.30, 0.31), "rollback"),      # in-project regressed
    (0.55, 0.50, guard(0.20, 0.31), "rollback"),      # pass@1 dropped > 2 pts
    (0.55, 0.50, NO_GUARD, "rollback"),               # guard missing => fails
])
def test_prediction_is_the_registrys_rule(registry_rule, cand, base, guard_status, expected):
    action, reason = predict(cand, base, 0.0, guard_status)
    assert action == expected
    assert reason.startswith("promoted:" if expected == "promote" else "rolled back:")


def test_noise_band_is_passed_through(registry_rule):
    action, _ = predict(0.51, 0.50, 0.05, guard(0.3, 0.3))
    assert action == "rollback"          # +0.01 does not clear a 0.05 band


def test_without_the_registry_package_nothing_is_made_up(monkeypatch):
    real_import = builtins.__import__

    def no_registry(name, *args, **kwargs):
        if name == "registry.promotion" or name.startswith("registry"):
            raise ImportError("registry not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_registry)
    action, why = predict(0.55, 0.50, 0.0, NO_GUARD)
    assert action == "unavailable"
    assert "registry" in why


def test_rollback_without_a_previous_version_is_not_made_up(registry_rule):
    action, why = predict(0.50, 0.55, 0.0, guard(0.3, 0.3), version=1, previous_version=None)
    assert action == "unavailable"
    assert "no previous version" in why


def test_promote_needs_no_previous_version(registry_rule):
    action, _ = predict(0.55, 0.50, 0.0, guard(0.30, 0.31), version=1, previous_version=None)
    assert action == "promote"


def test_previous_version_is_the_one_just_before():
    assert _previous_version([1, 2, 4, 5], 4) == 2
    assert _previous_version([1, 2, 3], 1) is None
