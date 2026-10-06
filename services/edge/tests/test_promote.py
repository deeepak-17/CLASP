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

from edge.promote import GuardStatus, predict_decision


def metrics(es: float) -> dict:
    return {"edit_similarity": es, "exact_match": 0.1, "perplexity": 2.5, "n_examples": 60}


def guard(candidate: float, baseline: float) -> GuardStatus:
    return GuardStatus(available=True, reason="test anchors",
                       candidate={"benchmark": "HumanEval", "pass_at_k": {"1": candidate}},
                       baseline={"benchmark": "HumanEval", "pass_at_k": {"1": baseline}})


NO_GUARD = GuardStatus(available=False, reason="no anchors — test")


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
    action, reason = predict_decision(metrics(cand), metrics(base), 0.0, guard_status)
    assert action == expected
    assert reason.startswith("promoted:" if expected == "promote" else "rolled back:")


def test_noise_band_is_passed_through(registry_rule):
    action, _ = predict_decision(metrics(0.51), metrics(0.50), 0.05, guard(0.3, 0.3))
    assert action == "rollback"          # +0.01 does not clear a 0.05 band


def test_without_the_registry_package_nothing_is_made_up(monkeypatch):
    real_import = builtins.__import__

    def no_registry(name, *args, **kwargs):
        if name == "registry.promotion" or name.startswith("registry"):
            raise ImportError("registry not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_registry)
    action, why = predict_decision(metrics(0.55), metrics(0.50), 0.0, NO_GUARD)
    assert action == "unavailable"
    assert "registry" in why
