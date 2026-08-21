"""Unit tests for the D5 two-sided promotion rule (P4, W5 catch-up)."""
from __future__ import annotations

import pytest
from contracts import (
    AdapterKind,
    AdapterRef,
    EvalResult,
    GuardMetrics,
    InProjectMetrics,
    PromotionAction,
)

from registry.promotion import decide


def _ref(version=2):
    return AdapterRef(name="django", version=version, kind=AdapterKind.CLIENT)


def _in_project(edit_similarity):
    return InProjectMetrics(
        edit_similarity=edit_similarity, exact_match=0.5, perplexity=3.0, n_examples=20
    )


def _guard(pass_at_1):
    return GuardMetrics(benchmark="HumanEval", pass_at_k={1: pass_at_1})


def test_promotes_when_improved_beyond_noise_and_guard_holds():
    eval_result = EvalResult(
        adapter=_ref(),
        in_project=_in_project(0.80),
        guard=(_guard(0.30),),
        baseline_in_project=_in_project(0.70),
        baseline_noise_band=0.02,
    )
    decision = decide(
        eval_result,
        baseline_guard=(_guard(0.31),),
        active_version_before=2,
        previous_version=1,
    )
    assert decision.action == PromotionAction.PROMOTE
    assert decision.active_version_after == 2


def test_rolls_back_when_improvement_does_not_clear_noise_band():
    eval_result = EvalResult(
        adapter=_ref(),
        in_project=_in_project(0.705),  # +0.005, noise band is 0.02
        guard=(_guard(0.30),),
        baseline_in_project=_in_project(0.70),
        baseline_noise_band=0.02,
    )
    decision = decide(
        eval_result,
        baseline_guard=(_guard(0.30),),
        active_version_before=2,
        previous_version=1,
    )
    assert decision.action == PromotionAction.ROLLBACK
    assert decision.active_version_after == 1
    assert "noise band" in decision.reason


def test_rolls_back_when_guard_regresses_beyond_tolerance():
    eval_result = EvalResult(
        adapter=_ref(),
        in_project=_in_project(0.90),  # clearly improved
        guard=(_guard(0.20),),  # baseline 0.31 -> drop of 0.11, way over 0.02 tolerance
        baseline_in_project=_in_project(0.70),
        baseline_noise_band=0.02,
    )
    decision = decide(
        eval_result,
        baseline_guard=(_guard(0.31),),
        active_version_before=2,
        previous_version=1,
    )
    assert decision.action == PromotionAction.ROLLBACK
    assert "pass@1" in decision.reason


def test_rolls_back_when_no_baseline_in_project_supplied():
    eval_result = EvalResult(
        adapter=_ref(),
        in_project=_in_project(0.90),
        guard=(_guard(0.30),),
        baseline_in_project=None,
        baseline_noise_band=0.02,
    )
    decision = decide(
        eval_result,
        baseline_guard=(_guard(0.30),),
        active_version_before=2,
        previous_version=1,
    )
    assert decision.action == PromotionAction.ROLLBACK
    assert "no baseline_in_project" in decision.reason


def test_rollback_without_a_previous_version_raises():
    """Never gate on pass@k alone — but also never silently roll back to nothing."""
    eval_result = EvalResult(
        adapter=_ref(version=1),
        in_project=_in_project(0.50),  # regressed
        guard=(_guard(0.30),),
        baseline_in_project=_in_project(0.70),
        baseline_noise_band=0.02,
    )
    with pytest.raises(ValueError, match="nothing to roll back to"):
        decide(
            eval_result,
            baseline_guard=(_guard(0.30),),
            active_version_before=1,
            previous_version=None,
        )


def test_missing_guard_metrics_forces_rollback():
    """Two-sided means both sides must actually be checkable — an absent guard
    is not treated as a pass (D5: never gate on pass@k alone, but also never
    skip the guard check silently)."""
    eval_result = EvalResult(
        adapter=_ref(),
        in_project=_in_project(0.90),
        guard=(),  # no HumanEval guard reported at all
        baseline_in_project=_in_project(0.70),
        baseline_noise_band=0.02,
    )
    decision = decide(
        eval_result,
        baseline_guard=(_guard(0.30),),
        active_version_before=2,
        previous_version=1,
    )
    assert decision.action == PromotionAction.ROLLBACK
    assert "guard metrics missing" in decision.reason
