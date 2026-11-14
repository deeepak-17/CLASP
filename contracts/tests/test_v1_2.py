"""v1.2.0: line-completion guard, cumulative epsilon, reject action (additive)."""
import contracts
import pytest
from contracts import (
    AdapterKind,
    AdapterRef,
    GuardMetrics,
    PrivacySpec,
    PromotionAction,
    PromotionDecision,
)


def test_version_is_1_2_0():
    assert contracts.CONTRACTS_VERSION == contracts.__version__ == "1.2.0"


def test_line_completion_guard_round_trips():
    g = GuardMetrics(benchmark="LineCompletion", edit_similarity=0.61, exact_match=0.22, n_examples=300)
    assert GuardMetrics.from_json(g.to_json()) == g and g.pass_at_k == {}


def test_guard_from_1_1_payload_still_parses():
    g = GuardMetrics.from_json({"benchmark": "HumanEval", "pass_at_k": {"1": 0.3}})
    assert g.pass_at_k == {1: 0.3} and g.edit_similarity is None


@pytest.mark.parametrize("field,value", [("edit_similarity", 1.2), ("exact_match", -0.1), ("n_examples", 0)])
def test_guard_rejects_out_of_range(field, value):
    with pytest.raises(ValueError):
        GuardMetrics(benchmark="LineCompletion", **{field: value})


def test_privacy_cumulative_round_trips_and_is_checked():
    p = PrivacySpec(epsilon=3.9, epsilon_cumulative=7.8, rounds_accounted=2)
    assert PrivacySpec.from_json(p.to_json()) == p
    assert PrivacySpec.from_json({"epsilon": 3.9, "delta": 1e-5}).epsilon_cumulative is None
    with pytest.raises(ValueError):
        PrivacySpec(epsilon=5.0, epsilon_cumulative=4.0)


def test_privacy_rounds_accounted_must_be_positive():
    with pytest.raises(ValueError):
        PrivacySpec(rounds_accounted=0)


def test_reject_action_round_trips():
    d = PromotionDecision(adapter=AdapterRef("cluster-web", 3, AdapterKind.CLUSTER),
                          action=PromotionAction.REJECT, active_version_after=2, reason="r")
    assert PromotionDecision.from_json(d.to_json()).action is PromotionAction.REJECT
