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

import json

from edge.promote import (
    GuardStatus,
    build_eval_result,
    predict_decision,
    resolve_guard,
)


def metrics(es: float) -> dict:
    return {"edit_similarity": es, "exact_match": 0.1, "perplexity": 2.5, "n_examples": 60}


def _lc(edit_similarity: float) -> dict:
    return {"benchmark": "LineCompletion", "edit_similarity": edit_similarity,
            "exact_match": 0.1, "n_examples": 300}


def guard(candidate: float, baseline: float) -> GuardStatus:
    return GuardStatus(available=True, reason="test guard",
                       candidate=_lc(candidate), baseline=_lc(baseline))


NO_GUARD = GuardStatus(available=False, reason="no anchors — test")


def predict(cand: float, base: float, noise_band: float, guard_status: GuardStatus,
            *, version: int = 3):
    """Predict on the same C2 body ``promote_candidate`` sends."""
    eval_result = build_eval_result(
        adapter_name="cluster-web", version=version, kind="cluster",
        in_project=metrics(cand), baseline_in_project=metrics(base), cluster_id="web",
        guard=[guard_status.candidate] if guard_status.available else [],
        baseline_noise_band=noise_band)
    baseline_guard = [guard_status.baseline] if guard_status.available else []
    return predict_decision(eval_result, baseline_guard)


@pytest.fixture
def registry_rule():
    return pytest.importorskip("registry.promotion")


@pytest.mark.parametrize("cand,base,guard_status,expected", [
    (0.55, 0.50, guard(0.30, 0.31), "promote"),       # both halves hold
    (0.50, 0.55, guard(0.30, 0.31), "reject"),        # in-project regressed
    (0.55, 0.50, guard(0.20, 0.31), "reject"),        # guard dropped > 2 pts
    (0.55, 0.50, NO_GUARD, "reject"),                 # guard missing => fails
])
def test_prediction_is_the_registrys_rule(registry_rule, cand, base, guard_status, expected):
    action, reason = predict(cand, base, 0.0, guard_status)
    assert action == expected
    assert reason.startswith("promoted:" if expected == "promote" else "rejected:")


def test_noise_band_is_passed_through(registry_rule):
    action, _ = predict(0.51, 0.50, 0.05, guard(0.3, 0.3))
    assert action == "reject"            # +0.01 does not clear a 0.05 band


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


def test_a_first_candidate_that_fails_is_rejected_not_unavailable(registry_rule):
    # Nothing is active before the first promote: a failing first candidate is
    # simply rejected, the rule has nothing to roll back to and needs nothing.
    action, why = predict(0.50, 0.55, 0.0, guard(0.3, 0.3), version=1)
    assert action == "reject" and why.startswith("rejected:")


def test_a_first_candidate_can_be_promoted(registry_rule):
    action, _ = predict(0.55, 0.50, 0.0, guard(0.30, 0.31), version=1)
    assert action == "promote"


def _evidence_file(tmp_path, cluster="web", **entry):
    doc = {"candidate_version": 2, "candidate_sha256": "b" * 64,
           "clusters": {cluster: {"guard": [_lc(0.60)], "baseline_guard": [_lc(0.61)], **entry}}}
    path = tmp_path / f"{cluster}-v2.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_resolve_guard_reads_the_line_guard_evidence_file(tmp_path):
    status = resolve_guard(_evidence_file(tmp_path))
    assert status.available and status.candidate == _lc(0.60) and status.baseline == _lc(0.61)
    assert status.candidate["benchmark"] == "LineCompletion"
    assert "line-completion" in status.reason
    assert status.as_dict()["benchmark"] == "LineCompletion"


def test_resolve_guard_without_a_file_says_so_and_invents_nothing(tmp_path):
    for status in (resolve_guard(), resolve_guard(tmp_path / "absent.json")):
        assert not status.available and status.candidate is None and status.baseline is None
        assert "line-completion guard" in status.reason and "HumanEval" not in status.reason


def test_resolve_guard_refuses_evidence_without_a_line_guard(tmp_path):
    path = _evidence_file(tmp_path, guard=[{"benchmark": "HumanEval", "pass_at_k": {"1": 0.3}}])
    status = resolve_guard(path)
    assert not status.available and "LineCompletion" in status.reason
    assert not resolve_guard(_evidence_file(tmp_path, cluster="web", guard=[],
                                            baseline_guard=[])).available


def test_resolve_guard_with_several_clusters_needs_the_cluster_named(tmp_path):
    path = _evidence_file(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["clusters"]["scientific"] = doc["clusters"]["web"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert not resolve_guard(path).available
    assert resolve_guard(path, cluster_id="scientific").available
