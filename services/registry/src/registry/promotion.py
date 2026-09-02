"""Two-sided promotion rule (D5) — P4 owns the rule, P5 owns the metrics that feed it.

    Promote iff:
      1. in-project quality improved beyond the measured noise band, AND
      2. HumanEval pass@1 did not regress by more than 2 points absolute.
    Otherwise: rollback to the previous version. Never gate on pass@k alone.

The registry always activates a version on save (see storage.save); this rule
runs *after* the fact, on whatever is currently active, and either confirms
that choice (PROMOTE) or reverts it (ROLLBACK to the previous version). It is
a pure function — no filesystem access — so it's cheap to unit test the D5
math in isolation from the storage/HTTP layers.
"""
from __future__ import annotations

from contracts import EvalResult, GuardMetrics, PromotionAction, PromotionDecision

#: Absolute pass@1 regression tolerated before it blocks a promotion (D5),
#: expressed as a fraction since GuardMetrics.pass_at_k is 0..1 (e.g. 0.31 = 31%).
GUARD_PASS_AT_1_TOLERANCE = 0.02


def _in_project_improved(eval_result: EvalResult) -> tuple[bool, str]:
    baseline = eval_result.baseline_in_project
    if baseline is None:
        return False, "no baseline_in_project supplied — cannot confirm improvement"
    delta = eval_result.in_project.edit_similarity - baseline.edit_similarity
    band = eval_result.baseline_noise_band
    if delta > band:
        return True, f"edit_similarity +{delta:.4f} clears noise band {band:.4f}"
    return False, f"edit_similarity delta {delta:.4f} does not clear noise band {band:.4f}"


def _guard_holds(
    eval_result: EvalResult, baseline_guard: tuple[GuardMetrics, ...]
) -> tuple[bool, str]:
    candidate = next((g for g in eval_result.guard if g.benchmark == "HumanEval"), None)
    baseline = next((g for g in baseline_guard if g.benchmark == "HumanEval"), None)
    if candidate is None or baseline is None:
        return False, "HumanEval guard metrics missing on candidate or baseline"
    if 1 not in candidate.pass_at_k or 1 not in baseline.pass_at_k:
        return False, "pass@1 missing from HumanEval guard metrics"
    drop = baseline.pass_at_k[1] - candidate.pass_at_k[1]
    if drop <= GUARD_PASS_AT_1_TOLERANCE:
        return True, f"HumanEval pass@1 drop {drop:.4f} within {GUARD_PASS_AT_1_TOLERANCE:.2f} tolerance"
    return False, f"HumanEval pass@1 dropped {drop:.4f}, exceeds {GUARD_PASS_AT_1_TOLERANCE:.2f} tolerance"


def decide(
    eval_result: EvalResult,
    *,
    baseline_guard: tuple[GuardMetrics, ...],
    active_version_before: int,
    previous_version: int | None,
) -> PromotionDecision:
    """Apply the D5 two-sided rule to a candidate that is currently `active`.

    Raises ``ValueError`` if the rule says rollback but there is no previous
    version to roll back to — that's a caller error (rollback candidates must
    have a version to fall back to), surfaced as a clean 422 at the API layer.
    """
    improved, improved_reason = _in_project_improved(eval_result)
    guard_ok, guard_reason = _guard_holds(eval_result, baseline_guard)

    if improved and guard_ok:
        return PromotionDecision(
            adapter=eval_result.adapter,
            action=PromotionAction.PROMOTE,
            active_version_after=active_version_before,
            reason=f"promoted: {improved_reason}; {guard_reason}",
        )

    if previous_version is None:
        raise ValueError(
            f"rollback indicated ({improved_reason}; {guard_reason}) but no previous "
            f"version exists for {eval_result.adapter.name!r} — nothing to roll back to"
        )
    return PromotionDecision(
        adapter=eval_result.adapter,
        action=PromotionAction.ROLLBACK,
        active_version_after=previous_version,
        reason=f"rolled back: {improved_reason}; {guard_reason}",
    )
