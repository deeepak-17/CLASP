"""Two-sided promotion rule (D5) — P4 owns the rule, P5 owns the metrics that feed it.

    Promote iff:
      1. in-project quality improved beyond the measured noise band, AND
      2. next-line completion edit similarity on the frozen guard set
         (benchmark ``LineCompletion``) did not drop against the baseline by
         more than 2 points absolute.
    Otherwise: rollback to the previous version. Never gate on one side alone.

The guard replaced HumanEval pass@1 (contracts >= 1.2.0 carries edit_similarity
on GuardMetrics). A guard without ``LineCompletion`` edit similarity on both
sides is not a pass: the rule refuses to promote, and the API answers 422
before anything is recorded.

The registry always activates a version on save (see storage.save); this rule
runs *after* the fact, on whatever is currently active, and either confirms
that choice (PROMOTE) or reverts it (ROLLBACK to the previous version). It is
a pure function — no filesystem access — so it's cheap to unit test the D5
math in isolation from the storage/HTTP layers.
"""
from __future__ import annotations

from contracts import AdapterRef, EvalResult, GuardMetrics, PromotionAction, PromotionDecision

#: The benchmark name the guard is reported under (evaluation's line guard).
GUARD_BENCHMARK = "LineCompletion"

#: Absolute edit-similarity drop tolerated before it blocks a promotion (D5),
#: a fraction since edit similarity is 0..1 (0.02 = 2 points).
GUARD_EDIT_SIMILARITY_TOLERANCE = 0.02

GUARD_MISSING_REASON = (
    f"{GUARD_BENCHMARK} guard metrics missing on candidate or baseline "
    "(contracts >= 1.2.0 required)"
)


def _in_project_improved(eval_result: EvalResult) -> tuple[bool, str]:
    baseline = eval_result.baseline_in_project
    if baseline is None:
        return False, "no baseline_in_project supplied — cannot confirm improvement"
    delta = eval_result.in_project.edit_similarity - baseline.edit_similarity
    band = eval_result.baseline_noise_band
    if delta > band:
        return True, f"edit_similarity +{delta:.4f} clears noise band {band:.4f}"
    return False, f"edit_similarity delta {delta:.4f} does not clear noise band {band:.4f}"


def guard_present(
    guard: tuple[GuardMetrics, ...], baseline_guard: tuple[GuardMetrics, ...]
) -> bool:
    """Whether both sides carry a ``LineCompletion`` entry with an edit similarity."""
    return all(
        entry is not None and entry.edit_similarity is not None
        for entry in (
            next((g for g in guard if g.benchmark == GUARD_BENCHMARK), None),
            next((g for g in baseline_guard if g.benchmark == GUARD_BENCHMARK), None),
        )
    )


def _guard_holds(
    eval_result: EvalResult, baseline_guard: tuple[GuardMetrics, ...]
) -> tuple[bool, str]:
    if not guard_present(eval_result.guard, baseline_guard):
        return False, GUARD_MISSING_REASON
    candidate = next(g for g in eval_result.guard if g.benchmark == GUARD_BENCHMARK)
    baseline = next(g for g in baseline_guard if g.benchmark == GUARD_BENCHMARK)
    drop = baseline.edit_similarity - candidate.edit_similarity
    tolerance = GUARD_EDIT_SIMILARITY_TOLERANCE
    # the epsilon absorbs float noise at exactly the tolerance (0.60 - 0.58 > 0.02 in binary)
    if drop <= tolerance + 1e-9:
        return True, f"{GUARD_BENCHMARK} edit similarity drop {drop:.4f} within {tolerance:.2f} tolerance"
    return False, f"{GUARD_BENCHMARK} edit similarity dropped {drop:.4f}, exceeds {tolerance:.2f} tolerance"


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


def restore_decision(
    candidate: AdapterRef, *, to_version: int, reason: str
) -> PromotionDecision:
    """An operator restore, recorded in the same audit trail as D5 decisions.

    Moving back is a ROLLBACK; moving forward (e.g. re-activating a fixed
    version) is recorded as PROMOTE but says plainly that no D5 rule ran.
    """
    backwards = to_version < candidate.version
    gate = "" if backwards else " — not a D5 decision"
    return PromotionDecision(
        adapter=candidate,
        action=PromotionAction.ROLLBACK if backwards else PromotionAction.PROMOTE,
        active_version_after=to_version,
        reason=f"operator restore v{candidate.version} -> v{to_version}{gate}: {reason}",
    )
