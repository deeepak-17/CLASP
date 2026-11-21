"""Two-sided promotion rule (D5) — P4 owns the rule, P5 owns the metrics that feed it.

    Promote iff:
      1. in-project quality improved beyond the measured noise band, AND
      2. next-line completion edit similarity on the frozen guard set
         (benchmark ``LineCompletion``) did not drop against the baseline by
         more than 2 points absolute.
    Otherwise: reject the candidate. Never gate on one side alone.

The guard replaced HumanEval pass@1 (contracts >= 1.2.0 carries edit_similarity
on GuardMetrics). A guard without ``LineCompletion`` edit similarity on both
sides is not a pass: the rule refuses to promote, and the API answers 422
before anything is recorded.

An uploaded version is an inactive *candidate* (see storage.save): the live
pointer does not move on upload. This rule runs on that candidate and either
makes it active (PROMOTE) or records REJECT and leaves the pointer where it
was, so a failed candidate never goes live. It is a pure function — no filesystem access — so it's cheap to unit test the D5
math in isolation from the storage/HTTP layers.
"""
from __future__ import annotations

from contracts import AdapterRef, EvalResult, GuardMetrics, PromotionAction, PromotionDecision

#: The benchmark name the guard is reported under (evaluation's line guard).
GUARD_BENCHMARK = "LineCompletion"

#: Absolute edit-similarity drop tolerated before it blocks a promotion (D5),
#: a fraction since edit similarity is 0..1 (0.02 = 2 points).
GUARD_EDIT_SIMILARITY_TOLERANCE = 0.02

#: Operator restores are recorded in the same log; their reason starts with this.
RESTORE_REASON_PREFIX = "operator restore"

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
    active_version_before: int | None,
) -> PromotionDecision:
    """Apply the D5 two-sided rule to an inactive candidate.

    PROMOTE makes the evaluated version active. Otherwise the decision is
    REJECT and the pointer stays where it was: ``active_version_after`` is
    ``active_version_before``, or 0 ("nothing active") when there is none.
    """
    improved, improved_reason = _in_project_improved(eval_result)
    guard_ok, guard_reason = _guard_holds(eval_result, baseline_guard)

    if improved and guard_ok:
        return PromotionDecision(
            adapter=eval_result.adapter,
            action=PromotionAction.PROMOTE,
            active_version_after=eval_result.adapter.version,
            reason=f"promoted: {improved_reason}; {guard_reason}",
        )
    return PromotionDecision(
        adapter=eval_result.adapter,
        action=PromotionAction.REJECT,
        active_version_after=active_version_before if active_version_before is not None else 0,
        reason=f"rejected: {improved_reason}; {guard_reason}",
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
        reason=f"{RESTORE_REASON_PREFIX} v{candidate.version} -> v{to_version}{gate}: {reason}",
    )


def live_versions(decisions: list[PromotionDecision]) -> set[int]:
    """Versions that were active at some point, as far as the audit trail knows.

    A version is live if it was promoted, it is where the pointer went after
    any recorded decision (operator restores included), or an operator restore
    moved the pointer away from it (it was active then). Versions activated
    without any such record (older volumes, ``set_active`` uploads) are absent.
    """
    live = {d.adapter.version for d in decisions if d.action == PromotionAction.PROMOTE}
    live |= {d.active_version_after for d in decisions if d.active_version_after > 0}
    live |= {d.adapter.version for d in decisions if d.reason.startswith(RESTORE_REASON_PREFIX)}
    return live


#: A restore's composite moves are recorded with this prefix (``cascade.apply_cascade``).
CASCADE_REASON_PREFIX = "cascade "


def is_refusal(decision: PromotionDecision) -> bool:
    """Whether ``decision`` refused the version it names — the one refusal rule.

    A REJECT refuses its candidate. So did a D5 ROLLBACK from before the
    candidate flow (uploads went live, a failing rule rolled the pointer back
    and recorded ``rollback`` naming the refused candidate). A ROLLBACK that is
    an operator restore, or a composite following one (cascade), refuses nothing.
    """
    if decision.action == PromotionAction.REJECT:
        return True
    return decision.action == PromotionAction.ROLLBACK and not decision.reason.startswith(
        (RESTORE_REASON_PREFIX, CASCADE_REASON_PREFIX))


def rejected_versions(decisions: list[PromotionDecision]) -> set[int]:
    """Versions refused by the rule above: never a candidate, never a restore target."""
    return {d.adapter.version for d in decisions if is_refusal(d)}


def candidate_version(
    versions: list[int], active: int | None, decided: set[int]
) -> int | None:
    """The newest version above ``active`` that has no decision yet, else None."""
    floor = active if active is not None else 0
    pending = [v for v in versions if v > floor and v not in decided]
    return max(pending) if pending else None
