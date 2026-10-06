"""Edge-side promotion flow (Integration Sprint · B6, seam C2).

    candidate version + baseline version + EvalResult
        -> POST /adapters/{name}/promote
        -> registry.promotion.decide  (D5, unmodified)
        -> PROMOTE or ROLLBACK, recorded in the registry's audit trail

Three facts about D5 that shape everything here, all read off
``registry/src/registry/promotion.py`` rather than assumed:

1. The rule is TWO-SIDED. It promotes only if in-project ``edit_similarity``
   improves beyond the noise band AND HumanEval pass@1 has not dropped more
   than ``GUARD_PASS_AT_1_TOLERANCE``. A missing guard on either side is not
   neutral — ``_guard_holds`` returns False, so the answer is ROLLBACK.
2. The rule only ever judges the version that is CURRENTLY ACTIVE, and the API
   rejects an eval whose version is not the active one (409).
3. ROLLBACK with no previous version raises, surfaced as a 422. So a promote
   call is only well-formed once the adapter has at least two versions.

Hence the shape of the flow: publish a baseline version, publish the candidate
(which auto-activates), evaluate both, then call promote.

The HumanEval guard
-------------------
``resolve_guard`` never fabricates a pass@1. It reads a real anchor file
produced by ``edge.humaneval_baseline`` (generation) plus evalplus scoring, and
if there is no such file it returns ``None`` with a reason string. The caller
then sends an EvalResult with an empty ``guard`` tuple and D5 does what it was
designed to do when it cannot see the guard: roll back. That is the sprint's
stated F3 position — a legitimate refusal to promote beats a fabricated number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

GUARD_BENCHMARK = "HumanEval"

#: What the registry's D5 rule tolerates, mirrored here only so the manifest
#: can state the threshold that was in force. The rule itself is not
#: re-implemented — ``registry.promotion.decide`` is the only place it lives.
GUARD_PASS_AT_1_TOLERANCE = 0.02


@dataclass(frozen=True)
class GuardStatus:
    """Whether a HumanEval guard number exists, and where it came from."""

    available: bool
    reason: str
    candidate: Optional[Dict] = None   # GuardMetrics dict, or None
    baseline: Optional[Dict] = None
    source: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "source": self.source,
            "benchmark": GUARD_BENCHMARK,
            "tolerance_pass_at_1": GUARD_PASS_AT_1_TOLERANCE,
            "candidate": self.candidate,
            "baseline": self.baseline,
        }


def _guard_metrics(pass_at_1: float) -> Dict:
    return {"benchmark": GUARD_BENCHMARK, "pass_at_k": {"1": float(pass_at_1)}}


def resolve_guard(candidate_anchor: Optional[Path | str] = None,
                  baseline_anchor: Optional[Path | str] = None) -> GuardStatus:
    """Load real HumanEval pass@1 anchors, or say plainly that there are none.

    An anchor is the JSON ``edge.humaneval_baseline`` writes: it carries
    ``base_pass_at_1`` over a frozen task subset together with the decoding
    config and model id that produced it. Both sides are required, because D5
    compares a candidate against a baseline; one anchor alone cannot answer the
    question the rule asks.
    """
    if candidate_anchor is None or baseline_anchor is None:
        return GuardStatus(
            available=False,
            reason=("no HumanEval anchor supplied for candidate and/or baseline — "
                    "evalplus imports the Unix-only `resource` module and cannot "
                    "score on Windows (W1 debt). D5 treats a missing guard as a "
                    "failed guard, so the rule will return ROLLBACK."),
        )
    cand_path, base_path = Path(candidate_anchor), Path(baseline_anchor)
    missing = [str(p) for p in (cand_path, base_path) if not p.exists()]
    if missing:
        return GuardStatus(
            available=False,
            reason=f"HumanEval anchor file(s) not found: {missing}; guard unavailable",
        )
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    for label, payload, path in (("candidate", cand, cand_path), ("baseline", base, base_path)):
        if "base_pass_at_1" not in payload:
            return GuardStatus(
                available=False,
                reason=f"{label} anchor {path} has no base_pass_at_1 field; guard unavailable")
    return GuardStatus(
        available=True,
        reason=(f"HumanEval pass@1 from evalplus over a frozen "
                f"{cand.get('subset', {}).get('n', '?')}-problem subset"),
        candidate=_guard_metrics(cand["base_pass_at_1"]),
        baseline=_guard_metrics(base["base_pass_at_1"]),
        source=f"candidate={cand_path}, baseline={base_path}",
    )


#: What goes in ``in_project`` when nothing was measured. contracts v1.0 has no
#: "unmeasured" encoding — ``InProjectMetrics`` requires four floats — so the
#: honest signal is ``n_examples == 0``: zero examples were scored, therefore the
#: other three fields are not measurements and must not be read as any. Every
#: record produced from this carries ``in_project_measured: false``, and the
#: resulting ROLLBACK is marked provisional. This is the sprint's F3 position:
#: a refusal to promote beats a fabricated edit_similarity.
UNMEASURED_IN_PROJECT: Dict[str, float] = {
    "edit_similarity": 0.0,
    "exact_match": 0.0,
    "perplexity": 0.0,
    "n_examples": 0,
}


def is_measured(in_project: Optional[Dict]) -> bool:
    return bool(in_project) and int(in_project.get("n_examples", 0)) > 0


def build_eval_result(*, adapter_name: str, version: int, kind: str,
                      in_project: Dict, baseline_in_project: Optional[Dict],
                      cluster_id: Optional[str] = None,
                      guard: Sequence[Dict] = (), baseline_noise_band: float = 0.0,
                      seed: int = 0) -> Dict:
    """The JSON body of ``contracts.EvalResult`` the registry's promote endpoint
    parses (``registry.app._eval_result_from``)."""
    return {
        "adapter": {"name": adapter_name, "version": version, "kind": kind,
                    "cluster_id": cluster_id},
        "in_project": in_project,
        "guard": list(guard),
        "baseline_in_project": baseline_in_project,
        "baseline_noise_band": float(baseline_noise_band),
        "seed": seed,
    }


def _guard_contract(metrics: Dict):
    """A guard dict as sent over HTTP -> ``contracts.GuardMetrics`` (int k keys)."""
    from contracts import GuardMetrics

    return GuardMetrics(benchmark=metrics["benchmark"],
                        pass_at_k={int(k): float(v) for k, v in metrics["pass_at_k"].items()})


def predict_decision(in_project: Dict, baseline_in_project: Optional[Dict],
                     noise_band: float, guard: GuardStatus) -> Tuple[str, str]:
    """What D5 will say, and why — for the manifest ONLY.

    Computed by the registry's own rule, ``registry.promotion.decide`` (P4), on
    exactly the inputs being sent: the edge does not keep a copy of D5. The
    authoritative decision is still whatever the live registry returns; this
    sits beside it in the manifest so a disagreement between the inputs'
    implications and the service's answer is visible instead of invisible.

    Returns ("unavailable", why) if the registry package is not installed
    alongside the edge — a prediction the edge cannot make is not made up.
    """
    try:
        from contracts import AdapterKind, AdapterRef, EvalResult, InProjectMetrics
        from registry.promotion import decide
    except ImportError as exc:
        return "unavailable", (f"registry.promotion is not importable here ({exc}); "
                               f"the registry's live answer is the only decision")

    def metrics(d: Dict) -> InProjectMetrics:
        return InProjectMetrics(edit_similarity=float(d["edit_similarity"]),
                                exact_match=float(d["exact_match"]),
                                perplexity=float(d["perplexity"]),
                                n_examples=int(d["n_examples"]))

    # Placeholder versions: decide() reads them only to fill
    # active_version_after; the action and reason depend on the metrics alone.
    result = EvalResult(
        adapter=AdapterRef(name="prediction", version=2, kind=AdapterKind.CLUSTER),
        in_project=metrics(in_project),
        guard=(_guard_contract(guard.candidate),) if guard.available else (),
        baseline_in_project=metrics(baseline_in_project) if baseline_in_project else None,
        baseline_noise_band=float(noise_band))
    baseline_guard = (_guard_contract(guard.baseline),) if guard.available else ()
    decision = decide(result, baseline_guard=baseline_guard,
                      active_version_before=2, previous_version=1)
    return decision.action.value, decision.reason


def promote_candidate(client, adapter_name: str, *, version: int, kind: str,
                      in_project: Optional[Dict], baseline_in_project: Optional[Dict],
                      cluster_id: Optional[str] = None,
                      guard: Optional[GuardStatus] = None,
                      noise_band: float = 0.0, noise_band_source: str = "",
                      seed: int = 0) -> Dict:
    """Run seam C2 and return the registry's decision plus its full input.

    ``client`` is an ``edge.registry_client.RegistryClient``. The returned dict
    is what the round manifest records: the decision the registry made, the
    EvalResult that produced it, and the guard's status — so a reader can check
    the reasoning without re-running anything.
    """
    guard = guard or resolve_guard()
    measured = is_measured(in_project)
    if not measured:
        in_project = dict(UNMEASURED_IN_PROJECT)
    if not is_measured(baseline_in_project):
        baseline_in_project = None
    eval_result = build_eval_result(
        adapter_name=adapter_name, version=version, kind=kind,
        in_project=in_project, baseline_in_project=baseline_in_project,
        cluster_id=cluster_id,
        guard=[guard.candidate] if guard.available else [],
        baseline_noise_band=noise_band, seed=seed)
    baseline_guard: List[Dict] = [guard.baseline] if guard.available else []
    decision = client.promote(adapter_name, eval_result, baseline_guard)
    expected_action, expected_why = predict_decision(
        in_project, baseline_in_project, noise_band, guard)
    return {
        "adapter": adapter_name,
        "candidate_version": version,
        "decision": decision,
        "eval_result_sent": eval_result,
        "baseline_guard_sent": baseline_guard,
        "guard": guard.as_dict(),
        "noise_band": noise_band,
        "noise_band_source": noise_band_source or (
            "PLACEHOLDER 0.0 — with a band of zero 'improved' means 'improved by "
            "any amount'; P5's measured band is still outstanding"),
        "locally_predicted": {"action": expected_action, "why": expected_why,
                              "rule": "registry.promotion.decide"},
        "registry_agrees_with_local_prediction":
            (None if expected_action == "unavailable"
             else decision["action"] == expected_action),
        "in_project_measured": measured,
        "in_project_note": (
            "measured on the client's held-out files"
            if measured else
            "NOT MEASURED — the in_project block sent is the n_examples=0 "
            "placeholder contracts v1.0 forces; its edit_similarity/exact_match/"
            "perplexity are structural zeros, not results"),
        "decision_is_authoritative": guard.available and measured,
        "authority_note": (
            "authoritative: both halves of the D5 two-sided rule were evaluated"
            if (guard.available and measured) else
            "PROVISIONAL: " + "; ".join(
                ([] if guard.available else
                 ["the HumanEval half of D5 could not be scored"])
                + ([] if measured else ["the in-project half was not measured"]))
            + " — this decision reflects a missing input, not a measured regression"),
    }
