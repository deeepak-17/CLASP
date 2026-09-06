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


def predict_decision(in_project: Dict, baseline_in_project: Optional[Dict],
                     noise_band: float, guard: GuardStatus) -> Tuple[str, str]:
    """What D5 will say, and why — computed locally for the manifest ONLY.

    The authoritative decision is whatever the registry returns; this exists so
    the round manifest can record the inputs' implications next to the actual
    answer, which makes a disagreement between the two visible instead of
    invisible. It is never substituted for the registry's decision.
    """
    if baseline_in_project is None:
        improved, why = False, "no baseline_in_project"
    else:
        delta = in_project["edit_similarity"] - baseline_in_project["edit_similarity"]
        improved = delta > noise_band
        why = f"edit_similarity delta {delta:+.4f} vs band {noise_band:.4f}"
    if not guard.available:
        return "rollback", f"{why}; guard unavailable ({guard.reason.split('—')[0].strip()})"
    drop = guard.baseline["pass_at_k"]["1"] - guard.candidate["pass_at_k"]["1"]
    guard_ok = drop <= GUARD_PASS_AT_1_TOLERANCE
    action = "promote" if (improved and guard_ok) else "rollback"
    return action, f"{why}; HumanEval pass@1 drop {drop:+.4f}"


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
        "locally_predicted": {"action": expected_action, "why": expected_why},
        "registry_agrees_with_local_prediction":
            decision["action"] == expected_action,
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
