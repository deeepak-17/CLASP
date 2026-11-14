"""Edge-side promotion flow (Integration Sprint · B6, seam C2).

    candidate version + baseline version + EvalResult
        -> POST /adapters/{name}/promote
        -> registry.promotion.decide  (D5, unmodified)
        -> PROMOTE or ROLLBACK, recorded in the registry's audit trail

Three facts about D5 that shape everything here, all read off
``registry/src/registry/promotion.py`` rather than assumed:

1. The rule is TWO-SIDED. It promotes only if in-project ``edit_similarity``
   improves beyond the noise band AND the regression guard has not dropped more
   than ``GUARD_TOLERANCE``. A missing guard on either side is not
   neutral — ``_guard_holds`` returns False, so the answer is ROLLBACK. The
   guard is next-line completion on the frozen line-completion guard set
   (``LineCompletion``), and the registry's rule reads exactly that.
2. The rule only ever judges the version that is CURRENTLY ACTIVE, and the API
   rejects an eval whose version is not the active one (409).
3. ROLLBACK with no previous version raises, surfaced as a 422. So a promote
   call is only well-formed once the adapter has at least two versions.

Hence the shape of the flow: publish a baseline version, publish the candidate
(which auto-activates), evaluate both, then call promote.

The line-completion guard
-------------------------
``resolve_guard`` never fabricates a guard score. It reads the evidence file
``scripts/evaluate_candidate.py`` wrote (the candidate's and the baseline's
``LineCompletion`` metrics on the frozen guard set), and if there is no such
file it returns an unavailable status with a reason string. The caller
then sends an EvalResult with an empty ``guard`` tuple and D5 does what it was
designed to do when it cannot see the guard: roll back. That is the sprint's
stated F3 position — a legitimate refusal to promote beats a fabricated number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Edge-local ``decision.action`` for a promote that was never sent (not a
#: contracts ``PromotionAction``; the registry never produces it).
NOT_SENT = "not_sent"

GUARD_BENCHMARK = "LineCompletion"  # evaluation.eval_harness.line_guard.GUARD_BENCHMARK

#: The edit-similarity drop the line-completion guard tolerates
#: (``guard_thresholds.yaml``: ``line_guard.tolerance_edit_similarity``), stated
#: here only so the manifest can say which threshold was meant. The rule itself
#: is not re-implemented — ``registry.promotion.decide`` is the only place it lives.
GUARD_TOLERANCE = 0.02


@dataclass(frozen=True)
class GuardStatus:
    """Whether a line-completion guard score exists, and where it came from."""

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
            "tolerance_edit_similarity": GUARD_TOLERANCE,
            "candidate": self.candidate,
            "baseline": self.baseline,
        }


_NO_GUARD_EVIDENCE = ("no line-completion guard evidence supplied: the line-completion guard was "
                      "not scored (scripts/evaluate_candidate.py writes it). D5 treats a missing "
                      "guard as a failed guard, so the rule will return ROLLBACK.")


def _line_guard(entries) -> Optional[Dict]:
    """The ``LineCompletion`` entry of a guard list, or None."""
    return next((g for g in entries or ()
                 if isinstance(g, dict) and g.get("benchmark") == GUARD_BENCHMARK
                 and g.get("edit_similarity") is not None), None)


def resolve_guard(evidence: Optional[Path | str] = None,
                  cluster_id: Optional[str] = None) -> GuardStatus:
    """Load the line-completion guard from an evidence file, or say plainly there is none.

    ``evidence`` is the JSON ``scripts/evaluate_candidate.py`` writes: its
    ``clusters[<cluster>]`` entry carries the candidate's ``guard`` and the
    ``baseline_guard``, each a ``LineCompletion`` metrics dict (``GuardMetrics.
    to_json`` shape). Both sides are required, because D5 compares a candidate
    against a baseline; one side alone cannot answer the question the rule asks.
    ``cluster_id`` picks the entry when the file holds several.
    """
    if evidence is None:
        return GuardStatus(available=False, reason=_NO_GUARD_EVIDENCE)
    path = Path(evidence)
    if not path.exists():
        return GuardStatus(
            available=False,
            reason=f"line-completion guard evidence file not found: {path}; guard unavailable")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return GuardStatus(available=False,
                           reason=f"line-completion guard evidence {path} is unreadable ({exc}); "
                                  f"guard unavailable")
    clusters = (doc.get("clusters") or {}) if isinstance(doc, dict) else {}
    if cluster_id is None and len(clusters) == 1:
        cluster_id = next(iter(clusters))
    entry = clusters.get(cluster_id) if cluster_id is not None else None
    if not entry:
        return GuardStatus(
            available=False,
            reason=(f"line-completion guard evidence {path} has no entry for cluster "
                    f"{cluster_id!r} (it holds {sorted(clusters)}); guard unavailable"))
    cand, base = _line_guard(entry.get("guard")), _line_guard(entry.get("baseline_guard"))
    if cand is None or base is None:
        return GuardStatus(
            available=False,
            reason=(f"evidence {path} has no {GUARD_BENCHMARK} guard for the "
                    f"{'candidate' if cand is None else 'baseline'}; guard unavailable"))
    return GuardStatus(
        available=True,
        reason=(f"line-completion guard: next-line edit similarity on the frozen "
                f"{cand.get('n_examples', '?')}-example guard set, from {path.name}"),
        candidate=dict(cand), baseline=dict(base), source=str(path))


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

    return GuardMetrics.from_json(metrics)


def _previous_version(versions: Sequence[int], version: int) -> Optional[int]:
    """The rollback target D5 would use: the version just before ``version``.

    Mirrors ``registry.storage.previous_version`` over a version list the
    registry itself returned; None when ``version`` is the first one.
    """
    earlier = [v for v in versions if v < version]
    return max(earlier) if earlier else None


def predict_decision(eval_result: Dict, baseline_guard: Sequence[Dict],
                     previous_version: Optional[int]) -> Tuple[str, str]:
    """What D5 will say, and why — for the manifest ONLY.

    Computed by the registry's own rule, ``registry.promotion.decide`` (P4), on
    exactly the inputs being sent: the EvalResult body and baseline guard of
    seam C2, with the candidate as the active version (the only one the
    registry evaluates) and ``previous_version`` as the rollback target. The
    edge does not keep a copy of D5. The authoritative decision is still
    whatever the live registry returns; this sits beside it in the manifest so
    a disagreement between the inputs' implications and the service's answer is
    visible instead of invisible.

    Returns ("unavailable", why) if the registry package is not installed
    alongside the edge, or if the rule cannot decide on these inputs (a
    rollback with no previous version) — a prediction the edge cannot make is
    not made up.
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

    ref = eval_result["adapter"]
    baseline = eval_result.get("baseline_in_project")
    result = EvalResult(
        adapter=AdapterRef(name=ref["name"], version=int(ref["version"]),
                           kind=AdapterKind(ref["kind"]), cluster_id=ref.get("cluster_id")),
        in_project=metrics(eval_result["in_project"]),
        guard=tuple(_guard_contract(g) for g in eval_result.get("guard", ())),
        baseline_in_project=metrics(baseline) if baseline else None,
        baseline_noise_band=float(eval_result.get("baseline_noise_band", 0.0)),
        seed=int(eval_result.get("seed", 0)))
    try:
        decision = decide(result,
                          baseline_guard=tuple(_guard_contract(g) for g in baseline_guard),
                          active_version_before=result.adapter.version)
    except ValueError as exc:
        return "unavailable", f"registry.promotion.decide could not decide: {exc}"
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
    # Read before C2: a rollback moves the active pointer, not the history.
    listing = client.list_versions(adapter_name)
    versions = [v["ref"]["version"] for v in listing["versions"]]
    previous_version = _previous_version(versions, version)
    expected_action, expected_why = predict_decision(
        eval_result, baseline_guard, previous_version)
    if guard.available:
        decision = client.promote(adapter_name, eval_result, baseline_guard)
    else:
        # The registry refuses a promote that carries no guard (422, nothing
        # recorded), so it is not asked. This is an edge-local record, not a
        # registry decision: its pointer is where it was.
        decision = {
            "adapter": eval_result["adapter"],
            "action": NOT_SENT,
            "active_version_after": listing.get("active"),
            "registry_called": False,
            "reason": f"promote not sent: {guard.reason}",
            "predicted": {"action": expected_action, "why": expected_why},
        }
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
            (None if expected_action == "unavailable" or decision["action"] == NOT_SENT
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
                 ["the line-completion guard half of D5 could not be scored"])
                + ([] if measured else ["the in-project half was not measured"]))
            + " — this decision reflects a missing input, not a measured regression"),
    }
