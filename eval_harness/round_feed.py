"""Dashboard feed for federated rounds — ``round{N}_manifest.json`` -> ``rounds.json``.

``scripts/demo_round.py`` (the one-command four-seam round) writes everything
a round did to ``experiments/<run>/results/round{N}_manifest.json``: both
registry versions per cluster, their in-project metrics, the HumanEval guard
status and the registry's D5 decision. That file is large and nested; the
dashboard is a static app that reads small JSON from ``public/data/``. This
module is the bridge: it reads one or more round manifests and writes the
compact ``eval_harness/results/rounds.json`` the dashboard's Rounds page
renders.

Every field is read from the manifest, never recomputed or filled in. A
metric the round did not measure stays ``null``; a missing manifest key is an
error naming the path, so a change on the producer's side fails loudly here
instead of rendering a blank chart.

Regression alerts (D5 cross-check)
-----------------------------------
For each cluster the feed raises an alert for every reason the registry's D5
rule (``registry.promotion.decide``) would refuse a promotion — in-project
gain not beyond the noise band, HumanEval pass@1 drop beyond tolerance, or an
input that was not measured — plus a round-over-round drop of a cluster's
candidate edit similarity. From the D5 reasons it predicts the action and
records whether the registry's actual decision agrees; a disagreement is
itself an alert, because it means either this logic or the rule changed.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from utils.errors import EvaluationError
from utils.timing import utc_timestamp

#: Version of the ``rounds.json`` shape (``dashboard/src/lib/rounds.ts`` mirrors it).
FEED_VERSION = "1.0.0"

#: Mirrors ``registry.promotion.GUARD_PASS_AT_1_TOLERANCE`` (asserted equal in
#: the tests when the registry package is importable).
GUARD_PASS_AT_1_TOLERANCE = 0.02

_LABELS = ("baseline", "candidate")


def _get(data: Mapping[str, Any], *path: str) -> Any:
    node: Any = data
    for i, key in enumerate(path):
        if not isinstance(node, Mapping) or key not in node:
            raise EvaluationError(f"round manifest is missing '{'.'.join(path[: i + 1])}'")
        node = node[key]
    return node


def _in_project(evaluation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The scored in-project block, or ``None`` when the round did not measure it."""
    if not evaluation:
        return None
    metrics = evaluation.get("in_project")
    if not metrics or int(metrics.get("n_examples", 0)) <= 0:
        return None
    return {
        "edit_similarity": float(metrics["edit_similarity"]),
        "exact_match": float(metrics["exact_match"]),
        "perplexity": float(metrics["perplexity"]),
        "n_examples": int(metrics["n_examples"]),
        "examples_sha256": evaluation.get("examples_sha256"),
    }


def _pass_at_1(guard_side: Mapping[str, Any] | None) -> float | None:
    if not guard_side:
        return None
    value = (guard_side.get("pass_at_k") or {}).get("1")
    return None if value is None else float(value)


def summarize_round(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """One round manifest -> one ``rounds.json`` round entry."""
    guard = _get(manifest, "humaneval_guard")
    noise_band = float(_get(manifest, "config", "noise_band"))
    clusters_out: list[dict[str, Any]] = []
    for cluster_id, block in sorted(_get(manifest, "clusters").items()):
        versions = _get(block, "seam_a_b", "versions")
        evaluation = block.get("evaluation") or {}
        decision = _get(block, "seam_c2", "decision")
        entry_versions: dict[str, Any] = {}
        for label in _LABELS:
            meta = _get(versions, label, "registry_version")
            entry_versions[label] = {
                "version": int(_get(meta, "ref", "version")),
                "aggregation": _get(versions, label, "aggregation"),
                "sha256": meta.get("sha256"),
                "in_project": _in_project(evaluation.get(label)),
            }
        base_ip = entry_versions["baseline"]["in_project"]
        cand_ip = entry_versions["candidate"]["in_project"]
        delta = (
            round(cand_ip["edit_similarity"] - base_ip["edit_similarity"], 6)
            if base_ip and cand_ip
            else None
        )
        clusters_out.append(
            {
                "cluster_id": cluster_id,
                "registry_name": _get(block, "seam_a_b", "registry_name"),
                "representative_client": block.get("representative_client"),
                "baseline": entry_versions["baseline"],
                "candidate": entry_versions["candidate"],
                "edit_similarity_delta": delta,
                "decision": {
                    "action": str(_get(decision, "action")),
                    "active_version_after": int(_get(decision, "active_version_after")),
                    "reason": str(decision.get("reason", "")),
                },
                "authoritative": bool(block["seam_c2"].get("decision_is_authoritative", False)),
            }
        )
    return {
        "round": int(_get(manifest, "round")),
        "finished_utc": manifest.get("finished_utc"),
        "wall_minutes": manifest.get("wall_minutes"),
        "nfr_round_minutes": manifest.get("nfr_round_minutes"),
        "nfr_met": manifest.get("nfr_met"),
        "noise_band": noise_band,
        "counters": dict(manifest.get("counters") or {}),
        "guard": {
            "available": bool(guard.get("available", False)),
            "reason": str(guard.get("reason", "")),
            "candidate_pass_at_1": _pass_at_1(guard.get("candidate")),
            "baseline_pass_at_1": _pass_at_1(guard.get("baseline")),
        },
        "clusters": clusters_out,
    }


def d5_alerts(round_entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Alerts for one round, plus the D5 cross-check against the registry's decision."""
    alerts: list[dict[str, Any]] = []
    guard = round_entry["guard"]
    band = float(round_entry["noise_band"])
    for cluster in round_entry["clusters"]:
        reasons: list[tuple[str, str]] = []
        delta = cluster["edit_similarity_delta"]
        if delta is None:
            reasons.append(("in_project_unmeasured", "in-project metric not measured on both versions"))
        elif delta <= band:
            reasons.append(
                ("in_project_not_improved", f"edit_similarity delta {delta:+.4f} does not clear band {band:.4f}")
            )
        if not guard["available"] or guard["candidate_pass_at_1"] is None or guard["baseline_pass_at_1"] is None:
            reasons.append(("guard_unavailable", "HumanEval guard not scored for candidate and baseline"))
        else:
            drop = guard["baseline_pass_at_1"] - guard["candidate_pass_at_1"]
            if drop > GUARD_PASS_AT_1_TOLERANCE:
                reasons.append(
                    ("guard_regression", f"HumanEval pass@1 dropped {drop:.4f} (> {GUARD_PASS_AT_1_TOLERANCE:.2f})")
                )
        predicted = "rollback" if reasons else "promote"
        actual = cluster["decision"]["action"]
        common = {"round": round_entry["round"], "cluster_id": cluster["cluster_id"]}
        for kind, message in reasons:
            alerts.append({**common, "kind": kind, "message": message})
        if predicted != actual:
            alerts.append(
                {
                    **common,
                    "kind": "decision_mismatch",
                    "message": f"D5 inputs imply {predicted.upper()} but the registry decided {actual.upper()}",
                }
            )
    return alerts


def round_over_round_alerts(rounds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A cluster's candidate edit similarity falling from one round to the next."""
    alerts: list[dict[str, Any]] = []
    previous: dict[str, float] = {}
    for entry in rounds:
        for cluster in entry["clusters"]:
            ip = cluster["candidate"]["in_project"]
            if ip is None:
                continue
            current = ip["edit_similarity"]
            before = previous.get(cluster["cluster_id"])
            if before is not None and current < before:
                alerts.append(
                    {
                        "round": entry["round"],
                        "cluster_id": cluster["cluster_id"],
                        "kind": "round_over_round_drop",
                        "message": f"candidate edit_similarity {before:.4f} -> {current:.4f}",
                    }
                )
            previous[cluster["cluster_id"]] = current
    return alerts


def build_feed(manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every round manifest -> the ``rounds.json`` document, rounds in order."""
    rounds = sorted((summarize_round(m) for m in manifests), key=lambda r: r["round"])
    seen: set[int] = set()
    for entry in rounds:
        if entry["round"] in seen:
            raise EvaluationError(f"round {entry['round']} supplied twice")
        seen.add(entry["round"])
    alerts = [a for entry in rounds for a in d5_alerts(entry)] + round_over_round_alerts(rounds)
    return {
        "feed_version": FEED_VERSION,
        "generated_at": utc_timestamp(),
        "guard_pass_at_1_tolerance": GUARD_PASS_AT_1_TOLERANCE,
        "rounds": rounds,
        "alerts": alerts,
    }
