"""Evaluation evidence for the live panel's Promote (demo plan v3 item 8).

Promote used to send constants (edit_similarity 0.5/0.45, HumanEval 0.30/0.31).
This module replaces them with numbers evaluation actually measured, and says
where each came from, so the panel can show the guide the provenance next to
the decision. Nothing here makes up a value: a number that was not measured is
absent, and D5 treats an absent guard as a failed guard (ROLLBACK).

Where the numbers come from, first match wins:

1. ``CLASP_EVAL_EVIDENCE`` — a JSON file an evaluation run wrote for *this*
   round (e.g. a GPU laptop re-scoring the live aggregate)::

       {"source": "...", "clusters": {"web": {
           "in_project": {...}, "baseline_in_project": {...},
           "guard": [...], "baseline_guard": [...], "noise_band": 0.0}}}

2. The newest four-seam round manifest (``scripts/demo_round.py``, default
   ``experiments/w12-integration/results/round*_manifest.json``): the
   in-project completion metric (``evaluation.completion``, 60 held-out
   examples, greedy) measured on a GPU for that cluster's candidate and
   baseline versions.

The HumanEval guard comes from evaluation's anchors in
``services/evaluation/results/humaneval_guard/``: ``base_anchor.json`` is the
baseline; a candidate anchor (``<registry name>_anchor.json``) is used only if
evaluation has scored one. The D5 noise band is P5's ``noise_report.json``
value for the cluster's representative client.

Option 2 measures an earlier aggregate of the same cluster, not the one just
built live (scoring a 1.3B model needs a GPU, which laptop A may not have).
``stale`` says so, and the panel shows it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS = Path(os.environ.get(
    "CLASP_EVAL_RESULTS", REPO_ROOT / "services" / "evaluation" / "results"))
ROUND_MANIFESTS = Path(os.environ.get(
    "CLASP_ROUND_MANIFESTS", REPO_ROOT / "experiments" / "w12-integration" / "results"))
GUARD_BENCHMARK = "HumanEval"


class EvidenceMissing(LookupError):
    """No measured in-project metric exists for this cluster."""


def _rel(path: Path) -> str:
    """``path`` as the panel shows it: repo-relative when it can be."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _read(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _guard(pass_at_1: float) -> List[Dict]:
    return [{"benchmark": GUARD_BENCHMARK, "pass_at_k": {"1": float(pass_at_1)}}]


def guard_anchors(registry_name: str, results: Path = EVAL_RESULTS) -> Dict:
    """Evaluation's measured HumanEval pass@1 for the baseline and, if scored, the candidate."""
    folder = results / "humaneval_guard"
    base = _read(folder / "base_anchor.json") or {}
    cand = _read(folder / f"{registry_name}_anchor.json") or {}
    out = {"baseline": None, "candidate": None, "n_problems": base.get("n_problems")}
    if base.get("base_pass_at_1") is not None:
        out["baseline"] = float(base["base_pass_at_1"])
        out["baseline_source"] = _rel(folder / "base_anchor.json")
    if cand.get("base_pass_at_1") is not None:
        out["candidate"] = float(cand["base_pass_at_1"])
        out["candidate_source"] = _rel(folder / f"{registry_name}_anchor.json")
    return out


def noise_band(cluster_id: str, results: Path = EVAL_RESULTS) -> Dict:
    """P5's D5 noise band for this cluster's representative client."""
    report = _read(results / "noise_report.json") or {}
    band = (report.get("in_project") or {}).get("d5_noise_band") or {}
    per_client = band.get("per_client") or {}
    for client, entry in per_client.items():
        if client.split("/", 1)[0] == cluster_id:
            return {"value": float(entry["band"]), "client": client, "note": entry.get("note"),
                    "source": _rel(results / "noise_report.json")}
    if band.get("value") is not None:
        return {"value": float(band["value"]), "client": None, "note": band.get("definition"),
                "source": _rel(results / "noise_report.json")}
    return {"value": 0.0, "client": None, "source": None,
            "note": "no measured band; 0.0 means any improvement counts"}


def _from_override(cluster_id: str) -> Optional[Dict]:
    path = os.environ.get("CLASP_EVAL_EVIDENCE")
    if not path:
        return None
    doc = _read(Path(path))
    entry = ((doc or {}).get("clusters") or {}).get(cluster_id)
    if not entry:
        return None
    return {
        "in_project": entry["in_project"], "baseline_in_project": entry["baseline_in_project"],
        "guard": entry.get("guard", []), "baseline_guard": entry.get("baseline_guard", []),
        "noise_band": float(entry.get("noise_band", 0.0)),
        "provenance": {"in_project": doc.get("source", path), "guard": doc.get("source", path),
                       "noise_band": doc.get("source", path), "stale": False,
                       "note": "measured for this round by an evaluation run (CLASP_EVAL_EVIDENCE)"},
    }


def _newest_round_manifest(cluster_id: str, folder: Path = ROUND_MANIFESTS) -> Optional[tuple]:
    found = []
    for path in folder.glob("round*_manifest.json"):
        doc = _read(path)
        ev = ((doc or {}).get("clusters") or {}).get(cluster_id, {}).get("evaluation") or {}
        if (ev.get("candidate") or {}).get("in_project") and (ev.get("baseline") or {}).get("in_project"):
            found.append((doc.get("finished_utc") or "", path, doc))
    return max(found, key=lambda t: t[0])[1:] if found else None


def evidence_for(cluster_id: str) -> Dict:
    """Measured EvalResult inputs for ``cluster_id``'s Promote, with provenance."""
    override = _from_override(cluster_id)
    if override:
        return override
    hit = _newest_round_manifest(cluster_id)
    if hit is None:
        raise EvidenceMissing(
            f"no measured in-project metric for cluster {cluster_id!r}: run "
            f"scripts/demo_round.py on a GPU machine (writes {_rel(ROUND_MANIFESTS)}"
            f"/round<N>_manifest.json), or point CLASP_EVAL_EVIDENCE at an evaluation run's file")
    path, doc = hit
    cluster = doc["clusters"][cluster_id]
    ev = cluster["evaluation"]
    registry_name = cluster.get("seam_a_b", {}).get("registry_name", f"cluster-{cluster_id}")
    versions = {k: v.get("registry_version", {}).get("ref", {}).get("version")
                for k, v in cluster.get("seam_a_b", {}).get("versions", {}).items()}
    anchors = guard_anchors(registry_name)
    band = noise_band(cluster_id)
    rel = _rel(path)
    return {
        "in_project": ev["candidate"]["in_project"],
        "baseline_in_project": ev["baseline"]["in_project"],
        "guard": _guard(anchors["candidate"]) if anchors["candidate"] is not None else [],
        "baseline_guard": _guard(anchors["baseline"]) if anchors["baseline"] is not None else [],
        "noise_band": band["value"],
        "provenance": {
            "in_project": (f"{rel} — round {doc.get('round')}, {cluster.get('representative_client')}'s "
                           f"held-out files, {ev['candidate']['in_project'].get('n_examples')} examples; "
                           f"candidate {registry_name} v{versions.get('candidate')} vs baseline "
                           f"v{versions.get('baseline')}, measured {doc.get('finished_utc', '')[:10]}"),
            "guard": (f"baseline pass@1 {anchors['baseline']} over {anchors['n_problems']} HumanEval "
                      f"problems ({anchors.get('baseline_source')}); candidate "
                      + (f"{anchors['candidate']} ({anchors['candidate_source']})"
                         if anchors["candidate"] is not None else
                         "not scored by evaluation — D5 counts a missing guard as failed")),
            "noise_band": f"{band['value']} from {band['source'] or 'nowhere'} ({band.get('client') or 'all'})",
            "stale": True,
            "note": ("measured on this cluster's round-"
                     f"{doc.get('round')} aggregate, not re-scored for the aggregate built live "
                     "(scoring the 1.3B model needs a GPU)"),
        },
    }
