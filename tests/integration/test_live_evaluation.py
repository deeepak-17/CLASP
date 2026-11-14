"""Promote judges the candidate on its own measurements (D5), never on another adapter's.

``scripts/evaluate_candidate.py`` scores the registry's active version and the
one before it on laptop A's GPU and writes ``<cluster>-v<N>.json``;
``demo_ui.evidence.evidence_for`` hands Promote that file only when its
version and sha256 are the candidate's. The GPU scoring itself runs in the
live check; here its pure parts and the selection rule are pinned.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from demo_ui import evidence

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_candidate.py"


def _script():
    spec = importlib.util.spec_from_file_location("evaluate_candidate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v(version, sha):
    return {"ref": {"name": "cluster-web", "version": version}, "sha256": sha, "round": version - 1}


SCORES = {
    "candidate": {"in_project": {"edit_similarity": 0.52, "exact_match": 0.2, "perplexity": 3.1,
                                 "n_examples": 60}, "pass_at_1": 0.45},
    "baseline": {"in_project": {"edit_similarity": 0.50, "exact_match": 0.2, "perplexity": 3.2,
                                "n_examples": 60}, "pass_at_1": 0.5},
}


def test_the_candidate_is_the_active_version_and_the_baseline_the_one_before():
    mod = _script()
    listing = {"active": 3, "versions": [_v(1, "a" * 64), _v(3, "c" * 64), _v(2, "b" * 64)]}
    cand, base = mod.pick_versions(listing)
    assert cand["ref"]["version"] == 3 and base["ref"]["version"] == 2
    with pytest.raises(SystemExit, match="previous version"):
        mod.pick_versions({"active": 1, "versions": [_v(1, "a" * 64)]})
    with pytest.raises(SystemExit, match="no active version"):
        mod.pick_versions({"active": None, "versions": []})


def _write(tmp_path, mod, version=2, sha="b" * 64):
    doc = mod.evidence_document("web", _v(version, sha), _v(version - 1, "a" * 64),
                                SCORES["candidate"], SCORES["baseline"],
                                band={"value": 0.01, "client": "web/client-flask", "source": "x"},
                                details={"alpha": 0.5})
    path = mod.evidence_path(tmp_path, "web", version)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_evidence_document_is_what_promote_sends(tmp_path):
    doc = _write(tmp_path, _script())
    entry = doc["clusters"]["web"]
    assert entry["in_project"]["edit_similarity"] == 0.52
    assert entry["baseline_in_project"]["edit_similarity"] == 0.50
    assert entry["guard"] == [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.45}}]
    assert entry["baseline_guard"] == [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.5}}]
    assert entry["noise_band"] == 0.01
    assert doc["candidate_version"] == 2 and doc["candidate_sha256"] == "b" * 64
    assert doc["baseline_version"] == 1


def test_promote_gets_the_candidates_own_evaluation_or_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("CLASP_EVAL_EVIDENCE", raising=False)
    monkeypatch.setattr(evidence, "LIVE_EVAL_DIR", tmp_path)
    _write(tmp_path, _script())
    ev = evidence.evidence_for("web", version=2, sha256="b" * 64)
    assert ev["in_project"]["edit_similarity"] == 0.52
    assert ev["provenance"]["stale"] is False and "v2" in ev["provenance"]["in_project"]
    # the same version number with other bytes is another adapter: not its evidence
    with pytest.raises(evidence.EvidenceMissing, match="Evaluate"):
        evidence.evidence_for("web", version=2, sha256="c" * 64)
    # and an unevaluated version never falls back to an older round's numbers
    with pytest.raises(evidence.EvidenceMissing, match="Evaluate"):
        evidence.evidence_for("web", version=3, sha256="d" * 64)


def test_an_override_file_for_another_version_does_not_win(tmp_path, monkeypatch):
    """Review #4: CLASP_EVAL_EVIDENCE counts only if it names this version and sha."""
    mod = _script()
    monkeypatch.setattr(evidence, "LIVE_EVAL_DIR", tmp_path / "none")
    doc = mod.evidence_document("web", _v(1, "a" * 64), _v(0 + 1, "a" * 64), SCORES["candidate"],
                                SCORES["baseline"], band={"value": 0.0}, details={})
    f = tmp_path / "override.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setenv("CLASP_EVAL_EVIDENCE", str(f))
    with pytest.raises(evidence.EvidenceMissing):
        evidence.evidence_for("web", version=2, sha256="b" * 64)
    assert evidence.evidence_for("web", version=1, sha256="a" * 64)["in_project"]


def test_the_baseline_is_the_candidates_parent_not_the_highest_lower_number():
    """Review #2: v4 built on v2 (v3 was rolled back) is compared with v2."""
    mod = _script()
    listing = {"active": 4, "versions": [_v(1, "a" * 64), _v(2, "b" * 64), _v(3, "c" * 64),
                                         _v(4, "d" * 64)]}
    cand, base = mod.pick_versions(listing, parent_sha256="b" * 64)
    assert cand["ref"]["version"] == 4 and base["ref"]["version"] == 2
