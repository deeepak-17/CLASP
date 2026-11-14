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


def _guard_score(es, rows):
    return {"benchmark": "LineCompletion", "edit_similarity": es, "exact_match": 0.1,
            "n_examples": len(rows)}


SCORES = {
    "candidate": {"in_project": {"edit_similarity": 0.52, "exact_match": 0.2, "perplexity": 3.1,
                                 "n_examples": 60},
                  "guard": _guard_score(0.60, [0.6, 0.6, 0.7, 0.5]), "guard_rows": [0.6, 0.6, 0.7, 0.5],
                  "guard_examples_sha256": "f" * 64},
    "baseline": {"in_project": {"edit_similarity": 0.50, "exact_match": 0.2, "perplexity": 3.2,
                                "n_examples": 60},
                 "guard": _guard_score(0.61, [0.6, 0.7, 0.6, 0.5]), "guard_rows": [0.6, 0.7, 0.6, 0.5],
                 "guard_examples_sha256": "f" * 64},
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
    assert entry["guard"] == [SCORES["candidate"]["guard"]]
    assert entry["baseline_guard"] == [SCORES["baseline"]["guard"]]
    assert entry["guard"][0]["benchmark"] == "LineCompletion"
    assert isinstance(doc["guard_noise"], float) and doc["guard_noise"] >= 0.0
    assert entry["guard_noise"] == doc["guard_noise"]
    assert doc["guard_examples_sha256"] == "f" * 64
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


# --------------------------------------------------------------------------- #
# score_version: the guard set replaces HumanEval
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Enough of a PeftModel for score_version's adapter bookkeeping."""

    def __init__(self):
        self.peft_config = {}

    def load_adapter(self, path, adapter_name):
        self.peft_config[adapter_name] = path

    def set_adapter(self, name):
        pass

    def eval(self):
        pass

    def delete_adapter(self, name):
        self.peft_config.pop(name, None)


class _FakeTokenizer:
    pad_token_id = 0


def _score_with_fake_model(tmp_path, monkeypatch, *, alpha=0.5):
    import argparse

    pytest.importorskip("torch")
    from edge import completion_eval, train_client
    from evaluation.eval_harness.line_guard import load_guard_set

    mod = _script()
    guard_examples = load_guard_set()
    calls = []

    def fake_evaluate_completion(model, tokenizer, examples, perplexity, **kw):
        calls.append(len(examples))
        rows = [{"edit_similarity": 0.5 + (i % 3) / 10, "exact_match": i % 5 == 0}
                for i in range(len(examples))]
        es = sum(r["edit_similarity"] for r in rows) / len(rows)
        em = sum(r["exact_match"] for r in rows) / len(rows)
        return {"in_project": {"edit_similarity": es, "exact_match": em,
                               "perplexity": perplexity, "n_examples": len(examples)},
                "examples_sha256": "e" * 64, "rows": rows, "wall_seconds": 0.0}

    monkeypatch.setattr(completion_eval, "evaluate_completion", fake_evaluate_completion)
    monkeypatch.setattr(train_client, "evaluate",
                        lambda model, chunks, pad: {"perplexity": 3.0})
    monkeypatch.setattr(mod, "_composite", lambda *a, **k: tmp_path / "composite")
    state = {"model": _FakeModel(), "tokenizer": _FakeTokenizer(), "peft": _FakeModel(),
             "client": "web/client-flask", "examples": guard_examples[:5],
             "guard_examples": guard_examples, "held_chunks": [[1]]}
    args = argparse.Namespace(alpha=alpha, ppl_max_blocks=4)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    result = mod.score_version(state, None, "cluster-web", _v(2, "b" * 64), args, work)
    return mod, state, args, work, result, calls


def test_score_version_scores_the_guard_set_not_humaneval(tmp_path, monkeypatch):
    _, _, _, _, result, calls = _score_with_fake_model(tmp_path, monkeypatch)
    assert result["guard"]["benchmark"] == "LineCompletion"
    assert result["guard"]["n_examples"] == 300
    assert set(result["guard"]) == {"benchmark", "edit_similarity", "exact_match", "n_examples"}
    assert len(result["guard_rows"]) == 300
    for gone in ("pass_at_1", "humaneval_solved", "humaneval_n"):
        assert gone not in result
    assert 300 in calls  # one evaluate_completion call on the guard set


def test_the_cache_key_carries_the_guard_fingerprint(tmp_path, monkeypatch):
    from evaluation.completion import examples_fingerprint
    from evaluation.eval_harness.line_guard import load_guard_set

    mod, state, args, work, _, calls = _score_with_fake_model(tmp_path, monkeypatch)
    fp = examples_fingerprint(load_guard_set())[:12]
    cached = list((work / "cache").glob("*.json"))
    assert len(cached) == 1 and fp in cached[0].name
    # a HumanEval-era cache entry (no fingerprint in its name) is never read
    old = work / "cache" / f"{'b' * 64}-alpha0.5-web_client-flask.json"
    old.write_text(json.dumps({"pass_at_1": 0.4}), encoding="utf-8")
    n = len(calls)
    again = mod.score_version(state, None, "cluster-web", _v(2, "b" * 64), args, work)
    assert len(calls) == n and "guard" in again  # served from the new cache, not the old file


def test_the_guard_benchmark_name_is_the_one_evaluation_froze():
    from evaluation.eval_harness import line_guard

    assert evidence.GUARD_BENCHMARK == line_guard.GUARD_BENCHMARK == "LineCompletion"


def test_without_live_guard_evidence_the_fallback_has_no_guard_and_says_why(tmp_path, monkeypatch):
    """The round-manifest fallback never yields a HumanEval guard (it used to read anchors)."""
    monkeypatch.delenv("CLASP_EVAL_EVIDENCE", raising=False)
    ip = {"edit_similarity": 0.5, "exact_match": 0.2, "perplexity": 3.0, "n_examples": 60}
    doc = {"round": 1, "finished_utc": "2026-01-01T00:00:00", "clusters": {"web": {
        "representative_client": "client-flask",
        "seam_a_b": {"registry_name": "cluster-web", "versions": {}},
        "evaluation": {"candidate": {"in_project": ip}, "baseline": {"in_project": ip}}}}}
    monkeypatch.setattr(evidence, "_newest_round_manifest",
                        lambda cluster_id: (tmp_path / "round1_manifest.json", doc))
    ev = evidence.evidence_for("web")
    assert ev["guard"] == [] and ev["baseline_guard"] == []
    note = ev["provenance"]["guard"]
    assert "line-completion guard was not scored" in note and "HumanEval" not in note
    assert "failed" in note  # D5 counts a missing guard as failed
    assert not hasattr(evidence, "guard_anchors")


def test_a_humaneval_era_evidence_file_counts_as_not_evaluated(tmp_path, monkeypatch):
    """A live-eval file written before the line guard is neither read nor a crash: re-evaluate."""
    monkeypatch.delenv("CLASP_EVAL_EVIDENCE", raising=False)
    monkeypatch.setattr(evidence, "LIVE_EVAL_DIR", tmp_path)
    ip = {"edit_similarity": 0.52, "exact_match": 0.2, "perplexity": 3.1, "n_examples": 60}
    old = {"candidate_version": 2, "candidate_sha256": "b" * 64, "baseline_version": 1,
           "clusters": {"web": {"in_project": ip, "baseline_in_project": ip, "noise_band": 0.0,
                                "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.45}}],
                                "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.5}}]}},
           "details": {"candidate": {"humaneval_n": 20, "humaneval_solved": 9}, "baseline": {}}}
    (tmp_path / "web-v2.json").write_text(json.dumps(old), encoding="utf-8")
    assert evidence.evaluated_version("web", 2, "b" * 64) is False
    with pytest.raises(evidence.EvidenceMissing, match="Evaluate"):
        evidence.evidence_for("web", version=2, sha256="b" * 64)
    # the same file with a line guard but no fingerprint is not trusted either
    entry = old["clusters"]["web"]
    entry["guard"] = entry["baseline_guard"] = [SCORES["candidate"]["guard"]]
    (tmp_path / "web-v2.json").write_text(json.dumps(old), encoding="utf-8")
    assert evidence.evaluated_version("web", 2, "b" * 64) is False
