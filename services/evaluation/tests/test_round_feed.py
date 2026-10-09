"""Tests for eval_harness/round_feed.py — demo_round manifests -> the dashboard's rounds.json."""

from __future__ import annotations

import json
import struct

import pytest

from evaluation.eval_harness.round_feed import (
    GUARD_PASS_AT_1_TOLERANCE,
    build_feed,
    d5_alerts,
    summarize_round,
)
from evaluation.utils.errors import EvaluationError


def _ip(edit_similarity: float) -> dict:
    return {"edit_similarity": edit_similarity, "exact_match": 0.1, "perplexity": 2.6, "n_examples": 60}


def _guard(candidate: float | None = None, baseline: float | None = None) -> dict:
    """``edge.promote.GuardStatus.as_dict()`` shape."""
    available = candidate is not None and baseline is not None
    return {
        "available": available,
        "reason": "scored" if available else "no HumanEval anchor supplied",
        "source": None,
        "benchmark": "HumanEval",
        "tolerance_pass_at_1": 0.02,
        "candidate": {"benchmark": "HumanEval", "pass_at_k": {"1": candidate}} if available else None,
        "baseline": {"benchmark": "HumanEval", "pass_at_k": {"1": baseline}} if available else None,
    }


def _cluster(cluster_id: str, base: float | None, cand: float | None, action: str, *, authoritative=False) -> dict:
    """One ``manifest['clusters'][cluster_id]`` block, keys as scripts/demo_round.py writes them."""
    name = f"cluster-{cluster_id}"

    def version(v: int, method: str) -> dict:
        return {
            "aggregation": method,
            "seconds": 6.0,
            "aggregation_manifest": {},
            "registry_version": {"ref": {"name": name, "version": v, "kind": "cluster", "cluster_id": cluster_id},
                                 "sha256": f"{v:064x}", "aggregation": "svd_exact"},
        }

    def evaluation(value):
        return None if value is None else {"in_project": _ip(value), "examples_sha256": "f" * 64, "rows": []}

    return {
        "seam_a_b": {"registry_name": name, "cluster_round_id": 1, "uploads": [],
                     "versions": {"baseline": version(1, "naive"), "candidate": version(2, "svd")}},
        "seam_c1": {},
        "composites": {},
        "evaluation": {"baseline": evaluation(base), "candidate": evaluation(cand)},
        "representative_client": "client-werkzeug",
        "examples_scored": 60,
        "seam_c2": {
            "decision": {"adapter": {}, "action": action, "active_version_after": 2 if action == "promote" else 1,
                         "reason": f"{action}: ...", "timestamp": "t"},
            "decision_is_authoritative": authoritative,
        },
    }


def _manifest(round_no: int, clusters: dict, guard: dict | None = None, noise_band: float = 0.0) -> dict:
    return {
        "round": round_no,
        "finished_utc": f"2026-09-0{round_no}T10:00:00+00:00",
        "wall_minutes": 14.33,
        "nfr_round_minutes": 30,
        "nfr_met": True,
        "config": {"noise_band": noise_band},
        "counters": {"uploads": 6, "aggregations": 4, "snapshots": 4, "composites": 4, "decisions": 2},
        "humaneval_guard": guard or _guard(),
        "clusters": clusters,
    }


#: The recorded live round (experiments/w12-integration/RESULTS.md): guard unavailable, both ROLLBACK.
LIVE_ROUND = _manifest(
    1,
    {"web": _cluster("web", 0.5337, 0.5470, "rollback"), "scientific": _cluster("scientific", 0.4837, 0.4861, "rollback")},
)


class TestSummarizeRound:
    def test_reads_versions_metrics_and_decision(self) -> None:
        entry = summarize_round(LIVE_ROUND)
        web = next(c for c in entry["clusters"] if c["cluster_id"] == "web")
        assert (web["baseline"]["version"], web["candidate"]["version"]) == (1, 2)
        assert (web["baseline"]["aggregation"], web["candidate"]["aggregation"]) == ("naive", "svd")
        assert web["edit_similarity_delta"] == pytest.approx(0.0133)
        assert web["decision"]["action"] == "rollback" and not web["authoritative"]
        assert entry["guard"] == {"available": False, "reason": "no HumanEval anchor supplied",
                                  "candidate_pass_at_1": None, "baseline_pass_at_1": None}

    def test_unmeasured_metric_stays_null(self) -> None:
        entry = summarize_round(_manifest(1, {"web": _cluster("web", None, None, "rollback")}))
        web = entry["clusters"][0]
        assert web["candidate"]["in_project"] is None and web["edit_similarity_delta"] is None

    def test_n_examples_zero_placeholder_is_not_a_measurement(self) -> None:
        cluster = _cluster("web", 0.5, 0.5, "rollback")
        cluster["evaluation"]["candidate"]["in_project"]["n_examples"] = 0
        assert summarize_round(_manifest(1, {"web": cluster}))["clusters"][0]["candidate"]["in_project"] is None

    def test_missing_key_names_the_path(self) -> None:
        broken = _manifest(1, {"web": _cluster("web", 0.5, 0.6, "promote")})
        del broken["clusters"]["web"]["seam_c2"]["decision"]
        with pytest.raises(EvaluationError, match="seam_c2.decision"):
            summarize_round(broken)


class TestD5Alerts:
    def test_live_round_alerts_agree_with_the_registry(self) -> None:
        alerts = d5_alerts(summarize_round(LIVE_ROUND))
        assert {a["kind"] for a in alerts} == {"guard_unavailable"}
        assert len(alerts) == 2  # one per cluster, no decision_mismatch

    def test_promote_when_both_halves_hold(self) -> None:
        m = _manifest(1, {"web": _cluster("web", 0.50, 0.55, "promote", authoritative=True)}, _guard(0.30, 0.30))
        assert d5_alerts(summarize_round(m)) == []

    def test_guard_regression(self) -> None:
        m = _manifest(1, {"web": _cluster("web", 0.50, 0.55, "rollback")}, _guard(0.25, 0.30))
        assert [a["kind"] for a in d5_alerts(summarize_round(m))] == ["guard_regression"]

    def test_tolerance_boundary_matches_the_rule(self) -> None:
        # drop == tolerance is allowed by registry.promotion (drop <= tolerance).
        m = _manifest(1, {"web": _cluster("web", 0.50, 0.55, "promote")}, _guard(0.28, 0.30))
        assert d5_alerts(summarize_round(m)) == []

    def test_not_beyond_noise_band(self) -> None:
        m = _manifest(1, {"web": _cluster("web", 0.50, 0.505, "rollback")}, _guard(0.3, 0.3), noise_band=0.01)
        assert [a["kind"] for a in d5_alerts(summarize_round(m))] == ["in_project_not_improved"]

    def test_disagreement_with_the_registry_is_flagged(self) -> None:
        m = _manifest(1, {"web": _cluster("web", 0.50, 0.55, "rollback")}, _guard(0.3, 0.3))
        assert [a["kind"] for a in d5_alerts(summarize_round(m))] == ["decision_mismatch"]

    def test_tolerance_mirrors_the_registry(self) -> None:
        promotion = pytest.importorskip("registry.promotion")
        assert GUARD_PASS_AT_1_TOLERANCE == promotion.GUARD_PASS_AT_1_TOLERANCE


class TestBuildFeed:
    def test_rounds_sorted_and_round_over_round_drop(self) -> None:
        r2 = _manifest(2, {"web": _cluster("web", 0.54, 0.52, "rollback")})
        feed = build_feed([r2, LIVE_ROUND])
        assert [r["round"] for r in feed["rounds"]] == [1, 2]
        drops = [a for a in feed["alerts"] if a["kind"] == "round_over_round_drop"]
        assert drops == [{"round": 2, "cluster_id": "web", "kind": "round_over_round_drop",
                          "message": "candidate edit_similarity 0.5470 -> 0.5200"}]
        assert feed["feed_version"] == "1.0.0"
        json.dumps(feed)  # serialisable

    def test_duplicate_round_refused(self) -> None:
        with pytest.raises(EvaluationError, match="twice"):
            build_feed([LIVE_ROUND, LIVE_ROUND])


def _blob(value: float) -> bytes:
    data = struct.pack("<4f", *([value] * 4))
    header = json.dumps({"lora_A": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


def test_feed_reads_what_the_real_producers_write(tmp_path, monkeypatch) -> None:
    """seam_c2 / humaneval_guard built by edge.promote against the real registry app."""
    promote = pytest.importorskip("edge.promote")
    registry_client = pytest.importorskip("edge.registry_client")
    appmod = pytest.importorskip("registry.app")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    appmod._store = None
    with TestClient(appmod.app) as client:
        published = {}
        for label, value in (("baseline", 0.1), ("candidate", 0.2)):
            resp = client.post("/adapters/cluster-web/versions",
                               files={"file": ("a.safetensors", _blob(value), "application/octet-stream")},
                               data={"meta": json.dumps({"kind": "cluster", "cluster_id": "web", "round": 1})})
            published[label] = resp.json()
        guard = promote.GuardStatus(available=False, reason="no anchor")
        seam_c2 = promote.promote_candidate(
            registry_client.RegistryClient(base_url="", session=client), "cluster-web",
            version=2, kind="cluster", cluster_id="web",
            in_project=_ip(0.547), baseline_in_project=_ip(0.5337), guard=guard)
    appmod._store = None

    cluster = _cluster("web", 0.5337, 0.547, "rollback")
    cluster["seam_c2"] = seam_c2
    for label in ("baseline", "candidate"):
        cluster["seam_a_b"]["versions"][label]["registry_version"] = published[label]
    feed = build_feed([_manifest(1, {"web": cluster}, guard.as_dict())])
    web = feed["rounds"][0]["clusters"][0]
    assert web["decision"]["action"] == "rollback" and web["decision"]["active_version_after"] == 1
    assert web["candidate"]["sha256"] == published["candidate"]["sha256"]
    assert [a["kind"] for a in feed["alerts"]] == ["guard_unavailable"]
