"""Tests for eval_harness/personalization.py — edge round manifests -> personalization.json."""

from __future__ import annotations

import copy

import pytest

from evaluation.eval_harness.personalization import build_feed, compare_rounds, summarize_round
from evaluation.utils.errors import EvaluationError
from evaluation.utils.io_utils import read_json
from evaluation.utils.paths import project_paths


def _client(cluster: str, base: float, client_only: float, composite: float, best_alpha: float) -> dict:
    """One ``manifest['clients'][id]`` block, keys as edge.round writes them."""
    return {
        "cluster": cluster,
        "n_train_files": 10,
        "n_held_out_blocks": 3,
        "best_alpha": best_alpha,
        "beta": 1.0,
        "full_split": {
            "base_ppl": base,
            "client_only_ppl": client_only,
            "composite_ppl": composite,
            "alpha_ref": 0.5,
            "alpha_ref_ppl": composite,
            "n_tokens": 1000,
        },
        "personalization_delta_ppl": round(composite - base, 3),
        "cluster_contribution_at_alpha_ref": round(composite - client_only, 3),
        "promotion": {"decision": "PROVISIONAL_PROMOTE", "decision_is_authoritative": False},
    }


def _manifest(round_no: int, clients: dict) -> dict:
    return {"round": round_no, "utc": "2026-10-01T00:00:00Z", "model_id": "m", "seed": 0,
            "alpha_grid": [0.0, 0.5], "clients": clients, "timings": {"total_minutes": 10.0}}


ROUND1 = _manifest(1, {
    "web/client-a": _client("web", 3.0, 2.8, 2.8, 0.0),
    "web/client-b": _client("web", 2.0, 1.9, 1.9, 0.0),
})
ROUND2 = _manifest(2, {
    "web/client-a": _client("web", 3.0, 2.85, 2.75, 0.5),
    "web/client-b": _client("web", 2.0, 1.95, 1.88, 0.5),
})


def test_summary_counts_improvement_and_cluster_effect() -> None:
    entry = summarize_round(ROUND2)
    assert entry["summary"]["n_improved"] == 2
    assert entry["summary"]["n_cluster_helps"] == 2
    assert entry["summary"]["n_alpha_zero"] == 0
    assert entry["summary"]["mean_delta_ppl"] == pytest.approx((-0.25 - 0.12) / 2)


def test_compare_rounds_reports_composite_change_per_client() -> None:
    comp = compare_rounds(summarize_round(ROUND1), summarize_round(ROUND2))
    by_client = {r["client_id"]: r for r in comp["clients"]}
    assert by_client["web/client-a"]["composite_ppl_change"] == pytest.approx(-0.05)
    assert comp["n_composite_better"] == 2


def test_feed_orders_rounds_and_rejects_duplicates() -> None:
    feed = build_feed([ROUND2, ROUND1])
    assert [r["round"] for r in feed["rounds"]] == [1, 2]
    assert len(feed["comparisons"]) == 1
    with pytest.raises(EvaluationError, match="more than once"):
        build_feed([ROUND1, ROUND1])


def test_missing_key_names_its_path() -> None:
    broken = copy.deepcopy(ROUND1)
    del broken["clients"]["web/client-a"]["full_split"]["base_ppl"]
    with pytest.raises(EvaluationError, match="full_split.base_ppl"):
        summarize_round(broken)


def test_real_rounds_reproduce_the_published_finding() -> None:
    """The vendored edge manifests: cluster layer harmful in round 1, helpful in round 2."""
    root = project_paths().eval_results / "edge_rounds"
    feed = build_feed([read_json(root / "round1_manifest.json"), read_json(root / "round2_d3_manifest.json")])
    r1, r2 = feed["rounds"]
    assert r1["summary"]["n_improved"] == r2["summary"]["n_improved"] == 6
    assert r1["summary"]["n_alpha_zero"] == 6 and r1["summary"]["n_cluster_helps"] == 0
    assert r2["summary"]["n_alpha_zero"] == 0 and r2["summary"]["n_cluster_helps"] == 6
    assert feed["comparisons"][0]["n_composite_better"] == 6
