"""Smoke + invariants for ``python -m cluster.evidence`` (small, real run)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from cluster import evidence


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("evidence") / "ev.json"
    assert evidence.main(["--seeds", "1", "--rounds", "3", "--output", str(out)]) == 0
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# P2 Cluster")
    return json.loads(out.read_text())


def test_all_sections_are_present_and_nothing_is_hard_coded_nan(result):
    for key in ("config", "mu_sweep", "dynamic_vs_static", "reproducibility", "isolation_stress",
                "three_client_training", "straggler_dropout", "static_fallback", "summary"):
        assert key in result
    assert result["config"]["environment"]["python"]


def test_reproducible_and_isolated(result):
    assert result["summary"]["reproducible_in_all_seeds"] is True
    assert result["summary"]["isolation_violations_total"] == 0
    assert all(r["sha256_run1"] == r["sha256_run2"] for r in result["reproducibility"])


def test_mu_sweep_marks_the_unstable_end(result):
    by_mu = {r["mu"]: r for r in result["mu_sweep"]}
    assert by_mu[0.01]["stable_in_all_seeds"] and by_mu[10.0]["stable_in_all_seeds"]
    # lr * mu >= 2 is outside explicit-Euler stability for the proximal term
    assert not by_mu[50.0]["stable_in_all_seeds"] and by_mu[50.0]["lr_times_mu"] >= 2


def test_dynamic_beats_static_on_a_wrong_label_and_ties_on_a_right_one(result):
    wrong = result["summary"]["one_wrong_label"]
    assert wrong["dynamic"]["mean_ari_vs_true_groups"] > wrong["static"]["mean_ari_vs_true_groups"]
    right = result["summary"]["correct_warm_start"]
    assert right["dynamic"]["mean_ari_vs_true_groups"] == right["static"]["mean_ari_vs_true_groups"] == 1.0


def test_three_client_training_learns(result):
    for agg in ("svd", "naive"):
        rec = result["three_client_training"][agg]
        assert rec["loss_decreased"] and rec["contributors_per_round"] == [3, 3, 3]


def test_straggler_dropout_and_quorum_behaviour(result):
    sd = result["straggler_dropout"]
    for rnd in sd["timeout_straggler"]["rounds"]:
        assert rnd["cluster-web"]["skipped"] == {"cluster-web/client-2": "timeout"}
        assert rnd["cluster-web"]["contributors"] == 2
    rounds = sd["dropout_and_rejoin"]["rounds"]
    assert rounds[0]["cluster-sci"]["skipped"] == {"cluster-sci/client-1": "offline"}
    assert rounds[2]["cluster-sci"]["contributors"] == 3  # rejoined
    q = sd["quorum_failure"]
    assert q["web_aggregated"] is False and q["web_adapter_unchanged"] and q["sci_adapter_updated"]
    assert q["recovered_round"]["cluster-web"]["aggregated"] is True


def test_static_fallback_cases(result):
    fb = result["static_fallback"]
    assert fb["too_few_updates"]["method"] == "static_fallback"
    assert fb["too_few_updates"]["assignment_unchanged"]
    assert fb["no_group_structure"]["method"] == "static_fallback"
    assert "weak separation" in fb["no_group_structure"]["reason"]
    assert fb["no_group_structure"]["wrong_label_kept"]
    assert fb["static_mode"]["history_methods"] == ["warm_start"]


def test_markdown_report_is_rendered_from_the_record(tmp_path):
    from cluster.evidence import render_markdown

    rec = {
        "config": {"seeds": [1], "rounds": 3, "dim": 32, "rank": 16, "lr": 0.1,
                   "environment": {"python": "3", "torch": "t", "flwr": "f", "numpy": "n",
                                   "platform": "p"}},
        "elapsed_s": 1.0,
        "summary": {
            "correct_warm_start": {"seeds": 1, "dynamic": {"mean_ari_vs_true_groups": 1.0, "seeds_recovering_true_groups": 1, "mean_final_loss": 1.5}, "static": {"mean_ari_vs_true_groups": 1.0, "seeds_recovering_true_groups": 1, "mean_final_loss": 1.5}},
            "one_wrong_label": {"seeds": 1, "dynamic": {"mean_ari_vs_true_groups": 1.0, "seeds_recovering_true_groups": 1, "mean_final_loss": 1.5}, "static": {"mean_ari_vs_true_groups": 0.25, "seeds_recovering_true_groups": 0, "mean_final_loss": 1.7}},
            "reproducible_in_all_seeds": True, "isolation_violations_total": 0,
        },
        "mu_sweep": [{"mu": 1.0, "lr_times_mu": 0.1, "stable_in_all_seeds": True, "mean_final_loss_over_stable_seeds": 3.0},
                     {"mu": 50.0, "lr_times_mu": 5.0, "stable_in_all_seeds": False, "mean_final_loss_over_stable_seeds": None}],
        "three_client_training": {"seed": 1, "rounds": 2, "svd": {"round_mean_losses": [2.0, 1.0], "loss_decreased": True}, "naive": {"round_mean_losses": [2.0, 1.5], "loss_decreased": True}},
        "straggler_dropout": {"seed": 1,
            "timeout_straggler": {"rounds": [{"cluster-web": {"contributors": 2}}]},
            "dropout_and_rejoin": {"rounds": [{"cluster-sci": {"contributors": 2}}]},
            "quorum_failure": {"web_aggregated": False, "web_adapter_unchanged": True, "sci_adapter_updated": True, "recovered_round": {"cluster-web": {"aggregated": True}}}},
        "static_fallback": {"seed": 1, "too_few_updates": {"method": "static_fallback", "reason": "r"},
            "no_group_structure": {"method": "static_fallback", "reason": "weak", "wrong_label_kept": True},
            "static_mode": {"history_methods": ["warm_start"]}},
    }
    md = render_markdown(rec)
    assert "| one_wrong_label | static | 0.250 | 0/1 | 1.7000 |" in md
    assert "| 50.0 | 5 | False | diverged |" in md
    assert "Isolation violations found across all runs: **0**" in md

