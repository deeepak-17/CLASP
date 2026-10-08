"""Contracts v1.0 freeze tests — guard the integration seam.

If any of these break, it means the frozen wire format changed: that needs a
semver bump + all-hands sign-off (MASTER_PLAN D9), not a silent edit.
"""
from dataclasses import fields

import contracts as c


def test_version_frozen_at_1_0():
    assert c.__version__ == "1.0.0"
    assert c.CONTRACTS_VERSION == "1.0.0"


def test_three_wire_seams_present():
    # Edge->Cluster, Cluster->Registry, Registry->Eval
    assert c.AdapterUpload and c.ClusterSnapshot and c.EvalResult


def test_extended_types_on_hparams():
    names = {f.name for f in fields(c.LoRAHyperParams)}
    assert {"rank", "target_modules", "alpha", "beta"} <= names


def test_privacy_and_repro_fields():
    assert {"epsilon", "delta"} <= {f.name for f in fields(c.PrivacySpec)}
    assert {"seed", "config_hash", "adapter_versions", "gpu_hours_estimate"} <= {
        f.name for f in fields(c.RunManifest)
    }


def test_adapter_upload_defaults():
    up = c.AdapterUpload(
        client_id="django",
        cluster_id="web",
        kind=c.AdapterKind.CLIENT,
        hparams=c.LoRAHyperParams(),
        num_train_samples=1200,
        round=1,
    )
    assert up.hparams.rank == 16
    assert up.contracts_version == "1.0.0"
    assert up.timestamp  # auto-stamped


def test_cluster_snapshot_carries_aggregation():
    snap = c.ClusterSnapshot(
        cluster_id="web",
        round=3,
        kind=c.AdapterKind.CLUSTER,
        hparams=c.LoRAHyperParams(),
        aggregation=c.AggregationMethod.SVD_EXACT,
        participating_clients=("django", "flask", "requests"),
    )
    assert snap.aggregation is c.AggregationMethod.SVD_EXACT
    assert len(snap.participating_clients) == 3


def test_eval_result_two_sided_shape():
    ev = c.EvalResult(
        adapter=c.AdapterRef("django", 2, c.AdapterKind.CLIENT),
        in_project=c.InProjectMetrics(0.71, 0.34, 2.9, 40),
        guard=(c.GuardMetrics("HumanEval", {1: 0.31}),),
        baseline_noise_band=0.01,
    )
    assert ev.in_project.edit_similarity == 0.71
    assert ev.guard[0].pass_at_k[1] == 0.31
