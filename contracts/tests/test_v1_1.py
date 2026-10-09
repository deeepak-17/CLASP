"""Contracts v1.1 — additive changes over the v1.0 freeze.

1. Composite adapters (D6): ``AdapterKind.COMPOSITE`` + ``CompositeProvenance``
   carried on ``AdapterMetadata.composed_from``.
2. ``GuardMetrics.pass_at_k`` survives JSON: integer keys come back as integers.
3. Every wire seam round-trips through ``to_json`` / ``from_json`` and through a
   real ``json.dumps`` -> ``json.loads`` cycle.
4. v1.0 payloads (no ``composed_from``, string pass@k keys) still deserialize.
"""
import json

import pytest

import contracts as c


def _wire(obj):
    """Serialize exactly the way an HTTP hop does."""
    return json.loads(json.dumps(obj.to_json()))


# --------------------------------------------------------------------------- #
# composite adapters (D6)
# --------------------------------------------------------------------------- #
def test_composite_kind_exists_and_old_kinds_unchanged():
    assert c.AdapterKind.COMPOSITE.value == "composite"
    assert c.AdapterKind("client") is c.AdapterKind.CLIENT
    assert c.AdapterKind("cluster") is c.AdapterKind.CLUSTER


def test_composite_provenance_fields():
    prov = c.CompositeProvenance(
        cluster_name="cluster-web", cluster_version=3,
        client_name="flask", client_version=2, alpha=0.5, beta=1.0,
    )
    assert prov.base_model  # defaulted, recorded for reproducibility
    assert _wire(prov) == prov.to_json()
    assert c.CompositeProvenance.from_json(_wire(prov)) == prov


def test_metadata_composed_from_defaults_to_none():
    meta = c.AdapterMetadata(
        ref=c.AdapterRef("flask", 1, c.AdapterKind.CLIENT),
        hparams=c.LoRAHyperParams(), privacy=c.PrivacySpec(),
        aggregation=None, round=1, seed=0, sha256="0" * 64, num_bytes=10,
    )
    assert meta.composed_from is None


def test_metadata_round_trips_composite_provenance():
    prov = c.CompositeProvenance("cluster-web", 3, "flask", 2, alpha=0.5, beta=1.0)
    meta = c.AdapterMetadata(
        ref=c.AdapterRef("composite-flask", 1, c.AdapterKind.COMPOSITE, cluster_id="web"),
        hparams=c.LoRAHyperParams(rank=32, alpha=0.5), privacy=c.PrivacySpec(epsilon=4.0),
        aggregation=None, round=1, seed=7, sha256="a" * 64, num_bytes=99,
        source_clients=("flask",), composed_from=prov,
    )
    back = c.AdapterMetadata.from_json(_wire(meta))
    assert back == meta
    assert back.composed_from.cluster_version == 3


def test_v1_0_metadata_dict_still_loads():
    """A metadata.json written by a v1.0 registry has no composed_from key."""
    old = {
        "ref": {"name": "flask", "version": 1, "kind": "client", "cluster_id": None},
        "hparams": {"rank": 16, "lora_alpha": 32, "dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"], "alpha": 1.0, "beta": 1.0},
        "privacy": {"epsilon": None, "delta": 1e-5, "noise_multiplier": None,
                    "max_grad_norm": None},
        "aggregation": None, "round": None, "seed": 0, "sha256": "f" * 64,
        "num_bytes": 1, "source_clients": [], "created_at": "2026-08-01T00:00:00+00:00",
        "contracts_version": "1.0.0",
    }
    meta = c.AdapterMetadata.from_json(old)
    assert meta.composed_from is None
    assert meta.contracts_version == "1.0.0"
    assert meta.hparams.target_modules == ("q_proj", "v_proj")


# --------------------------------------------------------------------------- #
# pass@k keys (the D5 guard reads pass@1 after a wire hop)
# --------------------------------------------------------------------------- #
def test_guard_pass_at_k_int_keys_survive_json():
    g = c.GuardMetrics("HumanEval", {1: 0.31, 10: 0.52})
    back = c.GuardMetrics.from_json(_wire(g))
    assert back.pass_at_k[1] == 0.31  # was a KeyError on v1.0 after json.loads
    assert back == g


def test_guard_normalizes_string_keys_on_construction():
    g = c.GuardMetrics("HumanEval", {"1": 0.4})
    assert g.pass_at_k == {1: 0.4}
    assert g.pass_at(1) == 0.4
    assert g.pass_at(10) is None


@pytest.mark.parametrize("bad", [{"one": 0.1}, {1: 1.5}, {1: -0.1}, {0: 0.2}])
def test_guard_rejects_malformed_pass_at_k(bad):
    with pytest.raises(ValueError):
        c.GuardMetrics("HumanEval", bad)


# --------------------------------------------------------------------------- #
# the three wire seams round-trip
# --------------------------------------------------------------------------- #
def test_adapter_upload_round_trip():
    up = c.AdapterUpload(
        client_id="flask", cluster_id="web", kind=c.AdapterKind.CLIENT,
        hparams=c.LoRAHyperParams(), num_train_samples=1200, round=2,
        privacy=c.PrivacySpec(epsilon=3.2, noise_multiplier=1.1, max_grad_norm=1.0), seed=5,
    )
    assert c.AdapterUpload.from_json(_wire(up)) == up


def test_cluster_snapshot_round_trip():
    snap = c.ClusterSnapshot(
        cluster_id="web", round=3, kind=c.AdapterKind.CLUSTER, hparams=c.LoRAHyperParams(),
        aggregation=c.AggregationMethod.SVD_EXACT,
        participating_clients=("django", "flask", "requests"),
    )
    assert c.ClusterSnapshot.from_json(_wire(snap)) == snap


def test_eval_result_round_trip_keeps_pass_at_1():
    ev = c.EvalResult(
        adapter=c.AdapterRef("flask", 2, c.AdapterKind.CLIENT, cluster_id="web"),
        in_project=c.InProjectMetrics(0.71, 0.34, 2.9, 40),
        guard=(c.GuardMetrics("HumanEval", {1: 0.31}), c.GuardMetrics("MBPP", {1: 0.4})),
        baseline_in_project=c.InProjectMetrics(0.69, 0.30, 3.0, 40),
        baseline_noise_band=0.01, seed=3,
    )
    back = c.EvalResult.from_json(_wire(ev))
    assert back == ev
    assert back.guard[0].pass_at_k[1] == 0.31


def test_promotion_decision_and_run_manifest_round_trip():
    dec = c.PromotionDecision(
        adapter=c.AdapterRef("flask", 2, c.AdapterKind.CLIENT),
        action=c.PromotionAction.ROLLBACK, active_version_after=1, reason="guard",
    )
    assert c.PromotionDecision.from_json(_wire(dec)) == dec
    man = c.RunManifest(
        run_id="r1", seed=1, config_hash="h", adapter_versions={"flask": 2},
        gpu_hours_estimate=0.5,
    )
    assert c.RunManifest.from_json(_wire(man)) == man
