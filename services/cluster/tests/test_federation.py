"""Weeks 7-13 (P2): multi-cluster rounds, isolation, straggler/dropout handling,
dynamic re-clustering and static fallback — through the real FedProx clients
and the real aggregation/redistribution code (needs torch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from cluster.adapter_format import random_adapter
from cluster.clustering import ClusterAssigner
from cluster.federation import MultiClusterFederation
from cluster.simulation import make_clustered_clients
from cluster.straggler import StragglerPolicy

WEB, SCI = "cluster-web", "cluster-sci"
DIM = 32


def _build(seed=7, mode="dynamic", misassign=(), policy=None, send=None, delays=None, **assigner_kw):
    clients, truth = make_clustered_clients(seed=seed, dim=DIM)
    warm = dict(truth)
    for cid in misassign:
        warm[cid] = WEB if truth[cid] == SCI else SCI
    for cid, delay in (delays or {}).items():
        clients[cid].simulated_delay_s = delay
    assigner = ClusterAssigner(warm, [WEB, SCI], mode=mode, seed=0, **assigner_kw)
    fed = MultiClusterFederation(
        clients, assigner, random_adapter(DIM, DIM, seed=0), policy=policy, send=send
    )
    return fed, truth


def _cluster_ids(prefix):
    return {f"{prefix}/client-{i}" for i in range(3)}


def test_two_clusters_three_clients_each_complete_a_round_with_correct_shapes():
    fed, _ = _build(mode="static")
    rnd = fed.run_round()
    assert set(rnd.clusters) == {WEB, SCI}
    for cid, res in rnd.clusters.items():
        assert res.aggregated and len(res.contributors) == 3
        assert set(res.members) == _cluster_ids(cid)
        ad = res.adapter
        assert ad.rank == 16 and ad.num_layers == 1
        for m in ad.target_modules:
            assert ad.modules[0][m]["lora_A"].shape == (16, DIM)  # (r, in)
            assert ad.modules[0][m]["lora_B"].shape == (DIM, 16)  # (out, r)
            assert ad.delta_w(m).shape == (DIM, DIM)
        assert res.metrics.num_clients == 3 and res.metrics.duration_s >= 0
        assert res.metrics.mean_loss is not None
        assert res.delivery.all_delivered and set(res.delivery.delivered) == _cluster_ids(cid)
    assert fed.isolation_violations() == []


def test_cluster_adapters_diverge_and_each_client_receives_only_its_own_cluster():
    fed, truth = _build(mode="static")
    fed.run(2)
    web, sci = fed.cluster_adapters[WEB], fed.cluster_adapters[SCI]
    assert not np.allclose(web.delta_w("q_proj"), sci.delta_w("q_proj"), atol=1e-4)
    for cid, (cluster_id, _round) in fed.received.items():
        assert cluster_id == truth[cid]
        held = fed.client_adapters[cid]
        np.testing.assert_allclose(
            held.delta_w("q_proj"), fed.cluster_adapters[truth[cid]].delta_w("q_proj"), atol=1e-6
        )
    assert fed.isolation_violations() == []


def test_aggregate_contains_only_own_clusters_contributions():
    """Recompute cluster-web's adapter from web clients' returned updates alone."""
    from cluster.aggregation import aggregate_svd

    fed, _ = _build(mode="static")
    rnd = fed.run_round()
    for cid in (WEB, SCI):
        ups = [rnd.updates[c] for c in rnd.clusters[cid].contributors]
        # all clients have the same num_examples (64) => equal weights
        expected = aggregate_svd(iter(ups), [64.0] * len(ups), rank=16)
        np.testing.assert_allclose(
            rnd.clusters[cid].adapter.delta_w("o_proj"), expected.delta_w("o_proj"), atol=1e-6
        )


def test_isolation_check_detects_an_adversarial_delivery():
    fed, _ = _build(mode="static")
    fed.run_round()
    assert fed.isolation_violations() == []
    web_client = min(_cluster_ids("cluster-web"))
    fed.received[web_client] = (SCI, 0)  # inject: a web client was handed sci's adapter
    problems = fed.isolation_violations()
    assert len(problems) == 1 and web_client in problems[0]


def test_unassigned_client_rejected_up_front():
    clients, truth = make_clustered_clients(seed=1, dim=DIM)
    partial = {k: v for k, v in truth.items() if k != "cluster-web/client-0"}
    with pytest.raises(ValueError, match="without a cluster assignment"):
        MultiClusterFederation(
            clients, ClusterAssigner(partial, [WEB, SCI]), random_adapter(DIM, DIM, seed=0)
        )


# ------------------------------------------------------------- stragglers


def test_straggler_is_skipped_and_excluded_from_the_aggregate():
    slow = "cluster-web/client-1"
    fed, _ = _build(mode="static", policy=StragglerPolicy(timeout_s=5.0), delays={slow: 60.0})
    rnd = fed.run_round()
    res = rnd.clusters[WEB]
    assert res.skipped == {slow: "timeout"} and slow not in res.contributors
    assert len(res.contributors) == 2 and res.aggregated
    assert rnd.clusters[SCI].skipped == {}  # the other cluster is unaffected


def test_quorum_failure_keeps_previous_cluster_adapter_and_other_cluster_proceeds():
    delays = {f"cluster-web/client-{i}": 60.0 for i in (0, 1)}
    fed, _ = _build(
        mode="static", policy=StragglerPolicy(timeout_s=5.0, min_fraction=0.5), delays=delays
    )
    before = fed.cluster_adapters[WEB].delta_w("q_proj").copy()
    rnd = fed.run_round()
    assert not rnd.clusters[WEB].aggregated and rnd.clusters[SCI].aggregated
    np.testing.assert_array_equal(fed.cluster_adapters[WEB].delta_w("q_proj"), before)
    assert rnd.clusters[WEB].metrics.num_clients == 0


def test_offline_clients_are_dropouts_not_failures():
    fed, _ = _build(mode="static")
    rnd = fed.run_round(offline=["cluster-sci/client-0"])
    sci = rnd.clusters[SCI]
    assert sci.skipped == {"cluster-sci/client-0": "offline"} and len(sci.contributors) == 2
    assert fed.run_round().clusters[SCI].skipped == {}  # back online next round


# ---------------------------------------------------- redistribution retry


def test_flaky_delivery_is_retried_and_succeeds():
    attempts: dict[str, int] = {}

    def flaky(client_id, payload):
        attempts[client_id] = attempts.get(client_id, 0) + 1
        if client_id == "cluster-sci/client-2" and attempts[client_id] < 3:
            raise ConnectionError("blip")

    fed, _ = _build(mode="static", send=flaky)
    rnd = fed.run_round()
    assert rnd.clusters[SCI].delivery.all_delivered
    assert rnd.clusters[SCI].delivery.attempts["cluster-sci/client-2"] == 3
    assert fed.received["cluster-sci/client-2"][0] == SCI


def test_undeliverable_client_is_reported_and_trains_from_its_stale_adapter_only():
    def broken(client_id, payload):
        if client_id == "cluster-web/client-0":
            raise ConnectionError("down")

    fed, _ = _build(mode="static", send=broken)
    rnd = fed.run_round()
    rep = rnd.clusters[WEB].delivery
    assert not rep.all_delivered and "cluster-web/client-0" in rep.failed
    assert "cluster-web/client-0" not in fed.received  # never got the broadcast
    assert fed.isolation_violations() == []


# --------------------------------------------- dynamic re-clustering (D4)


def test_no_recluster_before_round_two_then_dynamic_fixes_a_wrong_warm_start():
    wrong = "cluster-sci/client-2"
    fed, truth = _build(misassign=[wrong])
    r1 = fed.run_round()
    assert r1.recluster is None and r1.assignment_after[wrong] == WEB  # still the warm start
    r2 = fed.run_round()  # 2 completed rounds => re-cluster
    rec = r2.recluster
    assert rec.method == "dynamic", rec.reason
    assert rec.moved == {wrong: (WEB, SCI)}
    assert r2.assignment_after == truth
    assert rec.comparison["agreement"] == pytest.approx(5 / 6)
    assert rec.comparison["dynamic_cohesion"] > rec.comparison["warm_start_cohesion"]
    # after the move the client contributes to (and only to) its new cluster
    r3 = fed.run_round()
    assert wrong in r3.clusters[SCI].contributors and wrong not in r3.clusters[WEB].contributors
    assert fed.received[wrong][0] == SCI
    assert fed.isolation_violations() == []


def test_correct_warm_start_is_left_unchanged_by_dynamic_clustering():
    fed, truth = _build()
    fed.run(2)
    rec = fed.rounds[1].recluster
    assert rec.method == "dynamic" and rec.moved == {}
    assert fed.assigner.assignment == truth


def test_static_mode_keeps_wrong_assignment_forever():
    wrong = "cluster-sci/client-2"
    fed, _ = _build(mode="static", misassign=[wrong])
    fed.run(4)
    assert all(r.recluster is None for r in fed.rounds)
    assert fed.assigner.assignment[wrong] == WEB


def test_fallback_to_static_when_not_enough_clients_survive_the_round():
    wrong = "cluster-sci/client-2"
    fed, _ = _build(misassign=[wrong])
    fed.run_round()
    survivors_offline = [c for c in fed.clients if c not in ("cluster-web/client-0",)]
    rnd = fed.run_round(offline=survivors_offline)  # only 1 client answers; k=2 needs >=2
    assert rnd.recluster.method == "static_fallback"
    assert fed.assigner.assignment[wrong] == WEB  # unchanged, no guessing


def test_recluster_is_robust_to_client_dropout_in_the_recluster_round():
    wrong = "cluster-sci/client-2"
    fed, _ = _build(misassign=[wrong])
    fed.run_round()
    rnd = fed.run_round(offline=["cluster-web/client-0", "cluster-sci/client-0"])
    rec = rnd.recluster
    assert rec.method == "dynamic", rec.reason
    assert fed.assigner.assignment[wrong] == SCI
    assert fed.assigner.assignment["cluster-web/client-0"] == WEB  # dropout kept its cluster
    assert fed.assigner.assignment["cluster-sci/client-0"] == SCI
    assert fed.isolation_violations() == []


def test_moved_client_with_failed_delivery_does_not_train_from_the_wrong_cluster():
    wrong = "cluster-sci/client-2"

    def fail_for_wrong(client_id, payload):
        if client_id == wrong and payload.round_id >= 1:
            raise ConnectionError("down after move")

    fed, _ = _build(misassign=[wrong], send=fail_for_wrong)
    fed.run(2)  # round-2 redistribution to `wrong` (now a sci member) fails
    assert fed.assigner.assignment[wrong] == SCI and fed.received[wrong][0] == WEB
    r3 = fed.run_round()
    assert r3.clusters[SCI].skipped[wrong].startswith("stale_adapter")
    assert wrong not in r3.clusters[SCI].contributors and wrong not in r3.clusters[WEB].contributors


# ------------------------------------------------------ reproducibility (G3)


def _fingerprint(fed):
    return {
        cid: np.concatenate([fed.cluster_adapters[cid].delta_w(m).ravel() for m in ("q_proj", "o_proj")])
        for cid in (WEB, SCI)
    }


def test_two_seeded_runs_are_bit_identical_including_reassignments():
    runs = []
    for _ in range(2):
        fed, _ = _build(misassign=["cluster-sci/client-2"])
        fed.run(4)
        runs.append((_fingerprint(fed), dict(fed.assigner.assignment),
                     [r.recluster.method if r.recluster else None for r in fed.rounds]))
    (fp_a, asg_a, rec_a), (fp_b, asg_b, rec_b) = runs
    for cid in (WEB, SCI):
        np.testing.assert_array_equal(fp_a[cid], fp_b[cid])
    assert asg_a == asg_b and rec_a == rec_b


def test_loss_decreases_over_rounds_in_every_cluster():
    fed, _ = _build(mode="static")
    rounds = fed.run(4)
    for cid in (WEB, SCI):
        losses = [r.clusters[cid].metrics.mean_loss for r in rounds]
        assert losses[-1] < losses[0], losses


def test_diverged_client_with_nan_update_is_skipped_not_fatal():
    import numpy as np

    class Diverged:
        def fit(self, parameters, config):
            return [np.full_like(p, np.nan) for p in parameters], 64, {"client_id": "bad"}

    fed, _ = _build(mode="static")
    fed.clients["cluster-web/client-0"] = Diverged()
    rnd = fed.run_round()
    res = rnd.clusters[WEB]
    assert "non-finite" in res.skipped["cluster-web/client-0"]
    assert res.aggregated and len(res.contributors) == 2
    assert np.isfinite(res.adapter.delta_w("q_proj")).all()
