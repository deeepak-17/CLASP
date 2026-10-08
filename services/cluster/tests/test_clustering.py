"""Week 10: delta_W flattening, cosine matrix, k-means, dynamic re-clustering, fallback."""

from __future__ import annotations

import numpy as np
import pytest

from cluster.clustering import (
    ClusterAssigner,
    adjusted_rand_index,
    align_labels,
    cohesion_and_separation,
    cosine_similarity_from_vectors,
    cosine_similarity_matrix,
    delta_gram,
    flatten_delta,
    flatten_update,
    kmeans_on_similarity,
    update_gram,
)
from tests.conftest import planted_directions, trained_adapter

WEB, SCI = "web", "sci"


# ---------------------------------------------------------------- flattening


def test_flatten_delta_shape_and_content():
    ad = trained_adapter(1, dim=8, rank=4, num_layers=3)
    vec = flatten_delta(ad)
    assert vec.ndim == 1 and vec.shape == (3 * 4 * 8 * 8,)  # layers * modules * out * in
    first = ad.delta_w("q_proj", 0).astype(np.float64).ravel()
    np.testing.assert_allclose(vec[: first.size], first)  # layer-major, module order
    last = ad.delta_w("o_proj", 2).astype(np.float64).ravel()
    np.testing.assert_allclose(vec[-last.size :], last)


def test_flatten_update_subtracts_start_and_checks_shape():
    a, s = trained_adapter(1, dim=8, rank=4), trained_adapter(2, dim=8, rank=4)
    np.testing.assert_allclose(flatten_update(a, s), flatten_delta(a) - flatten_delta(s))
    np.testing.assert_allclose(flatten_update(a), flatten_delta(a))
    with pytest.raises(ValueError):
        flatten_update(a, trained_adapter(3, dim=16, rank=4))


# ------------------------------------------------- factored Gram == flattened


@pytest.mark.parametrize("layers", [1, 3])
def test_factored_gram_equals_flattened_inner_products(layers):
    ads = [trained_adapter(s, dim=12, rank=4, num_layers=layers) for s in range(4)]
    vecs = np.stack([flatten_delta(a) for a in ads])
    np.testing.assert_allclose(delta_gram(ads), vecs @ vecs.T, rtol=1e-9, atol=1e-9)


def test_update_gram_with_shared_and_per_client_starts_matches_flattened_updates():
    ads = [trained_adapter(s, dim=12, rank=4, num_layers=2) for s in range(5)]
    start_a, start_b = trained_adapter(90, dim=12, rank=4, num_layers=2), trained_adapter(91, dim=12, rank=4, num_layers=2)
    starts = [start_a, start_a, start_b, None, start_b]  # shared, distinct and zero starts mixed
    vecs = np.stack([flatten_update(a, s) for a, s in zip(ads, starts)])
    np.testing.assert_allclose(update_gram(ads, starts), vecs @ vecs.T, rtol=1e-8, atol=1e-8)


def test_cosine_matrix_matches_reference_and_basic_properties():
    ads = [trained_adapter(s, dim=12, rank=4) for s in range(5)]
    starts = [trained_adapter(50, dim=12, rank=4)] * 5
    sim = cosine_similarity_matrix(ads, starts)
    ref = cosine_similarity_from_vectors(np.stack([flatten_update(a, s) for a, s in zip(ads, starts)]))
    np.testing.assert_allclose(sim, ref, atol=1e-9)
    assert sim.shape == (5, 5)
    np.testing.assert_allclose(sim, sim.T)
    np.testing.assert_allclose(np.diag(sim), 1.0)
    assert sim.min() >= -1.0 and sim.max() <= 1.0


def test_identical_updates_have_cosine_one_and_opposite_updates_minus_one():
    a = trained_adapter(1, dim=8, rank=4)
    neg = trained_adapter(1, dim=8, rank=4)
    for m in neg.target_modules:
        neg.modules[0][m]["lora_B"] = -neg.modules[0][m]["lora_B"]
    sim = cosine_similarity_matrix([a, a, neg])
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(-1.0)


def test_zero_update_client_is_orthogonal_to_everyone():
    a, b = trained_adapter(1, dim=8, rank=4), trained_adapter(2, dim=8, rank=4)
    sim = cosine_similarity_matrix([a, b, a], [None, None, a])  # third client returned its start unchanged
    assert sim[2, 0] == 0.0 and sim[2, 1] == 0.0 and sim[2, 2] == 1.0


def test_incompatible_adapters_rejected():
    with pytest.raises(ValueError):
        delta_gram([trained_adapter(1, dim=8, rank=4), trained_adapter(2, dim=8, rank=8)])


# ------------------------------------------------------------------- k-means


def _planted(n_per=3, noise=0.01, dim=16, rank=8):
    dirs = planted_directions(2, dim=dim, rank=rank)
    ads, truth = [], []
    for g in (0, 1):
        for i in range(n_per):
            ads.append(trained_adapter(100 * g + i, dim=dim, rank=rank, direction=dirs[g], noise=noise))
            truth.append(g)
    return ads, truth


def test_kmeans_recovers_planted_groups():
    ads, truth = _planted()
    labels, inertia = kmeans_on_similarity(cosine_similarity_matrix(ads), k=2, seed=0)
    assert adjusted_rand_index(labels, truth) == pytest.approx(1.0)
    assert labels[0] == 0  # canonical: first point is cluster 0
    assert inertia >= 0


def test_kmeans_is_deterministic_per_seed():
    ads, _ = _planted()
    sim = cosine_similarity_matrix(ads)
    a, _ = kmeans_on_similarity(sim, 2, seed=5)
    b, _ = kmeans_on_similarity(sim, 2, seed=5)
    np.testing.assert_array_equal(a, b)


def test_kmeans_matches_scikit_learn_on_flattened_vectors():
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    ads, truth = _planted(n_per=4, noise=0.05)
    vecs = np.stack([flatten_delta(a) for a in ads])
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sk = sklearn_cluster.KMeans(n_clusters=2, n_init=10, random_state=0).fit(unit).labels_
    ours, _ = kmeans_on_similarity(cosine_similarity_matrix(ads), 2, seed=0)
    assert adjusted_rand_index(ours, sk) == pytest.approx(1.0)
    assert adjusted_rand_index(ours, truth) == pytest.approx(1.0)


def test_kmeans_k_bounds():
    sim = np.eye(3)
    with pytest.raises(ValueError):
        kmeans_on_similarity(sim, k=4)
    with pytest.raises(ValueError):
        kmeans_on_similarity(sim, k=0)
    labels, _ = kmeans_on_similarity(sim, k=1)
    assert set(labels) == {0}


def test_align_labels_and_ari():
    aligned, mapping = align_labels([1, 1, 0, 0], [0, 0, 1, 1], 2)
    assert aligned.tolist() == [0, 0, 1, 1] and mapping == {0: 1, 1: 0}
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.0 + 1e-9


def test_cohesion_separation():
    ads, truth = _planted()
    sim = cosine_similarity_matrix(ads)
    within, between = cohesion_and_separation(sim, np.asarray(truth))
    assert within > 0.9 and abs(between) < 0.5


# -------------------------------------------------- assigner: re-clustering


def _clients(n_per=3):
    ads, truth = _planted(n_per=n_per)
    ids = [f"{WEB if g == 0 else SCI}-{i}" for g, i in zip(truth, [j % n_per for j in range(2 * n_per)])]
    return dict(zip(ids, ads)), {cid: (WEB if cid.startswith(WEB) else SCI) for cid in ids}


def test_not_due_before_round_two_and_only_once_by_default():
    ups, truth = _clients()
    a = ClusterAssigner(truth, [WEB, SCI], recluster_after_round=2)
    assert a.maybe_recluster(1, ups) is None
    assert a.maybe_recluster(2, ups) is not None
    assert a.maybe_recluster(3, ups) is None
    every = ClusterAssigner(truth, [WEB, SCI], recluster_after_round=2, recluster_every=2)
    assert [every.due(r) for r in range(1, 7)] == [False, True, False, True, False, True]


def test_dynamic_recluster_moves_misassigned_client_and_keeps_cluster_ids_stable():
    ups, truth = _clients()
    warm = dict(truth)
    warm["sci-2"] = WEB  # wrong warm-start label
    a = ClusterAssigner(warm, [WEB, SCI], seed=1)
    rec = a.maybe_recluster(2, ups)
    assert rec.method == "dynamic"
    assert rec.moved == {"sci-2": (WEB, SCI)}
    assert a.assignment == truth  # ids stayed "web"/"sci" (not permuted)
    cmp = rec.comparison
    assert cmp["agreement"] == pytest.approx(5 / 6)
    assert cmp["dynamic_cohesion"] > cmp["warm_start_cohesion"]  # dynamic groups updates better
    assert cmp["ari"] < 1.0
    assert np.array(rec.similarity).shape == (6, 6)


def test_dynamic_agrees_with_correct_warm_start():
    ups, truth = _clients()
    a = ClusterAssigner(truth, [WEB, SCI])
    rec = a.maybe_recluster(2, ups)
    assert rec.method == "dynamic" and rec.moved == {}
    assert rec.comparison["agreement"] == 1.0 and rec.comparison["ari"] == pytest.approx(1.0)


def test_static_mode_never_reclusters():
    ups, truth = _clients()
    warm = dict(truth)
    warm["sci-2"] = WEB
    a = ClusterAssigner(warm, [WEB, SCI], mode="static")
    assert a.maybe_recluster(2, ups) is None
    assert a.assignment == warm


def test_fallback_too_few_clients_for_k():
    ups, truth = _clients()
    a = ClusterAssigner(truth, [WEB, SCI])
    rec = a.maybe_recluster(2, {"web-0": ups["web-0"]})
    assert rec.method == "static_fallback" and "only 1 client" in rec.reason
    assert a.assignment == truth


def test_fallback_on_degenerate_tiny_cluster():
    ups, truth = _clients()
    a = ClusterAssigner(truth, [WEB, SCI], min_cluster_size=3, min_separation=-1.0)
    # drop the sci clients down to 1 so k-means yields a singleton cluster
    only = {k: v for k, v in ups.items() if k.startswith(WEB) or k == "sci-0"}
    rec = a.maybe_recluster(2, only)
    assert rec.method == "static_fallback" and "degenerate" in rec.reason
    assert a.assignment == truth


def test_fallback_on_weak_separation():
    rng_dirs = planted_directions(1, dim=16, rank=8)[0]
    same = {f"web-{i}": trained_adapter(i, dim=16, rank=8, direction=rng_dirs, noise=0.001) for i in range(3)}
    same.update({f"sci-{i}": trained_adapter(10 + i, dim=16, rank=8, direction=rng_dirs, noise=0.001) for i in range(3)})
    truth = {k: (WEB if k.startswith(WEB) else SCI) for k in same}
    a = ClusterAssigner(truth, [WEB, SCI])
    rec = a.maybe_recluster(2, same)  # all updates point the same way => nothing to separate
    assert rec.method == "static_fallback" and "weak separation" in rec.reason
    assert a.assignment == truth


def test_fallback_on_unknown_client_and_on_clustering_error():
    ups, truth = _clients()
    a = ClusterAssigner(truth, [WEB, SCI])
    rec = a.maybe_recluster(2, {**ups, "stranger": ups["web-0"]})
    assert rec.method == "static_fallback" and "unassigned" in rec.reason
    bad = dict(ups)
    bad["web-0"] = trained_adapter(0, dim=16, rank=4)  # wrong rank => incompatible
    rec = a.maybe_recluster(2, bad)
    assert rec.method == "static_fallback" and "clustering failed" in rec.reason
    assert a.assignment == truth


def test_dropout_clients_keep_their_assignment():
    ups, truth = _clients()
    warm = dict(truth)
    warm["sci-2"] = WEB
    a = ClusterAssigner(warm, [WEB, SCI])
    present = {k: v for k, v in ups.items() if k not in ("web-2", "sci-0")}  # two dropouts
    rec = a.maybe_recluster(2, present)
    assert rec.method == "dynamic"
    assert a.assignment["web-2"] == WEB and a.assignment["sci-0"] == SCI  # untouched
    assert a.assignment["sci-2"] == SCI  # still recovered
    assert sorted(rec.clients_clustered) == sorted(present)


def test_constructor_validation():
    with pytest.raises(ValueError):
        ClusterAssigner({"a": "x"}, ["y", "z"])
    with pytest.raises(ValueError):
        ClusterAssigner({"a": "x"}, ["x", "y"], mode="nope")
    with pytest.raises(ValueError):
        ClusterAssigner({"a": "x"}, ["x", "y"], k=3)
