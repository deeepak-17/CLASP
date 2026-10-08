"""Aggregation core: ΔW reconstruction, streamed mean, SVD, naive baseline, multi-layer."""

from __future__ import annotations

import numpy as np
import pytest

from cluster.adapter_format import LoRAAdapter
from cluster.aggregation import (
    StreamingWeightedMean,
    aggregate_naive,
    aggregate_svd,
    exact_average_delta,
    truncated_svd_refactor,
)
from tests.conftest import trained_adapter

DIM, RANK = 16, 4


def _adapters(n=3, layers=1, dim=DIM, rank=RANK, base=0):
    return [trained_adapter(base + i, dim=dim, rank=rank, num_layers=layers) for i in range(n)]


# ---- ΔW = B @ A -----------------------------------------------------------------


def test_delta_w_is_b_times_a_per_layer_and_module():
    ad = trained_adapter(1, dim=12, rank=3, num_layers=3)
    for layer in ad.layer_indices:
        for m in ad.target_modules:
            pair = ad.modules[layer][m]
            expected = pair["lora_B"].astype(np.float64) @ pair["lora_A"].astype(np.float64)
            np.testing.assert_allclose(ad.delta_w(m, layer), expected, rtol=1e-5, atol=1e-6)
            assert ad.delta_w(m, layer).shape == (12, 12)
    # layers differ (they were not accidentally aliased)
    assert not np.allclose(ad.delta_w("q_proj", 0), ad.delta_w("q_proj", 1))


def test_delta_w_shape_follows_in_and_out_features():
    from cluster.adapter_format import random_adapter

    ad = random_adapter(20, 12, rank=4, seed=0)
    ad.modules[0]["q_proj"]["lora_B"] += 1.0
    assert ad.delta_w("q_proj").shape == (12, 20)  # (out, in)


# ---- streamed exact mean --------------------------------------------------------


def test_streaming_mean_equals_weighted_average_and_is_order_independent():
    rng = np.random.default_rng(0)
    values = [rng.normal(size=(5, 4)) for _ in range(6)]
    weights = rng.uniform(0.5, 5.0, size=6)
    expected = np.average(np.stack(values), axis=0, weights=weights)

    def run(order):
        acc = StreamingWeightedMean()
        for i in order:
            acc.update(values[i], weights[i])
        return acc

    forward = run(range(6))
    backward = run(reversed(range(6)))
    np.testing.assert_allclose(forward.result(), expected, atol=1e-12)
    np.testing.assert_allclose(backward.result(), expected, atol=1e-12)
    assert forward.total_weight == pytest.approx(weights.sum())


def test_streaming_mean_guards():
    acc = StreamingWeightedMean()
    with pytest.raises(ValueError, match="no contributions"):
        acc.result()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="weight must be positive"):
            acc.update(np.ones(3), bad)


def test_streaming_mean_holds_one_accumulator_and_does_not_alias_inputs():
    acc = StreamingWeightedMean()
    first = np.ones(4)
    acc.update(first, 1.0)
    acc.update(np.full(4, 3.0), 1.0)
    np.testing.assert_allclose(acc.result(), 2.0)
    np.testing.assert_array_equal(first, np.ones(4))  # input untouched


# ---- exact average --------------------------------------------------------------


def test_exact_average_delta_is_the_weighted_mean_of_the_products():
    ads, w = _adapters(3), [1.0, 2.0, 5.0]
    exact = exact_average_delta(iter(ads), w)
    for m in ads[0].target_modules:
        ref = sum(wi * a.delta_w(m) for wi, a in zip(w, ads, strict=True)) / sum(w)
        np.testing.assert_allclose(exact[m], ref, atol=1e-6)


# ---- SVD aggregation ------------------------------------------------------------


def test_svd_is_exact_when_the_mean_fits_in_the_output_rank_and_respects_weights():
    ads, w = _adapters(2), [1.0, 3.0]
    merged = aggregate_svd(iter(ads), w, rank=2 * RANK)  # mean of two rank-4 maps has rank <= 8
    exact = exact_average_delta(iter(ads), w)
    assert merged.rank == 2 * RANK
    for m in merged.target_modules:
        np.testing.assert_allclose(merged.delta_w(m), exact[m], atol=1e-5)
    # the weights matter: the unweighted mean is a different matrix
    unweighted = exact_average_delta(iter(ads), [1.0, 1.0])
    assert not np.allclose(exact["q_proj"], unweighted["q_proj"], atol=1e-4)


def test_truncation_error_equals_the_discarded_singular_values():
    """Eckart-Young: ||mean - B'A'||_F^2 = sum of squared dropped singular values."""
    ads = _adapters(4)
    r = 5
    merged = aggregate_svd(iter(ads), [1.0] * 4, rank=r)
    exact = exact_average_delta(iter(ads), [1.0] * 4)
    for m in merged.target_modules:
        s = np.linalg.svd(exact[m], compute_uv=False)
        err = np.linalg.norm(merged.delta_w(m) - exact[m]) ** 2
        np.testing.assert_allclose(err, np.sum(s[r:] ** 2), rtol=1e-3, atol=1e-8)


def test_single_adapter_is_reproduced_and_dtype_is_preserved():
    (ad,) = _adapters(1)
    merged = aggregate_svd(iter([ad]), [7.0])
    for m in ad.target_modules:
        np.testing.assert_allclose(merged.delta_w(m), ad.delta_w(m), atol=1e-5)
        assert merged.modules[0][m]["lora_A"].dtype == np.float32
        assert merged.modules[0][m]["lora_B"].dtype == np.float32
    assert merged.rank == ad.rank and merged.alpha == ad.alpha


def test_multi_layer_adapters_are_aggregated_layer_by_layer():
    ads, w = _adapters(3, layers=3), [2.0, 1.0, 1.0]
    merged = aggregate_svd(iter(ads), w, rank=3 * RANK)
    assert merged.num_layers == 3
    for layer in range(3):
        exact = exact_average_delta(iter(ads), w, layer=layer)
        for m in merged.target_modules:
            np.testing.assert_allclose(merged.delta_w(m, layer), exact[m], atol=1e-5)
    merged.validate()


def test_adapters_are_consumed_as_a_single_stream():
    ads = _adapters(3)
    consumed: list[int] = []

    def stream():
        for i, a in enumerate(ads):
            consumed.append(i)
            yield a

    aggregate_svd(stream(), [1.0, 1.0, 1.0])
    assert consumed == [0, 1, 2]


@pytest.mark.parametrize("fn", [aggregate_svd, aggregate_naive, exact_average_delta])
def test_length_mismatch_between_adapters_and_weights_is_an_error_not_a_truncation(fn):
    ads = _adapters(3)
    with pytest.raises(ValueError, match="more entries than weights"):
        fn(iter(ads), [1.0, 1.0])
    with pytest.raises(ValueError, match="more entries than adapters"):
        fn(iter(ads[:2]), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("fn", [aggregate_svd, aggregate_naive, exact_average_delta])
def test_empty_input_is_an_error(fn):
    with pytest.raises(ValueError, match="no adapters"):
        fn(iter([]), [])


def test_non_positive_weight_is_rejected():
    with pytest.raises(ValueError, match="weight must be positive"):
        aggregate_svd(iter(_adapters(2)), [1.0, 0.0])


# ---- naive baseline -------------------------------------------------------------


def _independent(n, dim=DIM, rank=RANK):
    """Clients whose A *and* B differ (after local training A drifts too)."""
    out = []
    for seed in range(n):
        ad = trained_adapter(seed, dim=dim, rank=rank)
        rng = np.random.default_rng(100 + seed)
        for m in ad.target_modules:
            ad.modules[0][m]["lora_A"] = rng.normal(scale=0.1, size=(rank, dim)).astype(np.float32)
        out.append(ad)
    return out


def test_naive_is_the_weighted_mean_of_the_factors_and_differs_from_exact():
    ads, w = _independent(3), [1.0, 1.0, 2.0]
    naive = aggregate_naive(iter(ads), w)
    for m in naive.target_modules:
        for part in ("lora_A", "lora_B"):
            ref = sum(wi * a.modules[0][m][part] for wi, a in zip(w, ads, strict=True)) / sum(w)
            np.testing.assert_allclose(naive.modules[0][m][part], ref, atol=1e-6)
    exact = exact_average_delta(iter(ads), w)
    # mean(B)·mean(A) != mean(B·A) for independent clients
    assert not np.allclose(naive.delta_w("q_proj"), exact["q_proj"], atol=1e-4)


def test_naive_equals_exact_only_in_the_degenerate_shared_A_case():
    """Clients that share A (e.g. nobody trained A yet) make mean(B)·A == mean(B·A);
    that is the one case where the baseline is exact, and why it needs independent A to differ."""
    ads, w = _adapters(3), [1.0, 2.0, 3.0]  # trained_adapter shares A across clients
    naive = aggregate_naive(iter(ads), w)
    exact = exact_average_delta(iter(ads), w)
    np.testing.assert_allclose(naive.delta_w("q_proj"), exact["q_proj"], atol=1e-6)


def test_all_methods_agree_when_every_client_returns_the_same_adapter():
    (ad,) = _adapters(1)
    same = [ad, ad, ad]
    svd = aggregate_svd(iter(same), [1.0, 2.0, 3.0])
    naive = aggregate_naive(iter(same), [1.0, 2.0, 3.0])
    for m in ad.target_modules:
        np.testing.assert_allclose(svd.delta_w(m), ad.delta_w(m), atol=1e-5)
        np.testing.assert_allclose(naive.delta_w(m), ad.delta_w(m), atol=1e-5)


def test_svd_refactor_returns_a_valid_adapter_pair_shape():
    delta = np.random.default_rng(3).normal(size=(10, 14))
    a, b = truncated_svd_refactor(delta, 6)
    assert a.shape == (6, 14) and b.shape == (10, 6)
    s = np.linalg.svd(delta, compute_uv=False)
    np.testing.assert_allclose(
        np.linalg.norm(delta - b @ a) ** 2, np.sum(s[6:] ** 2), rtol=1e-8
    )
    assert isinstance(LoRAAdapter(), LoRAAdapter)  # sanity: default construction is cheap
