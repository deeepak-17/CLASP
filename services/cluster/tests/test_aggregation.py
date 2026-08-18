"""Aggregation correctness tests backing the Week 5 demo's SVD claim.

Covers the Week 3 Thu target ("Unit test: SVD agg == exact avg; naive avg
kept as ablation baseline") which had no automated coverage before Week 5 —
only services/cluster/tests/test_smoke.py (a bare import check) existed.
"""

from __future__ import annotations

import numpy as np

from cluster.adapter_format import random_adapter
from cluster.aggregation import aggregate_naive, aggregate_svd, exact_average_delta


def test_full_rank_svd_matches_exact_average():
    """At full rank the SVD re-factorization is a lossless reconstruction."""
    dim = 16
    adapters = [random_adapter(dim, dim, rank=dim, seed=s) for s in range(3)]
    weights = [1.0, 2.0, 3.0]

    exact = exact_average_delta(iter(adapters), weights)
    merged = aggregate_svd(iter(adapters), weights, rank=dim)

    for module in merged.target_modules:
        recon = merged.delta_w(module)
        np.testing.assert_allclose(recon, exact[module], atol=1e-5)


def test_truncated_svd_is_closer_to_exact_than_naive_averaging():
    """D2 ablation claim: mean(B)@mean(A) != mean(B@A); SVD path tracks the exact mean."""
    dim = 16
    adapters = [random_adapter(dim, dim, rank=8, seed=s) for s in range(3)]
    weights = [1.0, 1.0, 1.0]

    exact = exact_average_delta(iter(adapters), weights)
    svd_merged = aggregate_svd(iter(adapters), weights, rank=8)
    naive_merged = aggregate_naive(iter(adapters), weights)

    for module in svd_merged.target_modules:
        svd_err = np.linalg.norm(svd_merged.delta_w(module) - exact[module])
        naive_err = np.linalg.norm(naive_merged.delta_w(module) - exact[module])
        assert svd_err <= naive_err + 1e-9


def test_aggregate_svd_rejects_mismatched_target_modules():
    a = random_adapter(8, 8, rank=4, seed=0)
    b = random_adapter(8, 8, rank=4, seed=1, target_modules=("q_proj",))
    try:
        aggregate_svd(iter([a, b]), [1.0, 1.0])
        assert False, "expected ValueError for mismatched target_modules"
    except ValueError:
        pass
