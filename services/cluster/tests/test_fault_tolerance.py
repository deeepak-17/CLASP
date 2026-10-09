"""Week 4 Thu / Week 7 Thu: SVDLoRAStrategy.aggregate_fit under client failures."""

from __future__ import annotations

import numpy as np
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from cluster.adapter_format import LoRAAdapter, random_adapter
from cluster.aggregation import aggregate_svd
from cluster.server import build_strategy
from cluster.straggler import StragglerPolicy
from tests.conftest import trained_adapter


def _fit(adapter, n=100, loss=0.5, duration=None, code=Code.OK):
    metrics = {"loss": loss}
    if duration is not None:
        metrics["duration_s"] = duration
    return FitRes(
        status=Status(code=code, message=""),
        parameters=ndarrays_to_parameters(adapter.to_ndarrays()),
        num_examples=n,
        metrics=metrics,
    )


class _Proxy:
    def __init__(self, cid):
        self.cid = cid


def _strategy(**kw):
    return build_strategy(random_adapter(32, 32, seed=0), min_clients=1, **kw)


def test_happy_path_matches_core_and_reports_metrics():
    ads = [trained_adapter(s) for s in (1, 2, 3)]
    ns = [100, 200, 150]
    strat = _strategy()
    params, m = strat.aggregate_fit(
        1, [(_Proxy(f"c{i}"), _fit(a, n, 0.1 * (i + 1))) for i, (a, n) in enumerate(zip(ads, ns))], []
    )
    ref = aggregate_svd(iter(ads), [float(n) for n in ns], rank=16)
    got = LoRAAdapter.from_ndarrays(parameters_to_ndarrays(params))
    for mod in ref.target_modules:
        np.testing.assert_allclose(got.delta_w(mod), ref.delta_w(mod), atol=1e-6)
    assert m["num_clients"] == 3 and m["num_failures"] == 0 and m["round"] == 1
    assert m["duration_s"] >= 0 and abs(m["mean_loss"] - 0.2) < 1e-9
    assert strat.round_log[-1] == m


def test_transport_failures_counted_not_aggregated():
    strat = _strategy()
    params, m = strat.aggregate_fit(1, [(_Proxy("a"), _fit(trained_adapter(1)))], [object(), object()])
    assert params is not None
    assert m["num_clients"] == 1 and m["num_failures"] == 2


def test_non_ok_status_client_is_skipped():
    good, bad = trained_adapter(1), trained_adapter(2)
    strat = _strategy()
    params, m = strat.aggregate_fit(
        1,
        [(_Proxy("good"), _fit(good)), (_Proxy("bad"), _fit(bad, code=Code.FIT_NOT_IMPLEMENTED))],
        [],
    )
    got = LoRAAdapter.from_ndarrays(parameters_to_ndarrays(params))
    ref = aggregate_svd(iter([good]), [100.0], rank=16)  # `bad` did not contribute
    np.testing.assert_allclose(got.delta_w("q_proj"), ref.delta_w("q_proj"), atol=1e-6)
    assert m["num_clients"] == 1 and m["num_failures"] == 1


def test_malformed_payload_is_skipped_instead_of_crashing_the_round():
    corrupt = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([np.zeros((16, 32), dtype=np.float32)] * 2),
        num_examples=10,
        metrics={},
    )
    strat = _strategy()
    params, m = strat.aggregate_fit(1, [(_Proxy("a"), _fit(trained_adapter(1))), (_Proxy("b"), corrupt)], [])
    assert params is not None and m["num_clients"] == 1 and m["num_failures"] == 1


def test_timeout_straggler_skipped_and_survivors_sample_weighted():
    fast1, fast2, slow = trained_adapter(1), trained_adapter(2), trained_adapter(3)
    strat = _strategy(straggler_policy=StragglerPolicy(timeout_s=10.0))
    params, m = strat.aggregate_fit(
        1,
        [
            (_Proxy("f1"), _fit(fast1, n=100, duration=1.0)),
            (_Proxy("f2"), _fit(fast2, n=300, duration=2.0)),
            (_Proxy("slow"), _fit(slow, n=1000, duration=60.0)),
        ],
        [],
    )
    ref = aggregate_svd(iter([fast1, fast2]), [100.0, 300.0], rank=16)
    got = LoRAAdapter.from_ndarrays(parameters_to_ndarrays(params))
    np.testing.assert_allclose(got.delta_w("v_proj"), ref.delta_w("v_proj"), atol=1e-6)
    assert m["num_clients"] == 2 and m["num_failures"] == 1


def test_quorum_not_met_skips_round_entirely():
    strat = _strategy(straggler_policy=StragglerPolicy(min_fraction=0.5))
    results = [(_Proxy("ok"), _fit(trained_adapter(1)))] + [
        (_Proxy(f"s{i}"), _fit(trained_adapter(10 + i), code=Code.FIT_NOT_IMPLEMENTED)) for i in range(3)
    ]
    params, m = strat.aggregate_fit(1, results, [])
    assert params is None and m == {}
    assert strat.round_log[-1]["skipped"] is True


def test_no_results_returns_none():
    assert _strategy().aggregate_fit(1, [], []) == (None, {})


def test_strategy_handles_multi_layer_adapters():
    init = random_adapter(32, 32, num_layers=3, seed=0)
    strat = build_strategy(init, min_clients=1)
    ads = [trained_adapter(s, num_layers=3) for s in (1, 2)]
    params, _ = strat.aggregate_fit(1, [(_Proxy(f"c{i}"), _fit(a)) for i, a in enumerate(ads)], [])
    got = LoRAAdapter.from_ndarrays(parameters_to_ndarrays(params), num_layers=3)
    ref = aggregate_svd(iter(ads), [100.0, 100.0], rank=16)
    for layer in range(3):
        np.testing.assert_allclose(got.delta_w("k_proj", layer), ref.delta_w("k_proj", layer), atol=1e-6)
