"""Week 7 Thu: straggler policy (timeout-skip, sample-weighting, quorum)."""

from __future__ import annotations

import pytest

from cluster.straggler import ClientUpdate, StragglerPolicy, apply_policy
from tests.conftest import trained_adapter


def _upd(cid: str, n: int, dur: float | None, seed: int = 0) -> ClientUpdate:
    return ClientUpdate(cid, trained_adapter(seed), n, duration_s=dur)


def test_default_policy_accepts_everyone_sample_weighted():
    out = apply_policy([_upd("a", 10, 1.0), _upd("b", 30, 99.0)])
    assert [u.client_id for u in out.accepted] == ["a", "b"]
    assert out.weights == [10.0, 30.0]
    assert out.skipped == {}
    assert out.quorum_met


def test_timeout_skips_slow_clients_and_reweights_survivors():
    policy = StragglerPolicy(timeout_s=5.0)
    out = apply_policy([_upd("fast", 10, 1.0), _upd("slow", 50, 9.0), _upd("ok", 30, 5.0)], policy)
    assert [u.client_id for u in out.accepted] == ["fast", "ok"]  # boundary (== timeout) is kept
    assert out.skipped == {"slow": "timeout"}
    assert out.weights == [10.0, 30.0]


def test_client_without_duration_is_not_a_straggler():
    out = apply_policy([_upd("x", 10, None)], StragglerPolicy(timeout_s=1.0))
    assert [u.client_id for u in out.accepted] == ["x"]


def test_zero_example_clients_skipped():
    out = apply_policy([_upd("a", 0, 1.0), _upd("b", 5, 1.0)])
    assert out.skipped == {"a": "no_examples"}


def test_uniform_weighting():
    out = apply_policy([_upd("a", 10, 1.0), _upd("b", 90, 1.0)], StragglerPolicy(weighting="uniform"))
    assert out.weights == [1.0, 1.0]


def test_quorum_counts_clients_that_never_answered():
    policy = StragglerPolicy(min_fraction=0.5)
    # 4 clients were asked, only 1 answered => 25% < 50%
    out = apply_policy([_upd("a", 10, 1.0)], policy, expected=4)
    assert out.required == 2
    assert not out.quorum_met
    out = apply_policy([_upd("a", 10, 1.0), _upd("b", 10, 1.0)], policy, expected=4)
    assert out.quorum_met


def test_min_clients_floor():
    out = apply_policy([_upd("a", 10, 1.0)], StragglerPolicy(min_clients=2))
    assert not out.quorum_met


@pytest.mark.parametrize(
    "kwargs",
    [{"weighting": "bogus"}, {"timeout_s": 0}, {"min_clients": 0}, {"min_fraction": 1.5}],
)
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        StragglerPolicy(**kwargs)
