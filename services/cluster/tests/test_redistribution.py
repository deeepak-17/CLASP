"""Week 4 Tue-Fri: redistribution round-trip and retry/error handling."""

from __future__ import annotations

import numpy as np
import pytest

from cluster.aggregation import aggregate_svd
from cluster.redistribution import (
    RedistributionError,
    adapter_from_broadcast,
    build_broadcast,
    redistribute,
)
from tests.conftest import trained_adapter


def test_broadcast_round_trip_is_lossless_single_and_multi_layer():
    for layers in (1, 3):
        merged = aggregate_svd(
            iter([trained_adapter(s, num_layers=layers) for s in (1, 2, 3)]), [10.0, 20.0, 30.0]
        )
        bc = build_broadcast(merged, cluster_id="c1", round_id=4, num_clients=3)
        assert (bc.cluster_id, bc.round_id, bc.num_layers, bc.rank) == ("c1", 4, layers, merged.rank)
        back = adapter_from_broadcast(bc)
        assert back.rank == merged.rank and back.num_layers == merged.num_layers
        for layer in merged.layer_indices:
            for m in merged.target_modules:
                for part in ("lora_A", "lora_B"):
                    np.testing.assert_array_equal(
                        back.modules[layer][m][part], merged.modules[layer][m][part]
                    )


def test_broadcast_survives_json_wire_format():
    from cluster.schemas.messages import ClusterAdapterBroadcast

    merged = trained_adapter(5)
    bc = build_broadcast(merged, cluster_id="c1", round_id=0, num_clients=1)
    wire = ClusterAdapterBroadcast.model_validate_json(bc.model_dump_json())
    np.testing.assert_array_equal(adapter_from_broadcast(wire).delta_w("q_proj"), merged.delta_w("q_proj"))


def test_retry_recovers_from_transient_failures_with_exponential_backoff():
    calls: dict[str, int] = {}
    sleeps: list[float] = []

    def flaky(client_id, payload):
        calls[client_id] = calls.get(client_id, 0) + 1
        if client_id == "b" and calls[client_id] < 3:
            raise ConnectionError("temporary")

    report = redistribute(
        "payload", ["a", "b", "c"], flaky, max_retries=3, backoff_s=0.5, sleep=sleeps.append
    )
    assert report.all_delivered
    assert report.delivered == ["a", "b", "c"]
    assert report.attempts == {"a": 1, "b": 3, "c": 1}
    assert sleeps == [0.5, 1.0]  # 0.5 * 2**0, 0.5 * 2**1


def test_permanent_failure_is_reported_not_silenced_and_does_not_block_others():
    delivered = []

    def send(client_id, payload):
        if client_id == "dead":
            raise TimeoutError("unreachable")
        delivered.append(client_id)

    report = redistribute("p", ["a", "dead", "z"], send, max_retries=2, sleep=lambda _s: None)
    assert delivered == ["a", "z"]
    assert not report.all_delivered
    assert report.attempts["dead"] == 3  # first try + 2 retries
    assert "TimeoutError" in report.failed["dead"]


def test_require_all_raises():
    def send(client_id, payload):
        raise OSError("boom")

    with pytest.raises(RedistributionError, match="did not receive"):
        redistribute("p", ["a"], send, max_retries=0, require_all=True)


def test_negative_retries_rejected():
    with pytest.raises(ValueError):
        redistribute("p", ["a"], lambda *_: None, max_retries=-1)
