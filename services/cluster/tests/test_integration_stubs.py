"""P2-side registry seam: SnapshotSink / AdapterRef mapping (stub, not a registry)."""

from __future__ import annotations

import numpy as np
import pytest
from contracts.types import AdapterKind, AdapterRef
from fastapi.testclient import TestClient

from cluster import server
from cluster.integration import InMemorySnapshotSink, cluster_adapter_ref
from cluster.redistribution import adapter_from_broadcast
from tests.conftest import trained_adapter
from tests.http_helpers import reset_server_state, restore_server_state, upload_body

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def test_cluster_adapter_ref_maps_onto_the_contracts_type():
    ref = cluster_adapter_ref("cluster-web", 0)
    assert ref == AdapterRef("cluster-web", 1, AdapterKind.CLUSTER, "cluster-web")
    assert cluster_adapter_ref("cluster-web", 4).version == 5  # round r -> version r + 1


def test_http_aggregate_publishes_a_loadable_snapshot_per_round():
    sink = InMemorySnapshotSink()
    server.configure_snapshot_sink(sink)
    for rnd in range(2):
        for c in ("c1", "c2"):
            r = client.post(
                "/clusters/team-a/uploads",
                json=upload_body(c, trained_adapter(rnd * 10 + ord(c[1])), round_id=rnd),
            )
            assert r.status_code == 201
        resp = client.post("/clusters/team-a/aggregate")
        assert resp.status_code == 200
    assert [ref.version for ref, _ in sink.snapshots] == [1, 2]
    ref, broadcast = sink.latest("team-a")
    assert ref.kind is AdapterKind.CLUSTER and ref.cluster_id == "team-a"
    assert broadcast.peft_config["lora_alpha"] == 16.0 and broadcast.peft_config["r"] == 16
    # the published payload is the same adapter /adapters/active serves
    active = adapter_from_broadcast(server._clusters["team-a"].active)
    np.testing.assert_array_equal(
        adapter_from_broadcast(broadcast).delta_w("q_proj"), active.delta_w("q_proj")
    )
    assert client.get("/clusters/team-a/aggregate/manifest").json()["published_as"] == {
        "name": "team-a", "version": 2,
    }
    assert sink.latest("nobody") is None


def test_a_failing_sink_is_reported_but_never_loses_the_round():
    class Down:
        def publish(self, ref, broadcast):
            raise ConnectionError("registry unreachable")

    server.configure_snapshot_sink(Down())
    client.post("/clusters/team-a/uploads", json=upload_body("c1", trained_adapter(1)))
    resp = client.post("/clusters/team-a/aggregate")
    assert resp.status_code == 200
    assert server._clusters["team-a"].round_id == 1 and server._clusters["team-a"].active
    manifest = client.get("/clusters/team-a/aggregate/manifest").json()
    assert "ConnectionError" in manifest["publish_error"] and "published_as" not in manifest


def test_federation_publishes_each_aggregated_cluster_and_survives_a_dead_registry():
    pytest.importorskip("torch")
    from cluster.adapter_format import random_adapter
    from cluster.clustering import ClusterAssigner
    from cluster.federation import MultiClusterFederation
    from cluster.simulation import make_clustered_clients

    def build(sink):
        clients, truth = make_clustered_clients(seed=3, dim=32)
        assigner = ClusterAssigner(truth, ["cluster-web", "cluster-sci"], mode="static")
        return MultiClusterFederation(clients, assigner, random_adapter(32, 32, seed=0), sink=sink)

    sink = InMemorySnapshotSink()
    rnd = build(sink).run_round()
    assert {ref.cluster_id for ref, _ in sink.snapshots} == {"cluster-web", "cluster-sci"}
    assert all(res.published is not None and res.publish_error is None for res in rnd.clusters.values())

    class Down:
        def publish(self, ref, broadcast):
            raise OSError("registry down")

    rnd = build(Down()).run_round()
    for res in rnd.clusters.values():
        assert res.aggregated and res.published is None and "registry down" in res.publish_error
