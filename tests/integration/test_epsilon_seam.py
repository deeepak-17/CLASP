"""D7 across the seams: a client's epsilon survives Edge -> Cluster -> Registry.

The Edge states a client's spent budget in the contracts shape,
``"privacy": {"epsilon": 7.99, "delta": 1e-05, ...}`` (``edge.wire.privacy_block``).
The cluster has to read that block, publish the cluster's epsilon (the max over
the clients it aggregated) with the version it pushes to the registry, and the
registry has to record it. Before the cluster read ``privacy.epsilon`` every edge
upload arrived with epsilon = None and every cluster version was recorded with
``epsilon = null``, whatever the clients had spent.

In-process, like ``test_four_seams.py``: the adapters are tiny synthetic ones and
the cluster -> registry hop is the registry app's own TestClient.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from edge import wire
from tests.integration.test_four_seams import (
    CLUSTER_ID,
    REGISTRY_NAME,
    make_client_adapter,
    publish_over_http,
)

# The edge side of this seam (``privacy`` on ``upload_payload``) lives in the edge
# DP-training change; against an edge that predates it there is nothing to send.
pytestmark = pytest.mark.skipif(
    not hasattr(wire, "privacy_block"), reason="this edge does not send a privacy block yet"
)


@pytest.fixture
def registry_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    import registry.app as appmod

    appmod._store = None
    with TestClient(appmod.app) as c:
        yield c


@pytest.fixture
def cluster_client():
    import cluster.server as srv

    srv._reset_clusters()
    with TestClient(srv.app) as c:
        yield c
    srv._reset_clusters()


def _upload(cluster_client, client_id, seed, privacy):
    sd, cfg = make_client_adapter(seed)
    payload = wire.upload_payload(
        sd, cfg, client_id=client_id, cluster_id=CLUSTER_ID, round_id=0,
        num_examples=50, seed=0, privacy=privacy,
    )
    assert "privacy" in payload  # the real edge payload, not a hand-built one
    resp = cluster_client.post("/uploads", json=payload)
    assert resp.status_code == 201, resp.text


def _publish(cluster_client, registry_client, monkeypatch):
    with publish_over_http(registry_client, monkeypatch):
        resp = cluster_client.post(
            f"/adapters/{CLUSTER_ID}/publish",
            json={"registry_url": "http://registry:8004", "adapter_name": REGISTRY_NAME, "round": 1},
        )
    assert resp.status_code == 201, resp.text
    return resp.json()["version"]


def test_the_edges_epsilon_reaches_the_registry_cluster_version(
    cluster_client, registry_client, monkeypatch
):
    for i, eps in enumerate((2.5, 7.99, 5.0)):
        _upload(cluster_client, f"client-{i}", 100 + i, {"epsilon": eps, "delta": 1e-5})
    agg = cluster_client.post("/aggregate", json={"cluster_id": CLUSTER_ID, "include_manifest": True})
    assert agg.status_code == 200, agg.text
    assert agg.json()["epsilon"] == 7.99  # max over the aggregated clients
    assert cluster_client.get(f"/adapters/{CLUSTER_ID}/manifest").json()["epsilon"] == 7.99

    meta = _publish(cluster_client, registry_client, monkeypatch)
    assert meta["privacy"]["epsilon"] == 7.99  # recorded by the registry, not null

    # and it is what the registry serves afterwards
    listed = registry_client.get(f"/adapters/{REGISTRY_NAME}/versions").json()
    versions = listed["versions"] if isinstance(listed, dict) else listed
    assert versions[-1]["privacy"]["epsilon"] == 7.99


def test_one_client_trained_without_dp_leaves_the_cluster_epsilon_unknown(
    cluster_client, registry_client, monkeypatch
):
    _upload(cluster_client, "client-dp", 1, {"epsilon": 4.0})
    _upload(cluster_client, "client-plain", 2, None)  # DP off: epsilon None
    agg = cluster_client.post("/aggregate", json={"cluster_id": CLUSTER_ID})
    assert agg.status_code == 200 and agg.json()["epsilon"] is None
    meta = _publish(cluster_client, registry_client, monkeypatch)
    assert (meta.get("privacy") or {}).get("epsilon") is None
