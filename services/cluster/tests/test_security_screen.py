"""P3's adapter validation guards the cluster's door (D7, threat model).

Every upload is checked by ``security.validation.validate_adapter_integrity``
(NaN / Inf, LoRA rank above D8's cap, absurd tensor norms) before it can reach
an aggregate, and every round is screened by ``security.validation.
detect_poisoning`` (robust norm outliers, 3+ clients) before it is merged.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster import server
from tests.conftest import trained_adapter
from tests.http_helpers import create_cluster, reset_server_state, restore_server_state, upload_body

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def _clean():
    saved = reset_server_state()
    yield
    restore_server_state(saved)


def _poisoned(seed, scale):
    adapter = trained_adapter(seed)
    for layer in adapter.layer_indices:
        for m in adapter.target_modules:
            adapter.modules[layer][m]["lora_B"] = (adapter.modules[layer][m]["lora_B"] * scale
                                                   ).astype(np.float32)
    return adapter


def _with_tensor(body, name, value):
    from cluster.schemas.messages import TensorPayload

    body["tensors"] = [TensorPayload.from_numpy(t["name"], value).model_dump()
                       if t["name"] == name else t for t in body["tensors"]]
    return body


def test_an_upload_with_nan_never_reaches_the_buffer():
    create_cluster(client, "web", members={"web": ["a"]})
    body = upload_body("a", trained_adapter(1))
    name = body["tensors"][0]["name"]
    bad = trained_adapter(1).to_state_dict()[name].copy()
    bad[0, 0] = np.nan
    r = client.post("/clusters/web/uploads", json=_with_tensor(body, name, bad))
    assert r.status_code == 422
    assert server._clusters["web"].uploads == {}


def test_an_upload_with_an_absurd_norm_is_refused_by_the_security_check():
    create_cluster(client, "web", members={"web": ["a"]})
    body = upload_body("a", trained_adapter(1))
    name = next(t["name"] for t in body["tensors"] if "lora_B" in t["name"])
    huge = trained_adapter(1).to_state_dict()[name] * 1000.0
    r = client.post("/clusters/web/uploads", json=_with_tensor(body, name, huge.astype(np.float32)))
    assert r.status_code == 422 and "security.validation" in r.text and "norm" in r.text
    assert server._clusters["web"].uploads == {}


def test_an_upload_above_the_rank_cap_is_refused():
    create_cluster(client, "web", members={"web": ["a"]})
    r = client.post("/clusters/web/uploads", json=upload_body("a", trained_adapter(1, rank=32)))
    assert r.status_code == 422 and "max_rank" in r.text


def test_a_norm_outlier_is_left_out_of_the_merge_and_named():
    members = ["a", "b", "c", "d"]
    create_cluster(client, "web", members={"web": members})
    for i, c in enumerate(members[:3]):
        assert client.post("/clusters/web/uploads",
                           json=upload_body(c, trained_adapter(10 + i))).status_code == 201
    # 40x the others' norm: under the per-tensor cap, far outside the round
    assert client.post("/clusters/web/uploads",
                       json=upload_body("d", _poisoned(20, 40.0))).status_code == 201
    r = client.post("/clusters/web/aggregate", json={})
    assert r.status_code == 200, r.text
    assert r.json()["source_clients"] == ["a", "b", "c"]
    manifest = client.get("/clusters/web/aggregate/manifest").json()
    assert manifest["skipped_clients"]["d"].startswith("poisoning_suspect")
    assert manifest["screening"]["flagged"] == ["d"]


def test_a_normal_round_is_merged_whole():
    members = ["a", "b", "c"]
    create_cluster(client, "web", members={"web": members})
    for i, c in enumerate(members):
        client.post("/clusters/web/uploads", json=upload_body(c, trained_adapter(10 + i)))
    r = client.post("/clusters/web/aggregate", json={})
    assert r.json()["source_clients"] == members
    assert client.get("/clusters/web/aggregate/manifest").json()["screening"]["flagged"] == []
