"""A part's rollback/restore must change what the edge serves (D5 + D6 + D11).

The edge serves the pre-merged composite, so moving only the part's `active`
pointer would leave the bad weights live. These tests read the composite's
`composed_from` and payload after every rollback path.
"""
from __future__ import annotations

import numpy as np
from safetensors.numpy import load

from .test_composite import make_adapter
from .test_composite_api import _compose, _save, _seed
from .test_promote_api import _promote_body

COMPOSITE = "composite-flask"


def _active(api, name):
    return api.get(f"/adapters/{name}/active").json()


def _served_client_version(api):
    return _active(api, COMPOSITE)["composed_from"]["client_version"]


def _promote_v2_with_composite(api):
    """flask v1 composed, then v2 promoted with build-on-promote: composite v2 serves v2."""
    _seed(api)
    assert _compose(api).status_code == 201
    _save(api, "flask", make_adapter(3), kind="client", cluster_id="web")
    body = _promote_body("flask", 2, good=True)
    body["composite"] = {"name": COMPOSITE, "cluster": "cluster-web", "client": "flask",
                         "alpha": 0.5, "beta": 1.0}
    r = api.post("/adapters/flask/promote", json=body)
    assert r.status_code == 200, r.text
    assert _served_client_version(api) == 2


def test_d5_rollback_moves_the_served_composite_back(client):
    _promote_v2_with_composite(client)
    r = client.post("/adapters/flask/promote", json=_promote_body("flask", 2, good=False))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "rollback" and body["active_version_after"] == 1
    assert _active(client, "flask")["ref"]["version"] == 1
    assert _served_client_version(client) == 1
    assert body["composites"] == [{
        "name": COMPOSITE, "from_version": 2, "to_version": 1, "rebuilt": False,
        "composed_from": _active(client, COMPOSITE)["composed_from"],
    }]
    assert _active(client, COMPOSITE)["ref"]["version"] == 1  # reused, not rebuilt


def test_restore_moves_the_served_composite_back(client):
    _promote_v2_with_composite(client)
    r = client.post("/adapters/flask/restore", json={"reason": "bad adapter drill"})
    assert r.status_code == 200, r.text
    assert _served_client_version(client) == 1
    assert r.json()["composites"][0]["to_version"] == 1


def test_restore_rebuilds_a_composite_that_never_existed(client):
    """Composite only ever built from flask v2; restoring v1 builds the v1 composite."""
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client", cluster_id="web")
    assert _compose(client).json()["composed_from"]["client_version"] == 2
    r = client.post("/adapters/flask/restore", json={"reason": "drill"})
    assert r.status_code == 200, r.text
    moved = r.json()["composites"]
    assert moved[0]["rebuilt"] is True and moved[0]["to_version"] == 2
    meta = _active(client, COMPOSITE)
    assert meta["ref"]["version"] == 2 and meta["composed_from"]["client_version"] == 1

    sd = load(client.get(f"/adapters/{COMPOSITE}/versions/2/file").content)
    c, k = load(make_adapter(1)), load(make_adapter(2))  # cluster v1, flask v1
    key = "base_model.model.model.layers.1.self_attn.v_proj"
    got = sd[f"{key}.lora_B.weight"] @ sd[f"{key}.lora_A.weight"]
    want = (0.5 * c[f"{key}.lora_B.weight"] @ c[f"{key}.lora_A.weight"]
            + k[f"{key}.lora_B.weight"] @ k[f"{key}.lora_A.weight"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_cluster_restore_moves_the_composite_too(client):
    _seed(client)
    _save(client, "cluster-web", make_adapter(4), kind="cluster", aggregation="svd_exact",
          cluster_id="web", round=2, source_clients=["flask"])
    assert _compose(client).json()["composed_from"]["cluster_version"] == 2
    r = client.post("/adapters/cluster-web/restore", json={"reason": "bad aggregate"})
    assert r.status_code == 200, r.text
    assert _active(client, COMPOSITE)["composed_from"]["cluster_version"] == 1


def test_composite_pinned_to_another_version_is_left_alone(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")  # v2 active
    _compose(client, client={"name": "flask", "version": 1})
    r = client.post("/adapters/flask/restore", json={"reason": "drill"})
    assert r.status_code == 200 and r.json()["composites"] == []
    assert _active(client, COMPOSITE)["ref"]["version"] == 1


def test_cascade_is_in_the_composite_audit_trail_and_survives_gc(client):
    _promote_v2_with_composite(client)
    client.post("/adapters/flask/restore", json={"reason": "drill"})
    decisions = client.get(f"/adapters/{COMPOSITE}/promotions").json()["decisions"]
    assert decisions[-1]["action"] == "rollback"
    assert decisions[-1]["active_version_after"] == 1
    assert "cascade v2 -> v1" in decisions[-1]["reason"] and "drill" in decisions[-1]["reason"]
    gc = client.post(f"/adapters/{COMPOSITE}/gc", json={"keep_last": 1, "dry_run": False})
    assert gc.status_code == 200 and gc.json()["deleted"] == []


def test_unbuildable_replacement_refuses_and_changes_nothing(client):
    """flask v1 adapts only q_proj, so cluster(q,v) + flask v1 cannot be composed."""
    _save(client, "cluster-web", make_adapter(1), kind="cluster", aggregation="svd_exact",
          cluster_id="web", source_clients=["flask"])
    _save(client, "flask", make_adapter(2, modules=("q_proj",)), kind="client",
          hparams={"rank": 4, "lora_alpha": 4, "target_modules": ["q_proj"]})
    _save(client, "flask", make_adapter(3), kind="client")
    assert _compose(client).status_code == 201

    r = client.post("/adapters/flask/restore", json={"reason": "drill"})
    assert r.status_code == 409
    assert "cannot be built" in r.json()["detail"] and "different modules" in r.json()["detail"]
    assert _active(client, "flask")["ref"]["version"] == 2
    assert _served_client_version(client) == 2
    assert client.get("/adapters/flask/promotions").json()["decisions"] == []


def test_rollback_without_composites_reports_none(client):
    _save(client, "flask", make_adapter(2), kind="client")
    _save(client, "flask", make_adapter(3), kind="client")
    r = client.post("/adapters/flask/promote", json=_promote_body("flask", 2, good=False))
    assert r.status_code == 200 and r.json()["composites"] == []
