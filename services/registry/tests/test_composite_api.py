"""Composite storage over HTTP (D6): explicit compose + build-on-promote."""
from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import load, save

from .test_composite import MODULES, R, make_adapter
from .test_promote_api import _promote_body

HP = {"rank": R, "lora_alpha": R, "target_modules": list(MODULES)}


def _save(client, name, payload, **meta):
    r = client.post(
        f"/adapters/{name}/versions",
        files={"file": (f"{name}.safetensors", payload, "application/octet-stream")},
        data={"meta": json.dumps({"hparams": HP, **meta})},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _seed(client, *, cluster_eps=None, client_eps=None):
    _save(client, "cluster-web", make_adapter(1), kind="cluster", aggregation="svd_exact",
          cluster_id="web", round=1, source_clients=["flask", "requests"],
          privacy={"epsilon": cluster_eps} if cluster_eps is not None else None)
    _save(client, "flask", make_adapter(2), kind="client", cluster_id="web", round=1,
          privacy={"epsilon": client_eps} if client_eps is not None else None)


def _compose(api, name="composite-flask", **over):
    body = {"cluster": {"name": "cluster-web"}, "client": {"name": "flask"},
            "alpha": 0.5, "beta": 1.0, **over}
    return api.post(f"/adapters/{name}/compose", json=body)


# --------------------------------------------------------------------------- #
# POST /adapters/{name}/compose
# --------------------------------------------------------------------------- #
def test_compose_stores_a_composite_from_active_parts(client):
    _seed(client)
    r = _compose(client)
    assert r.status_code == 201, r.text
    meta = r.json()
    assert meta["ref"]["kind"] == "composite"
    assert meta["ref"]["cluster_id"] == "web"
    assert meta["composed_from"] == {
        "cluster_name": "cluster-web", "cluster_version": 1,
        "client_name": "flask", "client_version": 1,
        "alpha": 0.5, "beta": 1.0,
        "base_model": "deepseek-ai/deepseek-coder-1.3b-base",
    }
    assert meta["hparams"]["rank"] == 2 * R
    assert client.get("/adapters/composite-flask/active").json()["ref"]["version"] == 1


def test_composite_file_is_the_exact_weighted_sum(client):
    _seed(client)
    _compose(client, alpha=0.5, beta=1.0)
    blob = client.get("/adapters/composite-flask/versions/1/file").content
    sd, c, k = load(blob), load(make_adapter(1)), load(make_adapter(2))
    key = "base_model.model.model.layers.0.self_attn.q_proj"
    got = sd[f"{key}.lora_B.weight"] @ sd[f"{key}.lora_A.weight"]
    want = (0.5 * c[f"{key}.lora_B.weight"] @ c[f"{key}.lora_A.weight"]
            + k[f"{key}.lora_B.weight"] @ k[f"{key}.lora_A.weight"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_compose_pins_explicit_versions(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")  # v2 becomes active
    r = _compose(client, client={"name": "flask", "version": 1})
    assert r.json()["composed_from"]["client_version"] == 1


def test_identical_recompose_reuses_the_version(client):
    _seed(client)
    first = _compose(client).json()
    again = _compose(client)
    assert again.status_code == 200
    assert again.json()["ref"]["version"] == first["ref"]["version"]
    assert client.get("/adapters/composite-flask/versions").json()["versions"].__len__() == 1


def test_composite_epsilon_is_the_basic_composition_bound(client):
    _seed(client, cluster_eps=3.0, client_eps=2.5)
    assert _compose(client).json()["privacy"]["epsilon"] == 5.5


def test_composite_epsilon_is_none_if_any_part_is_not_private(client):
    _seed(client, cluster_eps=3.0)
    assert _compose(client).json()["privacy"]["epsilon"] is None


def test_compose_rejects_wrong_part_kinds(client):
    _seed(client)
    r = _compose(client, cluster={"name": "flask"}, client={"name": "cluster-web"})
    assert r.status_code == 422
    assert "kind" in r.json()["detail"]


def test_compose_unknown_part_404(client):
    _seed(client)
    assert _compose(client, cluster={"name": "cluster-ghost"}).status_code == 404


def test_compose_incompatible_parts_422(client):
    """A client whose model dims differ from the cluster's cannot merge exactly."""
    _seed(client)
    sd = {k: np.zeros((v.shape[0], v.shape[1] + 1), np.float32) if "lora_A" in k else v
          for k, v in load(make_adapter(6)).items()}
    _save(client, "broken", save(sd), kind="client")
    r = _compose(client, client={"name": "broken"})
    assert r.status_code == 422
    assert "shape" in r.json()["detail"]


def test_compose_validates_body(client):
    _seed(client)
    assert _compose(client, alpha="lots").status_code == 422
    assert client.post("/adapters/composite-flask/compose", json={}).status_code == 422


# --------------------------------------------------------------------------- #
# build-on-promote (D6: composite is built at promotion time)
# --------------------------------------------------------------------------- #
def _promote_with_composite(client, name, version, *, good):
    body = _promote_body(name, version, good=good)
    body["composite"] = {"name": "composite-flask", "cluster": "cluster-web",
                         "client": "flask", "alpha": 0.5, "beta": 1.0}
    return client.post(f"/adapters/{name}/promote", json=body)


def test_promote_builds_composite_from_the_promoted_client(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")  # candidate v2
    r = _promote_with_composite(client, "flask", 2, good=True)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "promote"
    assert body["composite"]["composed_from"]["client_version"] == 2
    assert body["composite"]["ref"]["name"] == "composite-flask"


def test_rollback_builds_no_composite(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")
    r = _promote_with_composite(client, "flask", 2, good=False)
    assert r.json()["action"] == "rollback"
    assert r.json()["composite"] is None
    assert client.get("/adapters/composite-flask/active").status_code == 404


def test_failed_composite_build_records_nothing(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")
    body = _promote_body("flask", 2, good=True)
    body["composite"] = {"name": "composite-flask", "cluster": "cluster-ghost",
                         "client": "flask", "alpha": 0.5, "beta": 1.0}
    r = client.post("/adapters/flask/promote", json=body)
    assert r.status_code == 404
    assert client.get("/adapters/flask/promotions").json()["decisions"] == []


def test_promote_without_composite_is_unchanged(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")
    r = client.post("/adapters/flask/promote", json=_promote_body("flask", 2, good=True))
    assert r.status_code == 200
    assert "composite" not in r.json()


def test_one_adapter_name_never_mixes_kinds(client):
    """Composing into an existing client's name would corrupt its history."""
    _seed(client)
    r = _compose(client, name="flask")
    assert r.status_code == 409
    assert "kind" in r.json()["detail"]
    assert client.get("/adapters/flask/active").json()["ref"]["kind"] == "client"


def test_promote_with_composite_into_wrong_kind_records_nothing(client):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client")
    body = _promote_body("flask", 2, good=True)
    body["composite"] = {"name": "cluster-web", "cluster": "cluster-web",
                         "client": "flask", "alpha": 0.5, "beta": 1.0}
    assert client.post("/adapters/flask/promote", json=body).status_code == 409
    assert client.get("/adapters/flask/promotions").json()["decisions"] == []
