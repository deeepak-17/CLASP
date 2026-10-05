"""Operator restore (rollback drill) + the D11 restore-to-previous <= 10 s NFR."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from safetensors.numpy import save

from .conftest import make_safetensors

#: D11: registry restore-to-previous must complete within 10 s.
RESTORE_NFR_SECONDS = 10.0


def _upload(client, name, blob=None, kind="client"):
    r = client.post(
        f"/adapters/{name}/versions",
        files={"file": (f"{name}.safetensors", blob or make_safetensors(), "application/octet-stream")},
        data={"meta": json.dumps({"kind": kind})},
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_restore_defaults_to_previous_version(client):
    _upload(client, "flask")
    _upload(client, "flask")
    r = client.post("/adapters/flask/restore", json={"reason": "bad adapter drill"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "rollback"
    assert body["active_version_after"] == 1
    assert body["adapter"]["version"] == 2
    assert "operator restore" in body["reason"] and "bad adapter drill" in body["reason"]
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 1


def test_restore_to_explicit_version_and_forward(client):
    for _ in range(3):
        _upload(client, "flask")
    assert client.post("/adapters/flask/restore",
                       json={"to_version": 1, "reason": "x"}).json()["active_version_after"] == 1
    fwd = client.post("/adapters/flask/restore", json={"to_version": 3, "reason": "fixed"})
    assert fwd.json()["action"] == "promote"
    assert "not a D5 decision" in fwd.json()["reason"]
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 3


def test_restore_is_recorded_in_the_audit_trail(client):
    _upload(client, "flask")
    _upload(client, "flask")
    client.post("/adapters/flask/restore", json={"reason": "drill"})
    decisions = client.get("/adapters/flask/promotions").json()["decisions"]
    assert len(decisions) == 1 and decisions[0]["active_version_after"] == 1


@pytest.mark.parametrize("body,status", [
    ({}, 422),                                    # reason is required
    ({"reason": ""}, 422),
    ({"reason": "x", "to_version": "two"}, 422),
    ({"reason": "x", "to_version": 9}, 404),      # unknown version
    ({"reason": "x", "to_version": 2}, 409),      # already active
])
def test_restore_validation(client, body, status):
    _upload(client, "flask")
    _upload(client, "flask")
    assert client.post("/adapters/flask/restore", json=body).status_code == status


def test_restore_with_no_previous_version_is_409(client):
    _upload(client, "flask")
    assert client.post("/adapters/flask/restore", json={"reason": "x"}).status_code == 409


def test_restore_unknown_adapter_404(client):
    assert client.post("/adapters/ghost/restore", json={"reason": "x"}).status_code == 404


def _realistic_adapter(seed: int) -> bytes:
    """~24 MB: 24 layers x q/k/v/o, rank 16, hidden 2048 — a 1.3B-class adapter."""
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(24):
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"base_model.model.model.layers.{layer}.self_attn.{m}"
            sd[f"{key}.lora_A.weight"] = rng.standard_normal((16, 2048), dtype=np.float32)
            sd[f"{key}.lora_B.weight"] = rng.standard_normal((2048, 16), dtype=np.float32)
    return save(sd)


def test_restore_nfr_under_10_seconds_on_a_real_sized_adapter(client):
    """D11: restore-to-previous, measured as the edge sees it — repoint the
    pointer, then fetch the active metadata and the full payload back."""
    good, bad = _realistic_adapter(1), _realistic_adapter(2)
    _upload(client, "flask", good)
    _upload(client, "flask", bad)

    start = time.perf_counter()
    client.post("/adapters/flask/restore", json={"reason": "nfr"}).raise_for_status()
    active = client.get("/adapters/flask/active").json()
    payload = client.get(f"/adapters/flask/versions/{active['ref']['version']}/file").content
    elapsed = time.perf_counter() - start

    assert payload == good
    assert elapsed < RESTORE_NFR_SECONDS, f"restore took {elapsed:.2f}s"
