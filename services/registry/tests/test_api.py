"""Integration tests across all registry endpoints (P4, W3 Thu)."""
from __future__ import annotations

import json

from .conftest import make_safetensors


def _upload(client, name, blob, meta):
    return client.post(
        f"/adapters/{name}/versions",
        files={"file": (f"{name}.safetensors", blob, "application/octet-stream")},
        data={"meta": json.dumps(meta)},
    )


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["contracts"] == "1.0.0"


def test_empty_registry_lists_nothing(client):
    assert client.get("/adapters").json() == {"adapters": []}


def test_save_then_list_and_get(client, safetensors_blob):
    r = _upload(client, "django", safetensors_blob, {"kind": "client", "cluster_id": "web", "seed": 7})
    assert r.status_code == 201
    body = r.json()
    assert body["ref"]["version"] == 1
    assert body["ref"]["kind"] == "client"
    assert body["seed"] == 7
    assert len(body["sha256"]) == 64

    assert client.get("/adapters").json()["adapters"] == ["django"]

    versions = client.get("/adapters/django/versions").json()
    assert versions["active"] == 1
    assert len(versions["versions"]) == 1

    one = client.get("/adapters/django/versions/1").json()
    assert one["ref"]["name"] == "django"


def test_versioning_increments(client, safetensors_blob):
    _upload(client, "flask", safetensors_blob, {"kind": "client"})
    r2 = _upload(client, "flask", safetensors_blob, {"kind": "client"})
    assert r2.json()["ref"]["version"] == 2
    assert [v["ref"]["version"] for v in client.get("/adapters/flask/versions").json()["versions"]] == [1, 2]


def test_download_roundtrip(client, safetensors_blob):
    _upload(client, "requests", safetensors_blob, {"kind": "client"})
    r = client.get("/adapters/requests/versions/1/file")
    assert r.status_code == 200
    assert r.content == safetensors_blob


def test_cluster_snapshot_metadata(client, safetensors_blob):
    meta = {
        "kind": "cluster",
        "cluster_id": "web",
        "round": 3,
        "aggregation": "svd_exact",
        "privacy": {"epsilon": 7.5, "delta": 1e-5},
        "hparams": {"rank": 16, "alpha": 0.8, "beta": 0.5},
        "source_clients": ["django", "flask", "requests"],
    }
    r = _upload(client, "web-cluster", safetensors_blob, meta)
    body = r.json()
    assert body["aggregation"] == "svd_exact"
    assert body["privacy"]["epsilon"] == 7.5
    assert body["hparams"]["alpha"] == 0.8
    assert body["source_clients"] == ["django", "flask", "requests"]


def test_active_endpoint(client, safetensors_blob):
    _upload(client, "numpy", safetensors_blob, {"kind": "client"})
    _upload(client, "numpy", safetensors_blob, {"kind": "client"})
    active = client.get("/adapters/numpy/active").json()
    assert active["ref"]["version"] == 2


def test_rejects_bad_payload(client):
    r = _upload(client, "bad", b"garbage-not-safetensors", {"kind": "client"})
    assert r.status_code == 422


def test_missing_adapter_404(client):
    assert client.get("/adapters/ghost/versions").status_code == 404
    assert client.get("/adapters/ghost/versions/1").status_code == 404
    assert client.get("/adapters/ghost/active").status_code == 404


def test_active_endpoint_corrupt_pointer_is_a_clean_500(client, safetensors_blob):
    """L1: a hand-corrupted `active` file must surface as a structured 500,
    not an unhandled StorageError bubbling out of the handler."""
    import os
    from pathlib import Path

    _upload(client, "corrupt", safetensors_blob, {"kind": "client"})
    active_path = Path(os.environ["CLASP_REGISTRY_DATA"]) / "adapters" / "corrupt" / "active"
    active_path.write_text("not-an-int")

    r = client.get("/adapters/corrupt/active")
    assert r.status_code == 500
    assert "corrupt" in r.json()["detail"].lower()


def test_manifest_roundtrip(tmp_path):
    from registry.manifest import build_manifest, config_hash, read_manifest, write_manifest

    cfg = {"seed": 0, "rank": 16, "rounds": 5}
    assert config_hash(cfg) == config_hash({"rounds": 5, "rank": 16, "seed": 0})  # order-insensitive
    man = build_manifest("run-001", 0, cfg, {"django": 2}, gpu_hours_estimate=3.5)
    p = write_manifest(man, tmp_path / "manifest.json")
    back = read_manifest(p)
    assert back.run_id == "run-001"
    assert back.adapter_versions == {"django": 2}
    assert back.gpu_hours_estimate == 3.5


# make_safetensors imported so conftest helper is reachable from tests
_ = make_safetensors
