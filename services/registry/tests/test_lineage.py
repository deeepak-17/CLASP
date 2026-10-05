"""Composite metadata (contracts v1.1), lineage view, and clean error mapping."""
from __future__ import annotations

import json

import pytest
from contracts import AdapterKind, CompositeProvenance, LoRAHyperParams
from registry.storage import StorageError

from .conftest import make_safetensors
from .test_promote_api import _promote_body

PROV = CompositeProvenance("cluster-web", 1, "flask", 2, alpha=0.5, beta=1.0)


def _upload(client, name, meta=None):
    return client.post(
        f"/adapters/{name}/versions",
        files={"file": (f"{name}.safetensors", make_safetensors(), "application/octet-stream")},
        data={"meta": json.dumps(meta or {"kind": "client"})},
    )


# --------------------------------------------------------------------------- #
# storage: composite provenance
# --------------------------------------------------------------------------- #
def test_store_round_trips_composed_from(store, safetensors_blob):
    m = store.save(
        "composite-flask", safetensors_blob, kind=AdapterKind.COMPOSITE,
        hparams=LoRAHyperParams(rank=32), composed_from=PROV,
    )
    back = store.get_metadata("composite-flask", m.ref.version)
    assert back.ref.kind is AdapterKind.COMPOSITE
    assert back.composed_from == PROV


def test_composite_kind_requires_provenance(store, safetensors_blob):
    with pytest.raises(StorageError, match="composed_from"):
        store.save("composite-x", safetensors_blob, kind=AdapterKind.COMPOSITE,
                   hparams=LoRAHyperParams())


def test_provenance_only_allowed_on_composites(store, safetensors_blob):
    with pytest.raises(StorageError, match="composed_from"):
        store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT,
                   hparams=LoRAHyperParams(), composed_from=PROV)


def test_v1_0_metadata_on_disk_still_reads(store, safetensors_blob):
    """A metadata.json written before v1.1 has no composed_from key."""
    store.save("legacy", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    meta_path = store._adapter_dir("legacy") / "v1" / "metadata.json"
    d = json.loads(meta_path.read_text())
    d.pop("composed_from", None)
    d["contracts_version"] = "1.0.0"
    meta_path.write_text(json.dumps(d))
    back = store.get_metadata("legacy", 1)
    assert back.composed_from is None
    assert back.contracts_version == "1.0.0"


# --------------------------------------------------------------------------- #
# API: composite save + lineage
# --------------------------------------------------------------------------- #
def test_api_saves_composite_with_provenance(client):
    r = _upload(client, "composite-flask", {
        "kind": "composite", "hparams": {"rank": 32},
        "composed_from": PROV.to_json(),
    })
    assert r.status_code == 201, r.text
    assert r.json()["composed_from"]["cluster_version"] == 1
    assert r.json()["ref"]["kind"] == "composite"


def test_api_rejects_composite_without_provenance(client):
    r = _upload(client, "composite-flask", {"kind": "composite"})
    assert r.status_code == 422
    assert "composed_from" in r.json()["detail"]


def test_api_rejects_malformed_provenance(client):
    r = _upload(client, "composite-flask", {"kind": "composite", "composed_from": {"x": 1}})
    assert r.status_code == 422


def test_lineage_lists_versions_parents_and_decisions(client):
    _upload(client, "cluster-web", {
        "kind": "cluster", "aggregation": "svd_exact", "round": 1,
        "cluster_id": "web", "source_clients": ["flask", "requests", "werkzeug"],
    })
    _upload(client, "composite-flask", {"kind": "composite", "composed_from": PROV.to_json()})

    r = client.get("/adapters/cluster-web/lineage")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "cluster-web"
    assert body["active"] == 1
    v1 = body["versions"][0]
    assert v1["version"] == 1 and v1["is_active"] is True
    assert v1["parents"] == ["client:flask", "client:requests", "client:werkzeug"]
    assert len(v1["sha256"]) == 64

    comp = client.get("/adapters/composite-flask/lineage").json()["versions"][0]
    assert comp["parents"] == ["cluster-web@v1", "flask@v2"]
    assert comp["composed_from"]["alpha"] == 0.5


def test_lineage_attaches_promotion_events(client):
    _upload(client, "flask")
    _upload(client, "flask")
    r = client.post("/adapters/flask/promote", json=_promote_body("flask", 2, good=False))
    assert r.json()["action"] == "rollback"
    versions = client.get("/adapters/flask/lineage").json()["versions"]
    assert [e["action"] for e in versions[1]["decisions"]] == ["rollback"]
    assert versions[0]["is_active"] is True and versions[1]["is_active"] is False


def test_lineage_unknown_adapter_404(client):
    assert client.get("/adapters/nope/lineage").status_code == 404


# --------------------------------------------------------------------------- #
# error mapping: StorageError never escapes as an unhandled 500
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [
    "/adapters/bad name!/versions",
    "/adapters/bad name!/versions/1",
    "/adapters/bad name!/versions/1/file",
    "/adapters/bad name!/active",
    "/adapters/bad name!/lineage",
])
def test_invalid_name_is_422_not_500(client, path):
    assert client.get(path).status_code == 422


def test_unknown_version_404(client):
    _upload(client, "flask")
    assert client.get("/adapters/flask/versions/9").status_code == 404
    assert client.get("/adapters/flask/versions/9/file").status_code == 404
