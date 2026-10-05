"""Retention / GC policy: keep the last N + everything that was ever live.

A version is protected if ANY of these hold — the reasons are returned so an
operator can see why each survivor survived:

    active      it is the active version right now
    recent      it is among the newest ``keep_last`` versions
    promoted    a D5 decision ever promoted it
    restored    a rollback/restore ever made it active
    composite   some composite's provenance names it (deleting would orphan it)
"""
from __future__ import annotations

import json

import pytest
from contracts import (
    AdapterKind,
    AdapterRef,
    CompositeProvenance,
    LoRAHyperParams,
    PromotionAction,
    PromotionDecision,
)
from registry.retention import plan_retention

from .conftest import make_safetensors


def _dec(version: int, action: PromotionAction, after: int) -> PromotionDecision:
    return PromotionDecision(AdapterRef("flask", version, AdapterKind.CLIENT), action, after, "r")


# --------------------------------------------------------------------------- #
# the pure policy
# --------------------------------------------------------------------------- #
def test_keeps_last_n_and_active():
    plan = plan_retention(range(1, 8), active=7, decisions=(), referenced=set(), keep_last=3)
    assert plan.delete == (1, 2, 3, 4)
    assert set(plan.keep) == {5, 6, 7}
    assert "active" in plan.keep[7] and "recent" in plan.keep[7]


def test_never_deletes_promoted_restored_or_composite_parents():
    decisions = (
        _dec(2, PromotionAction.PROMOTE, 2),   # 2 was promoted
        _dec(5, PromotionAction.ROLLBACK, 3),  # 3 was made active by a rollback
    )
    plan = plan_retention(range(1, 9), active=8, decisions=decisions,
                          referenced={4}, keep_last=2)
    assert plan.delete == (1, 5, 6)
    assert plan.keep[2] == ("promoted",)
    assert plan.keep[3] == ("restored",)
    assert plan.keep[4] == ("composite",)


def test_active_survives_even_when_old():
    plan = plan_retention(range(1, 6), active=1, decisions=(), referenced=set(), keep_last=1)
    assert plan.delete == (2, 3, 4)
    assert set(plan.keep) == {1, 5}


@pytest.mark.parametrize("bad", [0, -1])
def test_keep_last_must_be_positive(bad):
    with pytest.raises(ValueError):
        plan_retention([1], active=1, decisions=(), referenced=set(), keep_last=bad)


# --------------------------------------------------------------------------- #
# storage + API
# --------------------------------------------------------------------------- #
def _upload(client, name, n=1, meta=None):
    for _ in range(n):
        r = client.post(
            f"/adapters/{name}/versions",
            files={"file": ("a.safetensors", make_safetensors(), "application/octet-stream")},
            data={"meta": json.dumps(meta or {"kind": "client"})},
        )
        assert r.status_code == 201, r.text


def test_gc_is_a_dry_run_by_default(client):
    _upload(client, "flask", 6)
    r = client.post("/adapters/flask/gc", json={"keep_last": 2})
    assert r.status_code == 200
    assert r.json()["dry_run"] is True
    assert r.json()["deleted"] == [1, 2, 3, 4]
    assert len(client.get("/adapters/flask/versions").json()["versions"]) == 6


def test_gc_deletes_and_keeps_numbering_monotonic(client):
    _upload(client, "flask", 6)
    r = client.post("/adapters/flask/gc", json={"keep_last": 2, "dry_run": False})
    assert r.json()["deleted"] == [1, 2, 3, 4]
    versions = client.get("/adapters/flask/versions").json()["versions"]
    assert [v["ref"]["version"] for v in versions] == [5, 6]
    _upload(client, "flask")  # the next save never reuses a deleted number
    assert client.get("/adapters/flask/active").json()["ref"]["version"] == 7
    assert client.get("/adapters/flask/versions/2").status_code == 404


def test_gc_protects_versions_named_by_any_composite(store, safetensors_blob):
    from registry.retention import collect_referenced, gc_adapter

    for _ in range(4):
        store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    store.save("composite-flask", safetensors_blob, kind=AdapterKind.COMPOSITE,
               hparams=LoRAHyperParams(),
               composed_from=CompositeProvenance("cluster-web", 1, "flask", 1, 0.5, 1.0))
    assert collect_referenced(store) == {("cluster-web", 1), ("flask", 1)}
    result = gc_adapter(store, "flask", keep_last=1, dry_run=False)
    assert result.delete == (2, 3)
    assert store.list_versions("flask") == [1, 4]


def test_gc_writes_an_audit_record_and_leaves_no_trash(client, tmp_path):
    _upload(client, "flask", 3)
    client.post("/adapters/flask/gc", json={"keep_last": 1, "dry_run": False})
    root = tmp_path / "registry_api" / "adapters" / "flask"
    log = [json.loads(line) for line in (root / "gc.jsonl").read_text().splitlines()]
    assert log[0]["deleted"] == [1, 2]
    assert not [p for p in root.iterdir() if p.name.startswith(".trash")]


def test_gc_all_adapters(client):
    _upload(client, "flask", 3)
    _upload(client, "cluster-web", 3, {"kind": "cluster"})
    r = client.post("/gc", json={"keep_last": 1, "dry_run": False})
    assert r.status_code == 200
    assert {a["name"]: a["deleted"] for a in r.json()["adapters"]} == {
        "cluster-web": [1, 2], "flask": [1, 2],
    }


def test_gc_defaults_keep_last_from_env(client, monkeypatch):
    monkeypatch.setenv("CLASP_REGISTRY_KEEP_LAST", "4")
    _upload(client, "flask", 6)
    assert client.post("/adapters/flask/gc", json={}).json()["deleted"] == [1, 2]


@pytest.mark.parametrize("body", [{"keep_last": 0}, {"keep_last": "x"}, {"dry_run": "no"}])
def test_gc_validates_body(client, body):
    _upload(client, "flask", 2)
    assert client.post("/adapters/flask/gc", json=body).status_code == 422


def test_gc_unknown_adapter_404(client):
    assert client.post("/adapters/ghost/gc", json={}).status_code == 404


def test_concurrent_writes_are_serialized(client):
    """Saves, restores and GC race in FastAPI's threadpool; none may 409/500."""
    from concurrent.futures import ThreadPoolExecutor

    def save(_):
        return client.post(
            "/adapters/flask/versions",
            files={"file": ("a.safetensors", make_safetensors(), "application/octet-stream")},
            data={"meta": json.dumps({"kind": "client"})},
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(save, range(24)))
    assert codes == [201] * 24
    versions = [v["ref"]["version"] for v in client.get("/adapters/flask/versions").json()["versions"]]
    assert versions == list(range(1, 25))

    def churn(i):
        if i % 2:
            return client.post("/adapters/flask/gc", json={"keep_last": 3, "dry_run": False}).status_code
        return client.post("/adapters/flask/restore", json={"reason": f"race {i}"}).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(churn, range(16)))
    assert all(c in (200, 409) for c in codes), codes
    active = client.get("/adapters/flask/active")
    assert active.status_code == 200
    file = client.get(f"/adapters/flask/versions/{active.json()['ref']['version']}/file")
    assert file.status_code == 200
