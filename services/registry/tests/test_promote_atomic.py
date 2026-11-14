"""A promotion (and an operator restore) commits as one transaction; orphans are swept.

Composite files are written before the transaction and registered inside it,
so a failure part-way leaves at most an unregistered directory: never served,
removed by ``retention.sweep_orphans``.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest
from contracts import AdapterKind, LoRAHyperParams

from .test_cascade import COMPOSITE, _promote_v2_with_composite
from .test_composite import _hp, make_adapter
from .test_composite_api import _compose, _save, _seed
from .test_promote_api import _promote_body

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(client):
    """The conftest client, but a server-side crash comes back as a 500."""
    from fastapi.testclient import TestClient

    import registry.app as appmod

    return TestClient(appmod.app, raise_server_exceptions=False)


@pytest.fixture
def store(client):
    """The store the app serves (same data root as ``client``)."""
    import registry.app as appmod

    return appmod.get_store()


def _boom(*_a, **_k):
    raise RuntimeError("injected failure")


def _fail_for(name, real):
    def wrapped(n, *a, **k):
        if n == name:
            raise RuntimeError("injected failure")
        return real(n, *a, **k)
    return wrapped


def _composite_body():
    body = _promote_body("flask", 2, good=True)
    body["composite"] = {"name": COMPOSITE, "cluster": "cluster-web", "client": "flask",
                         "alpha": 0.5, "beta": 1.0}
    return body


def _state(store):
    return {
        "flask_active": store.get_active("flask"),
        "composite_active": store.get_active(COMPOSITE),
        "flask_decisions": len(store.list_promotions("flask")),
        "composite_decisions": len(store.list_promotions(COMPOSITE)),
        "composite_versions": store.list_versions(COMPOSITE),
        "sync": store.repo.pending_cluster_sync(),
    }


# --------------------------------------------------------------------------- #
# promote
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fail_at", ["set_active", "record_decision", "add_cluster_sync"])
def test_failure_inside_the_transaction_leaves_no_trace(client, store, monkeypatch, fail_at):
    from registry.retention import sweep_orphans

    _seed(client)
    assert _compose(client).status_code == 201  # composite-flask v1 serves flask v1
    _save(client, "flask", make_adapter(3), kind="client", cluster_id="web", set_active=False)
    before = _state(store)

    monkeypatch.setattr(store.repo, fail_at, _boom)
    r = client.post("/adapters/flask/promote", json=_composite_body())
    assert r.status_code == 500
    monkeypatch.undo()

    assert _state(store) == before  # pointer, decision, composite row, sync row: none
    orphan = store.adapters_dir / COMPOSITE / "v2"
    assert orphan.is_dir()  # the files were written before the transaction ...
    assert client.get(f"/adapters/{COMPOSITE}/versions/2/file").status_code == 404  # ... never served
    assert sweep_orphans(store) == [str(orphan)]
    assert not orphan.exists()

    # and the same promotion now goes through, under the same version number
    r = client.post("/adapters/flask/promote", json=_composite_body())
    assert r.status_code == 200, r.text
    assert r.json()["composite"]["ref"]["version"] == 2


@pytest.mark.parametrize("good,action", [(True, "promote"), (False, "reject")])
def test_a_decision_queues_a_cluster_sync_row_with_it(client, store, good, action):
    _seed(client)
    _save(client, "flask", make_adapter(3), kind="client", cluster_id="web", set_active=False)
    r = client.post("/adapters/flask/promote", json=_promote_body("flask", 2, good=good))
    assert r.status_code == 200 and r.json()["action"] == action
    [row] = store.repo.pending_cluster_sync()
    assert (row["cluster_id"], row["sha256"], row["action"]) == (
        "web", store.get_metadata("flask", 2).sha256, action)


def test_an_adapter_without_a_cluster_queues_no_sync(client, store):
    _save(client, "solo", make_adapter(3), kind="client", set_active=False)
    assert client.post("/adapters/solo/promote",
                       json=_promote_body("solo", 1, good=True)).status_code == 200
    assert store.repo.pending_cluster_sync() == []


def test_failure_after_files_before_commit_is_cleaned_by_the_sweep(store, monkeypatch):
    from registry.composition import (
        parse_compose_request,
        plan_composite,
        register_composite,
        store_composite,
        write_composite_files,
    )
    from registry.retention import sweep_orphans
    from registry.storage import AdapterNotFound

    hp = _hp()
    store.save("cluster-web", make_adapter(1), kind=AdapterKind.CLUSTER, hparams=hp,
               cluster_id="web", set_active=True)
    store.save("flask", make_adapter(2), kind=AdapterKind.CLIENT, hparams=hp,
               cluster_id="web", set_active=True)
    req = parse_compose_request({"cluster": "cluster-web", "client": "flask",
                                 "alpha": 0.5, "beta": 1.0})
    plan = plan_composite(store, req, COMPOSITE)

    meta = write_composite_files(store, COMPOSITE, plan)
    vdir = store.adapters_dir / COMPOSITE / "v1"
    assert (vdir / "adapter.safetensors").read_bytes() == plan.payload
    assert store.list_versions(COMPOSITE) == []  # files only, no row

    monkeypatch.setattr(store.repo, "add_version", _boom)
    with pytest.raises(RuntimeError):
        with store.repo.transaction():
            register_composite(store.repo, meta)
    monkeypatch.undo()

    assert store.list_versions(COMPOSITE) == [] and store.get_active(COMPOSITE) is None
    with pytest.raises(AdapterNotFound):
        store.load_payload(COMPOSITE, 1)
    assert sweep_orphans(store) == [str(vdir)]
    assert not vdir.exists()
    stored, created = store_composite(store, COMPOSITE, plan)  # v1 is free again
    assert created and stored.ref.version == 1


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_sweep_never_removes_a_registered_version_or_a_fresh_staging_dir(store, safetensors_blob):
    from registry.retention import sweep_orphans

    store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    adir = store.adapters_dir / "flask"
    fresh_tmp = adir / ".tmp-v2-fresh"
    fresh_tmp.mkdir()
    fresh_trash = adir / ".trash-v1-fresh"
    fresh_trash.mkdir()
    old_tmp = adir / ".tmp-v2-old"
    old_tmp.mkdir()
    (old_tmp / "adapter.safetensors").write_bytes(safetensors_blob)
    _age(old_tmp, 120)
    old_trash = adir / ".trash-v1-old"
    old_trash.mkdir()
    _age(old_trash, 120)
    orphan = adir / "v7"
    orphan.mkdir()
    (orphan / "adapter.safetensors").write_bytes(safetensors_blob)  # brand new, still swept
    (adir / "gc.jsonl").write_text("{}\n")

    removed = sweep_orphans(store)

    assert sorted(removed) == sorted(str(p) for p in (old_tmp, old_trash, orphan))
    assert (adir / "v1" / "adapter.safetensors").exists()
    assert store.load_payload("flask", 1) == safetensors_blob
    assert fresh_tmp.is_dir() and fresh_trash.is_dir() and (adir / "gc.jsonl").exists()
    assert sweep_orphans(store) == []


def test_sweep_removes_a_directory_a_failed_delete_left_behind(store, safetensors_blob):
    """``delete_version`` drops the row first; if the directory removal never
    happens (crash), the directory has no row and the sweep takes it."""
    from registry.retention import sweep_orphans

    for _ in range(2):
        store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    store.repo.delete_version("flask", 1)  # the row goes; the crash keeps the files
    v1 = store.adapters_dir / "flask" / "v1"
    assert v1.is_dir()
    assert sweep_orphans(store) == [str(v1)]
    assert store.list_versions("flask") == [2]


def test_a_directory_without_a_row_is_never_served(client, store, safetensors_blob):
    store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    shutil.copytree(store.adapters_dir / "flask" / "v1", store.adapters_dir / "flask" / "v2")
    assert client.get("/adapters/flask/versions/2/file").status_code == 404
    assert client.get("/adapters/flask/versions/1/file").content == safetensors_blob


def test_the_sweep_runs_when_the_store_is_created_and_at_every_gc(client, safetensors_blob):
    import registry.app as appmod

    from .conftest import _drop_app_store

    _save(client, "flask", safetensors_blob, kind="client")
    root = Path(os.environ["CLASP_REGISTRY_DATA"]) / "adapters" / "flask"
    (root / "v5").mkdir()
    _drop_app_store(appmod)
    assert client.get("/adapters/flask/versions").status_code == 200  # store created here
    assert not (root / "v5").exists()

    for route in ("/adapters/flask/gc", "/gc"):
        (root / "v5").mkdir()
        assert client.post(route, json={"dry_run": True}).status_code == 200
        assert not (root / "v5").exists(), route


def test_gc_validation_errors_sweep_nothing(client, safetensors_blob):
    _save(client, "flask", safetensors_blob, kind="client")
    orphan = Path(os.environ["CLASP_REGISTRY_DATA"]) / "adapters" / "flask" / "v5"
    orphan.mkdir()
    assert client.post("/adapters/flask/gc", json={"keep_last": 0}).status_code == 422
    assert client.post("/adapters/ghost/gc", json={}).status_code == 404
    assert client.post("/gc", json={"dry_run": "yes"}).status_code == 422
    assert orphan.is_dir()


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rebuild", [False, True])
def test_restore_moves_pointer_decision_and_cascade_together(client, store, monkeypatch, rebuild):
    from registry.retention import sweep_orphans

    if rebuild:  # the composite was only ever built from flask v2: v1's must be built
        _seed(client)
        _save(client, "flask", make_adapter(3), kind="client", cluster_id="web")
        assert _compose(client).json()["composed_from"]["client_version"] == 2
    else:  # composite v1 (flask v1) exists and is reused
        _promote_v2_with_composite(client)
    before = _state(store)

    real = store.repo.record_decision
    monkeypatch.setattr(store.repo, "record_decision", _fail_for(COMPOSITE, real))
    r = client.post("/adapters/flask/restore", json={"reason": "drill", "to_version": 1})
    assert r.status_code == 500
    monkeypatch.undo()

    assert _state(store) == before  # flask pointer + decision rolled back with the cascade
    swept = sweep_orphans(store)
    assert swept == ([str(store.adapters_dir / COMPOSITE / "v2")] if rebuild else [])

    r = client.post("/adapters/flask/restore", json={"reason": "drill", "to_version": 1})
    assert r.status_code == 200, r.text
    assert store.get_active("flask") == 1
    served = client.get(f"/adapters/{COMPOSITE}/active").json()
    assert served["composed_from"]["client_version"] == 1
    assert len(store.list_promotions("flask")) == before["flask_decisions"] + 1
    assert len(store.list_promotions(COMPOSITE)) == before["composite_decisions"] + 1


# --------------------------------------------------------------------------- #
# a legacy volume, read over HTTP (plan Review Focus 1)
# --------------------------------------------------------------------------- #
def test_legacy_volume_reads_back_identically_over_the_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import registry.app as appmod

    from .conftest import _drop_app_store

    expected = json.loads((FIXTURES / "legacy_volume.expected.json").read_text())
    root = tmp_path / "volume"
    shutil.copytree(FIXTURES / "legacy_volume", root)
    assert not (root / "registry.db").exists()
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(root))
    monkeypatch.delenv("CLASP_REGISTRY_DB", raising=False)
    _drop_app_store(appmod)
    try:
        api = TestClient(appmod.app)
        assert api.get("/adapters").json() == {"adapters": expected["adapters"]}
        for name in expected["adapters"]:
            metas = expected["metadata"][name]
            active = expected["active"][name]
            assert api.get(f"/adapters/{name}/versions").json() == {
                "name": name, "active": active,
                "versions": [metas[str(v)] for v in expected["versions"][name]],
            }
            for v in expected["versions"][name]:
                assert api.get(f"/adapters/{name}/versions/{v}").json() == metas[str(v)]
                assert api.get(f"/adapters/{name}/versions/{v}/file").content == (
                    root / "adapters" / name / f"v{v}" / "adapter.safetensors").read_bytes()
            assert api.get(f"/adapters/{name}/active").json() == metas[str(active)]
            assert api.get(f"/adapters/{name}/promotions").json() == {
                "name": name, "decisions": expected["promotions"][name]}
            assert api.get(f"/adapters/{name}/lineage").json() == expected["lineage"][name]
    finally:
        _drop_app_store(appmod)
