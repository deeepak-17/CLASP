"""A volume written by the file-based registry is imported into SQLite on first start.

``fixtures/legacy_volume`` was written by the pre-database ``RegistryStore``
(see ``fixtures/make_legacy_volume.py``); ``legacy_volume.expected.json`` is
what that store answered for every read.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from contracts import AdapterKind, AdapterRef, LoRAHyperParams
from registry import db
from registry.lineage import build_lineage
from registry.migrate import import_volume
from registry.repository import SqlMetadataRepository
from registry.storage import RegistryStore, StorageError

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIXTURES / "legacy_volume.expected.json").read_text())
COUNTS = {"adapters": 3, "versions": 4, "pointers": 3, "decisions": 2}


@pytest.fixture
def volume(tmp_path, monkeypatch):
    monkeypatch.delenv(db.DB_ENV, raising=False)
    root = tmp_path / "volume"
    shutil.copytree(FIXTURES / "legacy_volume", root)
    return root


@pytest.fixture
def open_store():
    stores = []

    def _open(root):
        stores.append(RegistryStore(root))
        return stores[-1]

    yield _open
    for s in stores:
        s.close()


def _repo(root):
    engine = db.make_engine(f"sqlite:///{(root / 'registry.db').as_posix()}")
    db.create_all(engine)
    return engine, SqlMetadataRepository(engine)


def test_volume_written_by_the_old_registry_reads_back_identically(volume, open_store):
    assert not (volume / "registry.db").exists()
    store = open_store(volume)

    assert store.list_adapters() == EXPECTED["adapters"]
    for name in EXPECTED["adapters"]:
        assert store.list_versions(name) == EXPECTED["versions"][name]
        for v in store.list_versions(name):
            assert store.get_metadata(name, v).to_json() == EXPECTED["metadata"][name][str(v)]
        assert store.get_active(name) == EXPECTED["active"][name]
        assert [d.to_json() for d in store.list_promotions(name)] == EXPECTED["promotions"][name]
        assert json.loads(json.dumps(build_lineage(store, name))) == EXPECTED["lineage"][name]
    # the composite's parts are rows too, so retention and cascade can see them
    assert store.repo.referenced_versions("flask") == {2}
    assert store.repo.referenced_versions("cluster-web") == {1}
    assert store.repo.composites_built_from("flask", 2) == [
        AdapterRef("composite-flask", 1, AdapterKind.COMPOSITE, cluster_id="web")]
    # weights stay where they were
    assert store.load_payload("flask", 2) == (
        volume / "adapters" / "flask" / "v2" / "adapter.safetensors").read_bytes()


def test_import_is_idempotent_and_skipped_when_the_db_has_data(volume, open_store):
    engine, repo = _repo(volume)
    try:
        assert import_volume(volume, repo) == COUNTS
        assert import_volume(volume, repo) == dict.fromkeys(COUNTS, 0)
        assert repo.list_versions("flask") == [1, 2]
        assert len(repo.list_decisions("flask")) == 2
    finally:
        engine.dispose()

    # the store sees a populated database and does not import again: a version
    # directory that appears on disk afterwards is not picked up
    shutil.copytree(volume / "adapters" / "flask" / "v2", volume / "adapters" / "flask" / "v3")
    store = open_store(volume)
    assert store.list_versions("flask") == [1, 2]
    assert len(store.list_promotions("flask")) == 2


def test_corrupt_metadata_json_fails_the_import_and_leaves_the_db_empty(volume, open_store):
    bad = volume / "adapters" / "flask" / "v2" / "metadata.json"
    bad.write_text("{ not json")

    with pytest.raises(StorageError, match=r"flask[\\/]+v2[\\/]+metadata\.json"):
        open_store(volume)

    engine, repo = _repo(volume)
    try:
        assert repo.list_adapters() == []
        assert repo.list_decisions("flask") == []
    finally:
        engine.dispose()


def test_corrupt_active_file_fails_the_import_naming_the_file(volume, open_store):
    (volume / "adapters" / "flask" / "active").write_text("not-an-int")
    with pytest.raises(StorageError, match=r"corrupt active pointer.*flask[\\/]+active"):
        open_store(volume)


def test_active_file_naming_a_missing_version_fails_the_import(volume, open_store):
    (volume / "adapters" / "flask" / "active").write_text("9")
    with pytest.raises(StorageError, match=r"flask[\\/]+active"):
        open_store(volume)


def test_corrupt_promotions_line_fails_the_import_naming_the_line(volume, open_store):
    with (volume / "adapters" / "flask" / "promotions.jsonl").open("a") as f:
        f.write("{oops\n")
    with pytest.raises(StorageError, match=r"promotions\.jsonl:3"):
        open_store(volume)


def test_metadata_under_the_wrong_directory_fails_the_import(volume, open_store):
    meta = volume / "adapters" / "flask" / "v1" / "metadata.json"
    shutil.copy(volume / "adapters" / "flask" / "v2" / "metadata.json", meta)
    with pytest.raises(StorageError, match="describes flask v2"):
        open_store(volume)


def test_v1_0_metadata_imports(volume, open_store):
    """A metadata.json written before contracts v1.1 has no composed_from key."""
    meta_path = volume / "adapters" / "flask" / "v1" / "metadata.json"
    d = json.loads(meta_path.read_text())
    d.pop("composed_from", None)
    d["contracts_version"] = "1.0.0"
    meta_path.write_text(json.dumps(d))

    back = open_store(volume).get_metadata("flask", 1)
    assert back.composed_from is None
    assert back.contracts_version == "1.0.0"


def test_stray_entries_are_ignored(volume, open_store):
    """Staging/trash leftovers, version-less adapter dirs and plain files are not versions."""
    (volume / "adapters" / "flask" / ".tmp-v3-abc").mkdir()
    (volume / "adapters" / "flask" / "gc.jsonl").write_text("{}\n")
    (volume / "adapters" / "empty").mkdir()
    (volume / "adapters" / "README").write_text("x")
    store = open_store(volume)
    assert store.list_adapters() == EXPECTED["adapters"]
    assert store.list_versions("flask") == [1, 2]


def test_fresh_volume_path_is_created(tmp_path, monkeypatch, safetensors_blob):
    monkeypatch.delenv(db.DB_ENV, raising=False)
    store = RegistryStore(tmp_path / "does" / "not" / "exist")
    try:
        store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
        assert (tmp_path / "does" / "not" / "exist" / "registry.db").exists()
        assert store.list_versions("flask") == [1]
    finally:
        store.close()


def test_db_url_comes_from_the_environment_when_set(tmp_path, monkeypatch, safetensors_blob):
    url = f"sqlite:///{(tmp_path / 'elsewhere.db').as_posix()}"
    monkeypatch.setenv(db.DB_ENV, url)
    store = RegistryStore(tmp_path / "root")
    try:
        store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    finally:
        store.close()
    assert (tmp_path / "elsewhere.db").exists()
    assert not (tmp_path / "root" / "registry.db").exists()
