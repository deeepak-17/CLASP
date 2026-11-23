"""Task 6.1: SQLite schema and engine for the registry metadata."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, insert, text
from sqlalchemy.exc import IntegrityError

from registry import db

TABLES = {
    "adapters",
    "versions",
    "active_pointer",
    "composite_parts",
    "promotion_decisions",
    "cluster_sync",
}


@pytest.fixture
def engine(tmp_path):
    eng = db.make_engine(f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")
    yield eng
    eng.dispose()


def test_create_all_creates_tables_and_is_idempotent(engine):
    db.create_all(engine)
    assert TABLES <= set(inspect(engine).get_table_names())
    db.create_all(engine)
    assert TABLES <= set(inspect(engine).get_table_names())


def test_pragmas_applied_on_every_connection(engine):
    db.create_all(engine)
    for _ in range(2):
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            assert conn.execute(text("PRAGMA synchronous")).scalar() == 2
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def _version(adapter_id, version=1):
    return {
        "adapter_id": adapter_id,
        "version": version,
        "sha256": "ab" * 32,
        "metadata_json": "{}",
    }


def test_version_requires_known_adapter(engine):
    db.create_all(engine)
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert(db.versions).values(**_version(999)))


def test_duplicate_adapter_version_rejected(engine):
    db.create_all(engine)
    with engine.begin() as conn:
        aid = conn.execute(insert(db.adapters).values(name="a", kind="client")).inserted_primary_key[0]
        conn.execute(insert(db.versions).values(**_version(aid)))
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert(db.versions).values(**_version(aid)))


def test_default_url_uses_registry_data(monkeypatch, tmp_path):
    monkeypatch.delenv(db.DB_ENV, raising=False)
    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path))
    assert db.default_url() == f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    monkeypatch.setenv(db.DB_ENV, "sqlite:///x.db")
    assert db.default_url() == "sqlite:///x.db"




# --- fix round 1: constraints ------------------------------------------------ #


def _seed(conn):
    aid = conn.execute(insert(db.adapters).values(name="a", kind="client")).inserted_primary_key[0]
    vid = conn.execute(insert(db.versions).values(**_version(aid))).inserted_primary_key[0]
    did = conn.execute(
        insert(db.promotion_decisions).values(
            adapter_id=aid, version=1, action="promote", active_version_after=1, decision_json="{}"
        )
    ).inserted_primary_key[0]
    return aid, vid, did


def _sync(did, **over):
    row = {"decision_id": did, "cluster_id": "c", "sha256": "x", "action": "promote"}
    row.update(over)
    return row


def _rejects(engine, table, row):
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert(table).values(**row))


@pytest.fixture
def seeded(engine):
    db.create_all(engine)
    with engine.begin() as conn:
        return engine, _seed(conn)


@pytest.mark.parametrize("col", ["adapter_id", "action", "active_version_after", "decision_json", "version"])
def test_decision_nulls_rejected(seeded, col):
    eng, (aid, _vid, _did) = seeded
    row = {"adapter_id": aid, "version": 2, "action": "promote", "active_version_after": 1, "decision_json": "{}"}
    row[col] = None
    _rejects(eng, db.promotion_decisions, row)


@pytest.mark.parametrize("col", ["decision_id", "action", "status", "attempts", "cluster_id", "sha256"])
def test_cluster_sync_nulls_rejected(seeded, col):
    eng, (_aid, _vid, did) = seeded
    _rejects(eng, db.cluster_sync, _sync(did, **{col: None}))


def test_cluster_sync_checks_and_unique(seeded):
    eng, (_aid, _vid, did) = seeded
    _rejects(eng, db.cluster_sync, _sync(did, action="bogus"))
    _rejects(eng, db.cluster_sync, _sync(did, status="bogus"))
    with eng.begin() as conn:
        conn.execute(insert(db.cluster_sync).values(**_sync(did)))
        row = conn.execute(db.cluster_sync.select()).one()
        assert (row.status, row.attempts) == ("pending", 0)
    _rejects(eng, db.cluster_sync, _sync(did))


def test_cluster_sync_unknown_decision_rejected(seeded):
    eng, _ = seeded
    _rejects(eng, db.cluster_sync, _sync(999))


def test_composite_parts_constraints(seeded):
    eng, (_aid, vid, _did) = seeded
    ok = {"composite_version_id": vid, "part_version_id": vid, "role": "cluster", "weight": 0.5}
    _rejects(eng, db.composite_parts, {**ok, "part_version_id": None})
    _rejects(eng, db.composite_parts, {**ok, "role": "other"})
    _rejects(eng, db.composite_parts, {**ok, "part_version_id": 999})
    with eng.begin() as conn:
        conn.execute(insert(db.composite_parts).values(**ok))
    _rejects(eng, db.composite_parts, ok)  # PK (composite_version_id, role)


def test_active_pointer_foreign_keys(seeded):
    eng, (aid, vid, _did) = seeded
    _rejects(eng, db.active_pointer, {"adapter_id": 999, "version_id": vid})
    _rejects(eng, db.active_pointer, {"adapter_id": aid, "version_id": 999})
    _rejects(eng, db.active_pointer, {"adapter_id": aid, "version_id": None})


def test_indexes_exist(engine):
    db.create_all(engine)
    insp = inspect(engine)
    parts = {tuple(i["column_names"]) for i in insp.get_indexes("composite_parts")}
    assert ("part_version_id",) in parts
    dec = {tuple(i["column_names"]) for i in insp.get_indexes("promotion_decisions")}
    assert ("adapter_id", "id") in dec


@pytest.mark.parametrize("state", ["set", "empty", "unset"])
def test_default_url_inside_data_root(monkeypatch, tmp_path, state):
    from registry import config

    monkeypatch.delenv(db.DB_ENV, raising=False)
    if state == "set":
        monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path))
    elif state == "empty":
        monkeypatch.setenv("CLASP_REGISTRY_DATA", "")
    else:
        monkeypatch.delenv("CLASP_REGISTRY_DATA", raising=False)
    assert db.default_url() == f"sqlite:///{(config.data_root() / 'registry.db').as_posix()}"
    if state != "set":
        assert config.data_root().as_posix() == "/data/registry"


def test_data_root_explicit_wins(monkeypatch, tmp_path):
    from registry import config

    monkeypatch.setenv("CLASP_REGISTRY_DATA", "/elsewhere")
    assert config.data_root(tmp_path) == tmp_path
