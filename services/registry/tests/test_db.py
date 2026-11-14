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


