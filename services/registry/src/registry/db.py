"""SQLite metadata schema and engine for the State Registry (SQLAlchemy Core).

Adapter weight files stay on disk; this module holds only metadata: versions,
active pointers, promotion decisions and cluster-sync bookkeeping. PostgreSQL
later is a matter of setting ``CLASP_REGISTRY_DB`` to another URL.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (
    CheckConstraint,
    Column,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)

DB_ENV = "CLASP_REGISTRY_DB"

metadata = MetaData()

adapters = Table(
    "adapters",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("kind", String, nullable=False),
)

versions = Table(
    "versions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("adapter_id", Integer, ForeignKey("adapters.id"), nullable=False),
    Column("version", Integer, nullable=False),
    Column("sha256", String, nullable=False),
    Column("num_bytes", Integer),
    Column("created_at", String),
    Column("metadata_json", Text, nullable=False),
    UniqueConstraint("adapter_id", "version"),
)

active_pointer = Table(
    "active_pointer",
    metadata,
    Column("adapter_id", Integer, ForeignKey("adapters.id"), primary_key=True),
    Column("version_id", Integer, ForeignKey("versions.id"), nullable=False),
    Column("moved_at", String),
)

composite_parts = Table(
    "composite_parts",
    metadata,
    Column("composite_version_id", Integer, ForeignKey("versions.id"), primary_key=True),
    Column("part_version_id", Integer, ForeignKey("versions.id")),
    Column("role", String, primary_key=True),
    Column("weight", Float),
    CheckConstraint("role IN ('cluster','client')", name="ck_composite_parts_role"),
    Index("ix_composite_parts_part_version_id", "part_version_id"),
)

promotion_decisions = Table(
    "promotion_decisions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("adapter_id", Integer, ForeignKey("adapters.id")),
    Column("version", Integer, nullable=False),
    Column("action", String, nullable=False),
    Column("active_version_after", Integer, nullable=False),
    Column("reason", Text),
    Column("decided_at", String),
    Column("decision_json", Text, nullable=False),
    Index("ix_promotion_decisions_adapter_id_id", "adapter_id", "id"),
)

cluster_sync = Table(
    "cluster_sync",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("decision_id", Integer, ForeignKey("promotion_decisions.id"), unique=True),
    Column("cluster_id", String, nullable=False),
    Column("sha256", String, nullable=False),
    Column("action", String),
    Column("status", String, server_default="pending"),
    Column("attempts", Integer, server_default="0"),
    Column("last_error", Text),
    Column("updated_at", String),
    CheckConstraint("action IN ('promote','reject')", name="ck_cluster_sync_action"),
    CheckConstraint("status IN ('pending','done')", name="ck_cluster_sync_status"),
)


def default_url() -> str:
    """``CLASP_REGISTRY_DB`` if set, else SQLite under the registry data root."""
    explicit = os.environ.get(DB_ENV)
    if explicit:
        return explicit
    # Same resolution as RegistryStore.
    root = Path(os.environ.get("CLASP_REGISTRY_DATA", "/data/registry"))
    return f"sqlite:///{(root / 'registry.db').as_posix()}"


def make_engine(url: str) -> Engine:
    """Create an engine; SQLite connections get WAL, FULL sync and foreign keys."""
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=FULL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def create_all(engine: Engine) -> None:
    """Create any missing tables (idempotent)."""
    metadata.create_all(engine)
