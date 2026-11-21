"""Metadata repository over SQLAlchemy Core (versions, pointers, decisions, sync).

Adapter weight files stay on disk; this holds only metadata. ``transaction()``
is re-entrant *per thread*: FastAPI serves sync routes from a threadpool, so a
nested call on the same thread joins the outer transaction while another
thread gets its own connection (and never sees uncommitted writes). Outside an
explicit ``transaction()`` every method runs in its own ``engine.begin()``.

Only the exception classes are taken from ``registry.errors`` so that
``registry.storage`` can import this module without a cycle.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ContextManager, Protocol

from contracts import AdapterMetadata, AdapterRef, PromotionDecision, utcnow_iso
from sqlalchemy import Connection, Engine, delete, func, insert, select, update

from registry.config import RUN_DELTA
from registry.db import (
    access_log,
    active_pointer,
    adapters,
    cluster_sync,
    composite_parts,
    privacy_ledger,
    promotion_decisions,
    versions,
)
from registry.errors import AdapterNotFound, KindMismatch, StorageError, VersionExists
from registry.promotion import RESTORE_REASON_PREFIX


class MetadataRepository(Protocol):
    def transaction(self) -> ContextManager[None]: ...
    def list_adapters(self) -> list[str]: ...
    def list_versions(self, name: str) -> list[int]: ...
    def get_metadata(self, name: str, version: int) -> AdapterMetadata: ...
    def add_version(self, meta: AdapterMetadata) -> None: ...
    def delete_version(self, name: str, version: int) -> None: ...
    def get_active(self, name: str) -> int | None: ...
    def set_active(self, name: str, version: int) -> None: ...
    def record_decision(self, name: str, decision: PromotionDecision) -> int: ...
    def list_decisions(self, name: str) -> list[PromotionDecision]: ...
    def decided_versions(self, name: str) -> set[int]: ...
    def referenced_versions(self, name: str) -> set[int]: ...
    def composites_built_from(self, name: str, version: int) -> list[AdapterRef]: ...
    def add_cluster_sync(self, decision_id: int, cluster_id: str, sha256: str, action: str) -> int: ...
    def pending_cluster_sync(self) -> list[dict]: ...
    def mark_cluster_sync(self, sync_id: int, *, done: bool, error: str | None = None) -> bool: ...
    def add_privacy_rows(self, name: str, version: int, rows: list[dict]) -> None: ...
    def privacy_ledger(self, cluster_id: str | None = None) -> list[dict]: ...
    def add_access(self, row: dict) -> None: ...
    def access_log(self, limit: int = 200, identity: str | None = None) -> list[dict]: ...


#: The access log keeps this many of its newest rows (one row per request).
ACCESS_LOG_MAX_ROWS = 200_000
#: ... and drops the older ones once every this many inserts.
ACCESS_LOG_PRUNE_EVERY = 1_000


class SqlMetadataRepository:
    def __init__(self, engine: Engine, *, access_log_max_rows: int = ACCESS_LOG_MAX_ROWS,
                 access_log_prune_every: int = ACCESS_LOG_PRUNE_EVERY) -> None:
        if access_log_max_rows < 1 or access_log_prune_every < 1:
            raise ValueError("the access log cap and its prune interval must be >= 1")
        self._engine = engine
        self._local = threading.local()
        self._access_cap = access_log_max_rows
        self._access_prune_every = access_log_prune_every

    # -- transactions ----------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit on exit, roll back on exception; nested calls join the outer one."""
        local = self._local
        if getattr(local, "conn", None) is not None:
            local.depth += 1
            try:
                yield
            finally:
                local.depth -= 1
            return
        with self._engine.begin() as conn:
            local.conn = conn
            local.depth = 1
            try:
                yield
            finally:
                local.conn = None
                local.depth = 0

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            yield conn
            return
        with self._engine.begin() as new:
            yield new

    # -- lookups ---------------------------------------------------------

    @staticmethod
    def _adapter_id(conn: Connection, name: str) -> int | None:
        return conn.execute(select(adapters.c.id).where(adapters.c.name == name)).scalar()

    @staticmethod
    def _version_id(conn: Connection, name: str, version: int) -> int | None:
        return conn.execute(
            select(versions.c.id)
            .join(adapters, adapters.c.id == versions.c.adapter_id)
            .where(adapters.c.name == name, versions.c.version == version)
        ).scalar()

    # -- adapters / versions --------------------------------------------

    def list_adapters(self) -> list[str]:
        with self._connection() as conn:
            return list(conn.execute(select(adapters.c.name).order_by(adapters.c.name)).scalars())

    def list_versions(self, name: str) -> list[int]:
        with self._connection() as conn:
            return list(
                conn.execute(
                    select(versions.c.version)
                    .join(adapters, adapters.c.id == versions.c.adapter_id)
                    .where(adapters.c.name == name)
                    .order_by(versions.c.version)
                ).scalars()
            )

    def get_metadata(self, name: str, version: int) -> AdapterMetadata:
        with self._connection() as conn:
            raw = conn.execute(
                select(versions.c.metadata_json)
                .join(adapters, adapters.c.id == versions.c.adapter_id)
                .where(adapters.c.name == name, versions.c.version == version)
            ).scalar()
        if raw is None:
            raise AdapterNotFound(f"{name} v{version}")
        return AdapterMetadata.from_json(json.loads(raw))

    def add_version(self, meta: AdapterMetadata) -> None:
        name, version, kind = meta.ref.name, meta.ref.version, meta.ref.kind.value
        with self._connection() as conn:
            row = conn.execute(select(adapters.c.id, adapters.c.kind).where(adapters.c.name == name)).first()
            if row is not None and row.kind != kind:
                raise KindMismatch(f"{name} is {row.kind}, not {kind}")
            if row is not None and self._version_id(conn, name, version) is not None:
                raise VersionExists(f"{name} v{version}")
            # A part row only for a part the registry holds: the file-based
            # registry accepted provenance naming absent parts, and those
            # composites (and old volumes) must still save. The full
            # provenance stays in metadata_json either way.
            parts: list[tuple[str, int, float]] = []
            if meta.composed_from is not None:
                cf = meta.composed_from
                for role, pname, pver, weight in (
                    ("cluster", cf.cluster_name, cf.cluster_version, cf.alpha),
                    ("client", cf.client_name, cf.client_version, cf.beta),
                ):
                    part_id = self._version_id(conn, pname, pver)
                    if part_id is not None:
                        parts.append((role, part_id, weight))
            adapter_id = (
                row.id
                if row is not None
                else conn.execute(insert(adapters).values(name=name, kind=kind)).inserted_primary_key[0]
            )
            version_id = conn.execute(
                insert(versions).values(
                    adapter_id=adapter_id,
                    version=version,
                    sha256=meta.sha256,
                    num_bytes=meta.num_bytes,
                    created_at=meta.created_at,
                    metadata_json=json.dumps(meta.to_json()),
                )
            ).inserted_primary_key[0]
            for role, part_id, weight in parts:
                conn.execute(
                    insert(composite_parts).values(
                        composite_version_id=version_id, part_version_id=part_id, role=role, weight=weight
                    )
                )

    def delete_version(self, name: str, version: int) -> None:
        with self._connection() as conn:
            version_id = self._version_id(conn, name, version)
            if version_id is None:
                raise AdapterNotFound(f"{name} v{version}")
            if conn.execute(
                select(active_pointer.c.adapter_id).where(active_pointer.c.version_id == version_id)
            ).first():
                raise StorageError(f"refusing to delete the active version {name} v{version}")
            if conn.execute(
                select(composite_parts.c.role).where(composite_parts.c.part_version_id == version_id)
            ).first():
                raise StorageError(f"refusing to delete {name} v{version}: a composite is built from it")
            conn.execute(delete(composite_parts).where(composite_parts.c.composite_version_id == version_id))
            # the spend stays on the ledger; only its link to the deleted weights goes
            conn.execute(
                update(privacy_ledger)
                .where(privacy_ledger.c.adapter_version_id == version_id)
                .values(adapter_version_id=None)
            )
            conn.execute(delete(versions).where(versions.c.id == version_id))

    # -- active pointer --------------------------------------------------

    def get_active(self, name: str) -> int | None:
        with self._connection() as conn:
            return conn.execute(
                select(versions.c.version)
                .join(active_pointer, active_pointer.c.version_id == versions.c.id)
                .join(adapters, adapters.c.id == active_pointer.c.adapter_id)
                .where(adapters.c.name == name)
            ).scalar()

    def set_active(self, name: str, version: int) -> None:
        with self._connection() as conn:
            version_id = self._version_id(conn, name, version)
            if version_id is None:
                raise AdapterNotFound(f"{name} v{version}")
            adapter_id = self._adapter_id(conn, name)
            moved_at = utcnow_iso()
            updated = conn.execute(
                update(active_pointer)
                .where(active_pointer.c.adapter_id == adapter_id)
                .values(version_id=version_id, moved_at=moved_at)
            )
            if updated.rowcount == 0:
                conn.execute(
                    insert(active_pointer).values(adapter_id=adapter_id, version_id=version_id, moved_at=moved_at)
                )

    # -- decisions -------------------------------------------------------

    def record_decision(self, name: str, decision: PromotionDecision) -> int:
        with self._connection() as conn:
            adapter_id = self._adapter_id(conn, name)
            if adapter_id is None:
                raise AdapterNotFound(name)
            return conn.execute(
                insert(promotion_decisions).values(
                    adapter_id=adapter_id,
                    version=decision.adapter.version,
                    action=decision.action.value,
                    active_version_after=decision.active_version_after,
                    reason=decision.reason,
                    decided_at=decision.timestamp,
                    decision_json=json.dumps(decision.to_json()),
                )
            ).inserted_primary_key[0]

    def list_decisions(self, name: str) -> list[PromotionDecision]:
        with self._connection() as conn:
            rows = conn.execute(
                select(promotion_decisions.c.decision_json)
                .join(adapters, adapters.c.id == promotion_decisions.c.adapter_id)
                .where(adapters.c.name == name)
                .order_by(promotion_decisions.c.id)
            ).scalars()
            return [PromotionDecision.from_json(json.loads(raw)) for raw in rows]

    def decided_versions(self, name: str) -> set[int]:
        with self._connection() as conn:
            return set(
                conn.execute(
                    select(promotion_decisions.c.version)
                    .join(adapters, adapters.c.id == promotion_decisions.c.adapter_id)
                    .where(
                        adapters.c.name == name,
                        ~func.coalesce(promotion_decisions.c.reason, "").startswith(
                            RESTORE_REASON_PREFIX, autoescape=True
                        ),
                    )
                ).scalars()
            )

    # -- composites ------------------------------------------------------

    def referenced_versions(self, name: str) -> set[int]:
        with self._connection() as conn:
            return set(
                conn.execute(
                    select(versions.c.version)
                    .join(composite_parts, composite_parts.c.part_version_id == versions.c.id)
                    .join(adapters, adapters.c.id == versions.c.adapter_id)
                    .where(adapters.c.name == name)
                    .distinct()
                ).scalars()
            )

    def composites_built_from(self, name: str, version: int) -> list[AdapterRef]:
        with self._connection() as conn:
            part_id = self._version_id(conn, name, version)
            if part_id is None:
                return []
            rows = conn.execute(
                select(adapters.c.name, versions.c.version, versions.c.metadata_json)
                .select_from(composite_parts)
                .join(versions, versions.c.id == composite_parts.c.composite_version_id)
                .join(adapters, adapters.c.id == versions.c.adapter_id)
                .where(composite_parts.c.part_version_id == part_id)
                .distinct()
                .order_by(adapters.c.name, versions.c.version)
            ).all()
        return [AdapterMetadata.from_json(json.loads(r.metadata_json)).ref for r in rows]

    # -- cluster sync ----------------------------------------------------

    def add_cluster_sync(self, decision_id: int, cluster_id: str, sha256: str, action: str) -> int:
        with self._connection() as conn:
            return conn.execute(
                insert(cluster_sync).values(
                    decision_id=decision_id,
                    cluster_id=cluster_id,
                    sha256=sha256,
                    action=action,
                    status="pending",
                    attempts=0,
                    updated_at=utcnow_iso(),
                )
            ).inserted_primary_key[0]

    def pending_cluster_sync(self) -> list[dict[str, Any]]:
        """Pending rows, oldest first, each with the ``adapter`` and ``version`` decided,
        the ``active_version_after`` it (a restore's target), and ``superseded_by``: the id of a later *done* row for the same cluster, if any."""
        with self._connection() as conn:
            rows = conn.execute(
                select(
                    *cluster_sync.c,
                    adapters.c.name.label("adapter"),
                    promotion_decisions.c.version.label("version"),
                    promotion_decisions.c.active_version_after,
                )
                .select_from(cluster_sync)
                .join(promotion_decisions, promotion_decisions.c.id == cluster_sync.c.decision_id)
                .join(adapters, adapters.c.id == promotion_decisions.c.adapter_id)
                .where(cluster_sync.c.status == "pending")
                .order_by(cluster_sync.c.id)
            ).mappings()
            out = [dict(r) for r in rows]
            for row in out:
                row["superseded_by"] = conn.execute(
                    select(func.max(cluster_sync.c.id)).where(
                        cluster_sync.c.cluster_id == row["cluster_id"],
                        cluster_sync.c.status == "done",
                        cluster_sync.c.id > row["id"],
                    )
                ).scalar()
            return out

    def mark_cluster_sync(self, sync_id: int, *, done: bool, error: str | None = None) -> bool:
        """Mark a row done, or count a failed attempt (only while it is pending).

        Returns False when no row was changed (unknown id; or a failure
        reported for a row that is already done).
        """
        values: dict[str, Any] = {"updated_at": utcnow_iso(), "last_error": error}
        stmt = update(cluster_sync).where(cluster_sync.c.id == sync_id)
        if done:
            values["status"] = "done"
        else:
            values["attempts"] = cluster_sync.c.attempts + 1
            stmt = stmt.where(cluster_sync.c.status == "pending")
        with self._connection() as conn:
            return conn.execute(stmt.values(**values)).rowcount > 0

    # -- privacy ledger (D7) ---------------------------------------------

    def add_privacy_rows(self, name: str, version: int, rows: list[dict]) -> None:
        """Record each client's run spend with the version that contains its round.

        ``rows`` are the cluster's ``client_privacy`` entries. A client without
        a cumulative epsilon trained without DP and has nothing to record. The
        cluster, round and delta come from the version's own metadata. Call it
        inside the transaction that adds the version.
        """
        with self._connection() as conn:
            found = conn.execute(
                select(versions.c.id, versions.c.metadata_json)
                .join(adapters, adapters.c.id == versions.c.adapter_id)
                .where(adapters.c.name == name, versions.c.version == version)
            ).first()
            if found is None:
                raise AdapterNotFound(f"{name} v{version}")
            meta = json.loads(found.metadata_json)
            ref = meta.get("ref") or {}
            privacy = meta.get("privacy") or {}
            now = utcnow_iso()
            for row in rows:
                if row.get("epsilon_cumulative") is None:
                    continue
                conn.execute(
                    insert(privacy_ledger).values(
                        client_id=row["client_id"],
                        cluster_id=ref.get("cluster_id") or name,
                        adapter_version_id=found.id,
                        round=meta.get("round"),
                        epsilon_round=row.get("epsilon_round"),
                        epsilon_cumulative=row["epsilon_cumulative"],
                        rounds_accounted=row.get("rounds_accounted"),
                        delta=privacy.get("delta") if privacy.get("delta") is not None else RUN_DELTA,
                        recorded_at=now,
                    )
                )

    def privacy_ledger(self, cluster_id: str | None = None) -> list[dict[str, Any]]:
        """Every recorded spend, oldest first; ``adapter`` and ``version`` are None
        once the version it was recorded with has been deleted."""
        stmt = (
            select(
                privacy_ledger.c.client_id,
                privacy_ledger.c.cluster_id,
                adapters.c.name.label("adapter"),
                versions.c.version.label("version"),
                privacy_ledger.c.round,
                privacy_ledger.c.epsilon_round,
                privacy_ledger.c.epsilon_cumulative,
                privacy_ledger.c.rounds_accounted,
                privacy_ledger.c.delta,
                privacy_ledger.c.recorded_at,
            )
            .select_from(privacy_ledger)
            .outerjoin(versions, versions.c.id == privacy_ledger.c.adapter_version_id)
            .outerjoin(adapters, adapters.c.id == versions.c.adapter_id)
            .order_by(privacy_ledger.c.id)
        )
        if cluster_id is not None:
            stmt = stmt.where(privacy_ledger.c.cluster_id == cluster_id)
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]

    # -- access log (U4) ---------------------------------------------------

    def add_access(self, row: dict) -> None:
        """Record one request: ``at``, ``identity``, ``role``, ``method``,
        ``path``, ``adapter``, ``version``, ``status``. Every
        ``access_log_prune_every`` rows, rows older than the newest
        ``access_log_max_rows`` are deleted."""
        keys = ("at", "identity", "role", "method", "path", "adapter", "version", "status")
        with self._connection() as conn:
            row_id = conn.execute(
                insert(access_log).values(**{k: row.get(k) for k in keys})).inserted_primary_key[0]
            if row_id % self._access_prune_every == 0:
                conn.execute(delete(access_log).where(access_log.c.id <= row_id - self._access_cap))

    def access_log(self, limit: int = 200, identity: str | None = None) -> list[dict[str, Any]]:
        """The newest ``limit`` rows, newest first; with ``identity``, only that caller's."""
        stmt = select(*[c for c in access_log.c if c.name != "id"]).order_by(access_log.c.id.desc())
        if identity is not None:
            stmt = stmt.where(access_log.c.identity == identity)
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(stmt.limit(limit)).mappings()]
