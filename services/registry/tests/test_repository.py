"""Task 6.2: SQLAlchemy Core metadata repository."""

from __future__ import annotations

import threading

import pytest
from contracts import (
    AdapterKind,
    AdapterMetadata,
    AdapterRef,
    CompositeProvenance,
    LoRAHyperParams,
    PrivacySpec,
    PromotionAction,
    PromotionDecision,
)
from sqlalchemy import event

from registry import db
from registry.promotion import RESTORE_REASON_PREFIX
from registry.repository import SqlMetadataRepository
from registry.storage import AdapterNotFound, KindMismatch, StorageError, VersionExists


@pytest.fixture
def engine(tmp_path):
    eng = db.make_engine(f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")
    db.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def repo(engine):
    return SqlMetadataRepository(engine)


@pytest.fixture
def sql_log(engine):
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _log(_conn, _cur, statement, *_a):
        statements.append(statement)

    return statements


def _meta(name, version, kind=AdapterKind.CLIENT, composed_from=None):
    return AdapterMetadata(
        ref=AdapterRef(name=name, version=version, kind=kind),
        hparams=LoRAHyperParams(),
        privacy=PrivacySpec(),
        aggregation=None,
        round=None,
        seed=0,
        sha256=f"sha-{name}-{version}",
        num_bytes=10,
        composed_from=composed_from,
    )


def _decision(name, version, action=PromotionAction.PROMOTE, reason="ok"):
    return PromotionDecision(
        adapter=AdapterRef(name=name, version=version, kind=AdapterKind.CLIENT),
        action=action,
        active_version_after=version,
        reason=reason,
    )


def _composite(name, version, cluster=("c", 1), client=("a", 1)):
    return _meta(
        name,
        version,
        AdapterKind.COMPOSITE,
        CompositeProvenance(cluster[0], cluster[1], client[0], client[1], alpha=0.7, beta=0.3),
    )


def test_list_adapters(repo):
    assert repo.list_adapters() == []
    repo.add_version(_meta("b", 1))
    repo.add_version(_meta("a", 1))
    assert repo.list_adapters() == ["a", "b"]


def test_add_and_list_versions_and_get_metadata(repo):
    stored = _meta("a", 2)
    repo.add_version(stored)
    repo.add_version(_meta("a", 1))
    assert repo.list_versions("a") == [1, 2]
    assert repo.list_versions("missing") == []
    assert repo.get_metadata("a", 2) == stored


def test_get_metadata_unknown_raises(repo):
    with pytest.raises(AdapterNotFound):
        repo.get_metadata("a", 1)
    repo.add_version(_meta("a", 1))
    with pytest.raises(AdapterNotFound):
        repo.get_metadata("a", 9)


def test_add_version_duplicate_and_kind_mismatch(repo):
    repo.add_version(_meta("a", 1))
    with pytest.raises(VersionExists):
        repo.add_version(_meta("a", 1))
    with pytest.raises(KindMismatch):
        repo.add_version(_meta("a", 2, AdapterKind.CLUSTER))
    assert repo.list_versions("a") == [1]


def test_delete_version(repo):
    repo.add_version(_meta("a", 1))
    repo.add_version(_meta("a", 2))
    repo.delete_version("a", 1)
    assert repo.list_versions("a") == [2]
    with pytest.raises(AdapterNotFound):
        repo.delete_version("a", 1)


def test_delete_active_version_refused(repo):
    repo.add_version(_meta("a", 1))
    repo.set_active("a", 1)
    with pytest.raises(StorageError):
        repo.delete_version("a", 1)
    assert repo.list_versions("a") == [1]


def test_delete_composite_removes_its_parts_rows(repo):
    repo.add_version(_meta("c", 1, AdapterKind.CLUSTER))
    repo.add_version(_meta("a", 1))
    repo.add_version(_composite("m", 1))
    assert repo.referenced_versions("a") == {1}
    repo.delete_version("m", 1)
    assert repo.referenced_versions("a") == set()
    repo.delete_version("a", 1)


def test_get_and_set_active(repo):
    assert repo.get_active("a") is None
    repo.add_version(_meta("a", 1))
    repo.add_version(_meta("a", 2))
    assert repo.get_active("a") is None
    repo.set_active("a", 1)
    assert repo.get_active("a") == 1
    repo.set_active("a", 2)
    assert repo.get_active("a") == 2


def test_set_active_unknown_raises(repo):
    with pytest.raises(AdapterNotFound):
        repo.set_active("a", 1)
    repo.add_version(_meta("a", 1))
    with pytest.raises(AdapterNotFound):
        repo.set_active("a", 5)


def test_record_and_list_decisions(repo):
    repo.add_version(_meta("a", 1))
    d1 = _decision("a", 1)
    d2 = _decision("a", 1, PromotionAction.ROLLBACK, f"{RESTORE_REASON_PREFIX} v1")
    i1 = repo.record_decision("a", d1)
    i2 = repo.record_decision("a", d2)
    assert i2 > i1
    assert repo.list_decisions("a") == [d1, d2]
    assert repo.list_decisions("other") == []


def test_record_decision_unknown_adapter_raises(repo):
    with pytest.raises(AdapterNotFound):
        repo.record_decision("a", _decision("a", 1))


def test_decided_versions_excludes_restores(repo):
    repo.add_version(_meta("a", 1))
    repo.add_version(_meta("a", 2))
    repo.record_decision("a", _decision("a", 1))
    repo.record_decision("a", _decision("a", 2, reason=f"{RESTORE_REASON_PREFIX} v2"))
    assert repo.decided_versions("a") == {1}
    assert repo.decided_versions("none") == set()


def test_referenced_versions_and_composites_built_from(repo):
    repo.add_version(_meta("c", 1, AdapterKind.CLUSTER))
    repo.add_version(_meta("a", 1))
    repo.add_version(_meta("a", 2))
    repo.add_version(_composite("m", 1))
    assert repo.referenced_versions("a") == {1}
    assert repo.referenced_versions("c") == {1}
    assert repo.referenced_versions("m") == set()
    assert repo.composites_built_from("a", 1) == [
        AdapterRef(name="m", version=1, kind=AdapterKind.COMPOSITE)
    ]
    assert [r.name for r in repo.composites_built_from("c", 1)] == ["m"]
    assert repo.composites_built_from("a", 2) == []


def test_composite_parts_rows(repo, engine):
    repo.add_version(_meta("c", 1, AdapterKind.CLUSTER))
    repo.add_version(_meta("a", 1))
    repo.add_version(_composite("m", 1))
    with engine.connect() as conn:
        rows = conn.execute(
            db.composite_parts.select().order_by(db.composite_parts.c.role)
        ).all()
    assert [(r.role, r.weight) for r in rows] == [("client", 0.3), ("cluster", 0.7)]


def test_composite_with_missing_part_raises_and_leaves_nothing(repo):
    repo.add_version(_meta("a", 1))
    with pytest.raises(AdapterNotFound):
        repo.add_version(_composite("m", 1))
    assert repo.list_versions("m") == []
    assert repo.list_adapters() == ["a"]


def test_cluster_sync_lifecycle(repo):
    repo.add_version(_meta("a", 1))
    did = repo.record_decision("a", _decision("a", 1))
    sid = repo.add_cluster_sync(did, "cl1", "sha", "promote")
    pending = repo.pending_cluster_sync()
    assert len(pending) == 1
    row = pending[0]
    assert row["id"] == sid and row["decision_id"] == did
    assert row["cluster_id"] == "cl1" and row["sha256"] == "sha"
    assert row["action"] == "promote" and row["attempts"] == 0
    repo.mark_cluster_sync(sid, done=False, error="boom")
    row = repo.pending_cluster_sync()[0]
    assert row["attempts"] == 1 and row["last_error"] == "boom"
    repo.mark_cluster_sync(sid, done=True)
    assert repo.pending_cluster_sync() == []


def test_exception_inside_transaction_rolls_everything_back(repo):
    with pytest.raises(RuntimeError):
        with repo.transaction():
            repo.add_version(_meta("a", 1))
            repo.set_active("a", 1)
            repo.record_decision("a", _decision("a", 1))
            raise RuntimeError
    assert repo.list_versions("a") == []
    assert repo.get_active("a") is None
    assert repo.list_decisions("a") == []


def test_nested_transaction_joins_outer(repo):
    with pytest.raises(RuntimeError):
        with repo.transaction():
            with repo.transaction():
                repo.add_version(_meta("a", 1))
            assert repo.list_versions("a") == [1]
            raise RuntimeError
    assert repo.list_versions("a") == []
    with repo.transaction():
        with repo.transaction():
            repo.add_version(_meta("a", 1))
    assert repo.list_versions("a") == [1]


def test_referenced_versions_is_one_indexed_query(repo, sql_log):
    repo.add_version(_meta("c", 1, AdapterKind.CLUSTER))
    repo.add_version(_meta("a", 1))
    repo.add_version(_composite("m", 1))
    sql_log.clear()
    assert repo.referenced_versions("a") == {1}
    selects = [s for s in sql_log if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1 and "composite_parts" in selects[0]
    assert len(sql_log) == 1


def test_transaction_is_per_thread(repo):
    inside = threading.Event()
    release = threading.Event()
    seen: dict[str, object] = {}

    def writer():
        with repo.transaction():
            repo.add_version(_meta("a", 1))
            inside.set()
            assert release.wait(10)

    t = threading.Thread(target=writer)
    t.start()
    assert inside.wait(10)
    # Another thread: its own connection, and it must not see uncommitted data.
    seen["versions"] = repo.list_versions("a")
    with repo.transaction():
        seen["inner"] = repo.list_versions("a")
    release.set()
    t.join(10)
    assert seen["versions"] == [] and seen["inner"] == []
    assert repo.list_versions("a") == [1]


def test_delete_part_of_a_composite_refused(repo):
    repo.add_version(_meta("c", 1, AdapterKind.CLUSTER))
    repo.add_version(_meta("a", 1))
    repo.add_version(_composite("m", 1))
    with pytest.raises(StorageError):
        repo.delete_version("a", 1)
    assert repo.list_versions("a") == [1]
