"""Durable writes: every registry write reaches the disk before it is visible.

Atomic renames already stop a crash from leaving a half-written version; these
tests pin the other half: file contents and directory entries are fsynced, so
a power cut right after a save cannot lose it either.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from contracts import AdapterKind, LoRAHyperParams, PromotionAction, PromotionDecision
from contracts import RunManifest
from registry import durable, manifest, storage
from registry.runs import ResultStore, _run_point


@pytest.fixture
def synced(monkeypatch):
    """Record every path passed to storage._fsync, in order, without syncing."""
    calls: list[Path] = []
    monkeypatch.setattr(storage, "_fsync", lambda path: calls.append(Path(path)))
    return calls


def _save(store, blob, name="flask"):
    return store.save(name, blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())


def test_save_syncs_both_files_while_still_staged(store, safetensors_blob, synced):
    # Arrange / Act
    _save(store, safetensors_blob)

    # Assert: both files were synced inside the hidden staging dir, i.e. before
    # the rename published them as v1.
    staged = [p for p in synced if p.parent.name.startswith(".tmp-v1-")]
    assert {p.name for p in staged} == {"adapter.safetensors", "metadata.json"}


def test_save_syncs_the_directories_that_name_the_new_version(store, safetensors_blob, synced):
    _save(store, safetensors_blob)

    adapter_dir = store.adapters_dir / "flask"
    assert adapter_dir in synced           # the rename that created v1
    assert store.adapters_dir in synced    # the first save created the adapter dir


def test_set_active_writes_no_file_the_database_holds_it(store, safetensors_blob, synced):
    # The pointer is a row committed with synchronous=FULL, not a file.
    _save(store, safetensors_blob)
    _save(store, safetensors_blob)
    synced.clear()

    store.set_active("flask", 1)

    assert synced == []
    assert not (store.adapters_dir / "flask" / "active").exists()
    assert store.repo.get_active("flask") == 1


def test_promotion_decisions_write_no_file_the_database_holds_them(
        store, safetensors_blob, synced):
    meta = _save(store, safetensors_blob)
    synced.clear()

    decision = PromotionDecision(adapter=meta.ref, action=PromotionAction.PROMOTE,
                                 active_version_after=1, reason="test")
    store.record_promotion("flask", decision)

    assert synced == []
    assert not (store.adapters_dir / "flask" / "promotions.jsonl").exists()
    assert store.repo.list_decisions("flask") == [decision]


def test_file_audit_log_lines_are_synced(store, safetensors_blob, synced):
    # gc.jsonl stays a file log, synced on every append.
    _save(store, safetensors_blob)
    synced.clear()

    store.append_log("flask", "gc.jsonl", "{}")

    assert synced == [store.adapters_dir / "flask" / "gc.jsonl"]


def test_fsync_really_syncs_a_file_and_a_directory(tmp_path):
    # The real helper, no spy: must work on files and (off Windows) directories.
    target = tmp_path / "f.bin"
    target.write_bytes(b"x")

    storage._fsync(target)
    storage._fsync(tmp_path)


def test_fsync_skips_directories_on_windows(tmp_path, monkeypatch):
    # Windows cannot open a directory handle; NTFS journals its own metadata.
    monkeypatch.setattr(durable, "_IS_WINDOWS", True)
    opened = []
    monkeypatch.setattr(durable.os, "open", lambda *a, **k: opened.append(a))

    storage._fsync(tmp_path)

    assert opened == []


def test_atomic_write_text_syncs_file_then_replaces_then_syncs_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(durable, "fsync_path", lambda p: calls.append(("fsync", p.name)))
    real_replace = os.replace
    monkeypatch.setattr(
        os, "replace", lambda a, b: (calls.append(("replace", Path(b).name)), real_replace(a, b))
    )

    durable.atomic_write_text(tmp_path / "run.json", "{}")

    assert calls == [("fsync", "run.json.tmp"), ("replace", "run.json"), ("fsync", tmp_path.name)]


def test_run_record_config_and_manifest_all_go_through_atomic_write(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(durable, "atomic_write_text", lambda p, t: written.append(Path(p).name))
    config = {"name": "exp", "seed": 0, "gpu_hours_estimate": 0.1}

    # run.json and config.resolved.json come from one run of the sweep point path
    _run_point(config, {}, [sys.executable, "-c", "pass"], ResultStore(tmp_path / "store"), None)
    # the manifest is written on its own
    run_manifest = RunManifest(run_id="r1", seed=0, config_hash="h", adapter_versions={},
                               gpu_hours_estimate=0.1, notes="")
    manifest.write_manifest(run_manifest, tmp_path / "manifest.json")

    assert {"run.json", "config.resolved.json"} <= set(written)
    assert any(name.endswith(".json") and "manifest" in name for name in written)
