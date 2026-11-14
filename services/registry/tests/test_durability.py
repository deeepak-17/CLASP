"""Durable writes: every registry write reaches the disk before it is visible.

Atomic renames already stop a crash from leaving a half-written version; these
tests pin the other half: file contents and directory entries are fsynced, so
a power cut right after a save cannot lose it either.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from contracts import AdapterKind, LoRAHyperParams, PromotionAction, PromotionDecision
from registry import storage


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


def test_set_active_syncs_the_pointer_before_and_its_directory_after(
        store, safetensors_blob, synced):
    _save(store, safetensors_blob)
    _save(store, safetensors_blob)
    synced.clear()

    store.set_active("flask", 1)

    adapter_dir = store.adapters_dir / "flask"
    assert [p.name for p in synced] == ["active.tmp", "flask"]
    assert synced[-1] == adapter_dir
    assert store.get_active("flask") == 1


def test_audit_log_lines_are_synced(store, safetensors_blob, synced):
    meta = _save(store, safetensors_blob)
    synced.clear()

    store.record_promotion("flask", PromotionDecision(
        adapter=meta.ref, action=PromotionAction.PROMOTE,
        active_version_after=1, reason="test"))

    assert synced == [store.adapters_dir / "flask" / "promotions.jsonl"]


def test_fsync_really_syncs_a_file_and_a_directory(tmp_path):
    # The real helper, no spy: must work on files and (off Windows) directories.
    target = tmp_path / "f.bin"
    target.write_bytes(b"x")

    storage._fsync(target)
    storage._fsync(tmp_path)


def test_fsync_skips_directories_on_windows(tmp_path, monkeypatch):
    # Windows cannot open a directory handle; NTFS journals its own metadata.
    monkeypatch.setattr(storage, "_IS_WINDOWS", True)
    opened = []
    monkeypatch.setattr(storage.os, "open", lambda *a, **k: opened.append(a))

    storage._fsync(tmp_path)

    assert opened == []
