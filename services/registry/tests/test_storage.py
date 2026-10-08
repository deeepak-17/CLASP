"""Unit tests for the versioned storage layer (P4)."""
from __future__ import annotations

import pytest
from contracts import AdapterKind, AggregationMethod, LoRAHyperParams, PrivacySpec
from registry.storage import (
    AdapterNotFound,
    StorageError,
    VersionExists,
    is_safetensors,
)


def test_rejects_non_safetensors(store):
    with pytest.raises(Exception):
        store.save("django", b"not-a-tensor", kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())


def test_is_safetensors(safetensors_blob):
    assert is_safetensors(safetensors_blob)
    assert not is_safetensors(b"")
    assert not is_safetensors(b"\x00" * 8)


def test_save_assigns_incrementing_versions(store, safetensors_blob):
    m1 = store.save("django", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    m2 = store.save("django", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    assert m1.ref.version == 1
    assert m2.ref.version == 2
    assert store.list_versions("django") == [1, 2]


def test_save_records_digest_and_size(store, safetensors_blob):
    m = store.save("flask", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    assert m.num_bytes == len(safetensors_blob)
    assert len(m.sha256) == 64


def test_versions_are_immutable_on_disk(store, safetensors_blob, monkeypatch):
    store.save("numpy", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    # Force a version collision to prove save refuses to clobber (D9).
    monkeypatch.setattr(store, "latest_version", lambda name: 0)
    with pytest.raises(VersionExists):
        store.save("numpy", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())


def test_load_roundtrips_payload(store, safetensors_blob):
    store.save("pandas", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    assert store.load_payload("pandas", 1) == safetensors_blob


def test_metadata_roundtrips_cluster_fields(store, safetensors_blob):
    m = store.save(
        "web",
        safetensors_blob,
        kind=AdapterKind.CLUSTER,
        hparams=LoRAHyperParams(rank=16, alpha=0.8, beta=0.5),
        privacy=PrivacySpec(epsilon=7.2, delta=1e-5),
        aggregation=AggregationMethod.SVD_EXACT,
        round=3,
        cluster_id="web",
        source_clients=("django", "flask", "requests"),
    )
    reloaded = store.get_metadata("web", m.ref.version)
    assert reloaded.aggregation is AggregationMethod.SVD_EXACT
    assert reloaded.privacy.epsilon == 7.2
    assert reloaded.hparams.alpha == 0.8
    assert reloaded.round == 3
    assert reloaded.source_clients == ("django", "flask", "requests")


def test_active_pointer(store, safetensors_blob):
    store.save("sk", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    store.save("sk", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    assert store.get_active("sk") == 2  # last save is active by default
    store.set_active("sk", 1)
    assert store.get_active("sk") == 1


def test_missing_adapter_raises(store):
    with pytest.raises(AdapterNotFound):
        store.list_versions("does-not-exist")


def test_corrupt_active_pointer_degrades(store, safetensors_blob):
    # L1: a garbage `active` file must raise StorageError, not a bare ValueError.
    store.save("torch", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    (store._adapter_dir("torch") / "active").write_text("not-an-int")
    from registry.storage import StorageError

    with pytest.raises(StorageError):
        store.get_active("torch")


def test_set_active_is_atomic_no_tmp_left(store, safetensors_blob):
    # L2: temp file is renamed away, never left behind.
    store.save("scipy", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    store.set_active("scipy", 1)
    assert not (store._adapter_dir("scipy") / "active.tmp").exists()
    assert store.get_active("scipy") == 1


def test_invalid_name_rejected(store, safetensors_blob):
    with pytest.raises(StorageError):
        store.save("../evil", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())


def test_save_leaves_no_orphan_on_crash_between_writes(store, safetensors_blob, monkeypatch):
    # M5 (Thu target): a crash between the safetensors write and the metadata
    # write must never leave a version dir that `list_versions` can see with
    # one file missing. We simulate the crash by blowing up mid-staging,
    # after adapter.safetensors is already on disk but before metadata.json
    # is written / the publish rename happens.
    import registry.storage as storage_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-save")

    monkeypatch.setattr(storage_mod, "_metadata_to_dict", boom)

    with pytest.raises(RuntimeError):
        store.save("crashy", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())

    # No half-written v1/ ever became visible...
    assert store.list_versions("crashy") == []
    assert not (store._adapter_dir("crashy") / "v1").exists()
    # ...and no abandoned staging directory was left behind either.
    leftovers = list(store._adapter_dir("crashy").iterdir())
    assert leftovers == [], f"staging dir not cleaned up: {leftovers}"

    # The failed attempt didn't burn a version number — a real save still
    # lands on v1, not v2, proving nothing was claimed by the crash.
    monkeypatch.undo()
    m = store.save("crashy", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    assert m.ref.version == 1


def test_save_publishes_both_files_together(store, safetensors_blob):
    # The other half of the atomicity claim: once save() returns, both files
    # exist side by side — no window where one is present without the other.
    store.save("atomic-pair", safetensors_blob, kind=AdapterKind.CLIENT, hparams=LoRAHyperParams())
    vdir = store._adapter_dir("atomic-pair") / "v1"
    assert (vdir / "adapter.safetensors").exists()
    assert (vdir / "metadata.json").exists()
