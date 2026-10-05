"""Versioned, append-only adapter storage for the CLASP State Registry (P4).

Layout on the persistent docker volume (``CLASP_REGISTRY_DATA``)::

    <root>/
      adapters/
        <name>/
          v1/
            adapter.safetensors     # opaque payload (client-serialized)
            metadata.json           # AdapterMetadata, immutable
          v2/ ...
          active                    # text file: the active version int (D5 promotion)

Invariants (MASTER_PLAN D9):
  * safetensors blobs are **never overwritten in place** — a new version is a
    new directory. ``save`` refuses to clobber an existing version.
  * every payload gets a sha256 digest recorded in its metadata for audit.

Storage is intentionally stdlib-only: the registry treats adapter tensors as
opaque bytes, so this service never needs torch/numpy/safetensors installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from contracts import (
    AdapterKind,
    AdapterMetadata,
    AdapterRef,
    AggregationMethod,
    CompositeProvenance,
    LoRAHyperParams,
    PrivacySpec,
    PromotionDecision,
    utcnow_iso,
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ST_MAGIC_MAX_HEADER = 100_000_000  # sanity bound on safetensors header length


class StorageError(Exception):
    """Base class for registry storage errors."""


class AdapterNotFound(StorageError):
    pass


class KindMismatch(StorageError):
    """A name's versions all share one AdapterKind; a save tried to change it."""


class InvalidAdapterName(StorageError):
    """Name fails the path-safe pattern — a client error, never a 500."""


class VersionExists(StorageError):
    """Refused to overwrite an existing immutable version (D9)."""


def _validate_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise InvalidAdapterName(f"invalid adapter name: {name!r}")
    return name


def is_safetensors(payload: bytes) -> bool:
    """Cheap structural check: 8-byte LE header length + valid JSON header."""
    if len(payload) < 8:
        return False
    header_len = int.from_bytes(payload[:8], "little")
    if header_len <= 0 or header_len > _ST_MAGIC_MAX_HEADER or 8 + header_len > len(payload):
        return False
    try:
        json.loads(payload[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


class RegistryStore:
    """Filesystem-backed adapter store rooted at a persistent directory."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        root = root or os.environ.get("CLASP_REGISTRY_DATA", "/data/registry")
        self.root = Path(root)
        self.adapters_dir = self.root / "adapters"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------- #
    def _adapter_dir(self, name: str) -> Path:
        return self.adapters_dir / _validate_name(name)

    def _version_dir(self, name: str, version: int) -> Path:
        return self._adapter_dir(name) / f"v{version}"

    # -- queries ------------------------------------------------------------ #
    def list_adapters(self) -> list[str]:
        if not self.adapters_dir.exists():
            return []
        return sorted(p.name for p in self.adapters_dir.iterdir() if p.is_dir())

    def list_versions(self, name: str) -> list[int]:
        d = self._adapter_dir(name)
        if not d.exists():
            raise AdapterNotFound(name)
        vs = [int(p.name[1:]) for p in d.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        return sorted(vs)

    def latest_version(self, name: str) -> int:
        vs = self.list_versions(name)
        return vs[-1] if vs else 0

    def get_metadata(self, name: str, version: int) -> AdapterMetadata:
        meta_path = self._version_dir(name, version) / "metadata.json"
        if not meta_path.exists():
            raise AdapterNotFound(f"{name} v{version}")
        return _metadata_from_dict(json.loads(meta_path.read_text()))

    def load_payload(self, name: str, version: int) -> bytes:
        blob = self._version_dir(name, version) / "adapter.safetensors"
        if not blob.exists():
            raise AdapterNotFound(f"{name} v{version}")
        return blob.read_bytes()

    # -- active pointer (promotion groundwork, D5 lands fully in W5) --------- #
    def get_active(self, name: str) -> int | None:
        p = self._adapter_dir(name) / "active"
        if not p.exists():
            return None
        raw = p.read_text().strip()
        try:  # L1: a corrupt/hand-edited pointer must not 500 the rollback path
            return int(raw)
        except ValueError as e:
            raise StorageError(f"corrupt active pointer for {name!r}: {raw!r}") from e

    def set_active(self, name: str, version: int) -> None:
        if version not in self.list_versions(name):
            raise AdapterNotFound(f"{name} v{version}")
        # L2: atomic write (temp + os.replace) so the pointer is never truncated
        # mid-write — the D5 rollback / D11 restore path must stay reliable.
        target = self._adapter_dir(name) / "active"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(str(version))
        os.replace(tmp, target)

    def check_kind(self, name: str, kind: AdapterKind) -> None:
        """Raise KindMismatch if ``name`` already holds versions of another kind."""
        if name not in self.list_adapters():
            return
        versions = self.list_versions(name)
        if not versions:
            return
        existing = self.get_metadata(name, versions[-1]).ref.kind
        if existing is not kind:
            raise KindMismatch(
                f"{name} holds kind {existing.value!r}; refusing to add a {kind.value!r} version"
            )

    def previous_version(self, name: str, version: int) -> int | None:
        """The version immediately before ``version`` in this adapter's history.

        Used by the D5 rule as the rollback target. None if ``version`` is the
        first version (nothing to fall back to).
        """
        vs = self.list_versions(name)
        earlier = [v for v in vs if v < version]
        return max(earlier) if earlier else None

    # -- promotion audit trail (D5/D9) --------------------------------------- #
    def append_log(self, name: str, log_name: str, line: str) -> None:
        """Append one JSON line to an adapter's append-only log."""
        with (self._adapter_dir(name) / log_name).open("a") as f:
            f.write(line + "\n")

    def record_promotion(self, name: str, decision: PromotionDecision) -> None:
        """Append a PromotionDecision to this adapter's audit log (never rewritten)."""
        self.append_log(name, "promotions.jsonl", json.dumps(_promotion_decision_to_dict(decision)))

    def list_promotions(self, name: str) -> list[PromotionDecision]:
        log = self._adapter_dir(name) / "promotions.jsonl"
        if not log.exists():
            return []
        return [
            _promotion_decision_from_dict(json.loads(line))
            for line in log.read_text().splitlines()
            if line.strip()
        ]

    # -- mutation ----------------------------------------------------------- #
    def delete_version(self, name: str, version: int) -> None:
        """Remove one version — retention/GC only; refuses the active version.

        The directory is renamed to a hidden trash name first (atomic), so
        ``list_versions`` never sees a half-deleted version, then removed.
        """
        vdir = self._version_dir(name, version)
        if not vdir.exists():
            raise AdapterNotFound(f"{name} v{version}")
        if self.get_active(name) == version:
            raise StorageError(f"refusing to delete the active version {name} v{version}")
        trash = Path(tempfile.mkdtemp(prefix=f".trash-v{version}-", dir=self._adapter_dir(name)))
        os.replace(vdir, trash / "v")
        shutil.rmtree(trash, ignore_errors=True)

    def save(
        self,
        name: str,
        payload: bytes,
        *,
        kind: AdapterKind,
        hparams: LoRAHyperParams,
        privacy: PrivacySpec | None = None,
        aggregation: AggregationMethod | None = None,
        round: int | None = None,
        seed: int = 0,
        cluster_id: str | None = None,
        source_clients: tuple[str, ...] = (),
        set_active: bool = True,
        composed_from: CompositeProvenance | None = None,
    ) -> AdapterMetadata:
        """Write a new immutable version. Assigns the next version number."""
        _validate_name(name)
        if not is_safetensors(payload):
            raise StorageError("payload is not a valid safetensors blob")
        if (kind is AdapterKind.COMPOSITE) != (composed_from is not None):
            raise StorageError(
                "composed_from is required for composite adapters and only allowed on them"
            )
        self.check_kind(name, kind)

        adapter_dir = self._adapter_dir(name)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        version = self.latest_version(name) + 1
        vdir = self._version_dir(name, version)

        # M5: stage both files in a hidden, uniquely-named directory next to
        # the version tree, then publish them with a single directory rename.
        # A crash before the rename leaves only the (unlisted) staging dir —
        # never a version directory with one file missing (Thu target).
        staging = Path(tempfile.mkdtemp(prefix=f".tmp-v{version}-", dir=adapter_dir))
        try:
            (staging / "adapter.safetensors").write_bytes(payload)
            meta = AdapterMetadata(
                ref=AdapterRef(name=name, version=version, kind=kind, cluster_id=cluster_id),
                hparams=hparams,
                privacy=privacy or PrivacySpec(),
                aggregation=aggregation,
                round=round,
                seed=seed,
                sha256=hashlib.sha256(payload).hexdigest(),
                num_bytes=len(payload),
                source_clients=tuple(source_clients),
                created_at=utcnow_iso(),
                composed_from=composed_from,
            )
            (staging / "metadata.json").write_text(json.dumps(_metadata_to_dict(meta), indent=2))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        try:
            # os.rename onto an existing (non-empty) vdir fails, which we
            # still turn into a clean VersionExists (D9) — same external
            # semantics as the old mkdir-based claim, atomic claim moved here.
            os.rename(staging, vdir)
        except OSError as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise VersionExists(f"{name} v{version} already exists") from e
        if set_active:
            self.set_active(name, version)
        return meta


# --------------------------------------------------------------------------- #
# (de)serialization helpers — thin wrappers over the contracts' own JSON codecs
# (contracts v1.1), kept as module functions so tests can fault-inject them.
# --------------------------------------------------------------------------- #
def _metadata_to_dict(meta: AdapterMetadata) -> dict:
    return meta.to_json()


def _metadata_from_dict(d: dict) -> AdapterMetadata:
    return AdapterMetadata.from_json(d)


def _promotion_decision_to_dict(decision: PromotionDecision) -> dict:
    return decision.to_json()


def _promotion_decision_from_dict(d: dict) -> PromotionDecision:
    return PromotionDecision.from_json(d)
