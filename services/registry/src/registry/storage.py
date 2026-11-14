"""Versioned, append-only adapter storage for the CLASP State Registry (P4).

Layout on the persistent docker volume (``CLASP_REGISTRY_DATA``)::

    <root>/
      registry.db                   # metadata: versions, active pointers,
                                    # decisions (unless CLASP_REGISTRY_DB is set)
      adapters/
        <name>/
          v1/
            adapter.safetensors     # opaque payload (client-serialized)
            metadata.json           # AdapterMetadata sidecar, never read on requests
          v2/ ...
          gc.jsonl                  # retention audit log (still a file)

Volumes written before the database also hold ``active`` and
``promotions.jsonl`` per adapter; ``registry.migrate`` imports them once, on
the first start against an empty database, and they are not written again.

Invariants (MASTER_PLAN D9):
  * safetensors blobs are **never overwritten in place** — a new version is a
    new directory. ``save`` refuses to clobber an existing version.
  * every payload gets a sha256 digest recorded in its metadata for audit.

The registry treats adapter tensors as opaque bytes, so storage never needs
torch/numpy/safetensors installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import Engine

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

from .config import data_root
from .db import DB_ENV, create_all, make_engine
from .durable import fsync_path as _fsync
from .errors import (  # noqa: F401  (re-exported: callers import these from registry.storage)
    AdapterNotFound,
    InvalidAdapterName,
    KindMismatch,
    StorageError,
    VersionExists,
)
from .migrate import import_volume
from .repository import MetadataRepository, SqlMetadataRepository

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: Header size bound. A real 24-layer adapter's header is ~30 KB; parsing a
#: hostile 100 MB header would cost hundreds of MB of RAM.
_ST_MAX_HEADER = 8 * 1024 * 1024
_ST_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2,
                   "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8,
                   "U64": 8, "F64": 8}


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise InvalidAdapterName(f"invalid adapter name: {name!r}")
    return name


def _tensor_span(entry: object) -> tuple[int, int] | None:
    """(begin, end) of a well-formed header entry, else None."""
    if not isinstance(entry, dict):
        return None
    dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
    if dtype not in _ST_DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list):
        return None
    if len(offsets) != 2 or not all(isinstance(x, int) and x >= 0 for x in [*shape, *offsets]):
        return None
    count = 1
    for dim in shape:
        count *= dim
    begin, end = offsets
    return (begin, end) if end - begin == count * _ST_DTYPE_BYTES[dtype] else None


def is_safetensors(payload: bytes) -> bool:
    """Structural check without numpy: bounded JSON header whose tensors tile the
    data section exactly (contiguous, in bounds, sized by dtype x shape)."""
    if len(payload) < 8:
        return False
    header_len = int.from_bytes(payload[:8], "little")
    if header_len <= 0 or header_len > _ST_MAX_HEADER or 8 + header_len > len(payload):
        return False
    try:
        header = json.loads(payload[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(header, dict):
        return False
    spans = [_tensor_span(v) for k, v in header.items() if k != "__metadata__"]
    if any(span is None for span in spans):
        return False
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            return False
        cursor = end
    return cursor == len(payload) - 8 - header_len


def _db_url(root: Path) -> str:
    """``CLASP_REGISTRY_DB`` if set, else SQLite next to this store's adapters."""
    return os.environ.get(DB_ENV) or f"sqlite:///{(root / 'registry.db').as_posix()}"


class RegistryStore:
    """Adapter store: weight files under a persistent directory, metadata in a database.

    Without an explicit ``repo`` the store opens its own SQLite database (see
    ``_db_url``) and, while that database holds no adapters, imports the
    volume the file-based registry wrote (``registry.migrate``).
    """

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        repo: MetadataRepository | None = None,
    ) -> None:
        self.root = data_root(root)
        self.adapters_dir = self.root / "adapters"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)  # also creates root
        self._engine: Engine | None = None
        if repo is None:
            self._engine = make_engine(_db_url(self.root))
            create_all(self._engine)
            repo = SqlMetadataRepository(self._engine)
        self.repo: MetadataRepository = repo
        try:
            if not repo.list_adapters():
                import_volume(self.root, repo)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Dispose of the engine this store opened (Windows: before removing its files)."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    # -- paths -------------------------------------------------------------- #
    def _adapter_dir(self, name: str) -> Path:
        return self.adapters_dir / _validate_name(name)

    def _version_dir(self, name: str, version: int) -> Path:
        return self._adapter_dir(name) / f"v{version}"

    # -- queries ------------------------------------------------------------ #
    def list_adapters(self) -> list[str]:
        return self.repo.list_adapters()

    def list_versions(self, name: str) -> list[int]:
        versions = self.repo.list_versions(_validate_name(name))
        # A name with a directory but no registered version (its first save
        # failed) lists no versions, as before; only a name never seen is unknown.
        if not versions and name not in self.repo.list_adapters() \
                and not self._adapter_dir(name).is_dir():
            raise AdapterNotFound(name)
        return versions

    def latest_version(self, name: str) -> int:
        vs = self.list_versions(name)
        return vs[-1] if vs else 0

    def get_metadata(self, name: str, version: int) -> AdapterMetadata:
        return self.repo.get_metadata(_validate_name(name), version)

    def load_payload(self, name: str, version: int) -> bytes:
        blob = self._version_dir(name, version) / "adapter.safetensors"
        if not blob.exists():
            raise AdapterNotFound(f"{name} v{version}")
        return blob.read_bytes()

    # -- active pointer (D5) -------------------------------------------------- #
    def get_active(self, name: str) -> int | None:
        return self.repo.get_active(_validate_name(name))

    def set_active(self, name: str, version: int) -> None:
        self.repo.set_active(_validate_name(name), version)

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

    # -- audit trail (D5/D9) -------------------------------------------------- #
    def append_log(self, name: str, log_name: str, line: str) -> None:
        """Append one JSON line to an adapter's append-only file log (e.g. gc.jsonl).

        Promotion decisions are not file logs any more; see ``record_promotion``.
        """
        log = self._adapter_dir(name) / log_name
        with log.open("a") as f:
            f.write(line + "\n")
        _fsync(log)

    def record_promotion(self, name: str, decision: PromotionDecision) -> None:
        """Record a PromotionDecision in this adapter's audit trail (never rewritten)."""
        self.repo.record_decision(_validate_name(name), decision)

    def list_promotions(self, name: str) -> list[PromotionDecision]:
        return self.repo.list_decisions(_validate_name(name))

    # -- mutation ----------------------------------------------------------- #
    def delete_version(self, name: str, version: int) -> None:
        """Remove one version — retention/GC only; refuses the active version
        and any version a composite is built from.

        The metadata row goes first, so the version disappears from every
        listing at once; the directory is then renamed to a hidden trash name
        (atomic) and removed.
        """
        self.repo.delete_version(_validate_name(name), version)
        vdir = self._version_dir(name, version)
        if not vdir.exists():
            return
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
        set_active: bool = False,
        composed_from: CompositeProvenance | None = None,
    ) -> AdapterMetadata:
        """Write a new immutable version. Assigns the next version number.

        The version is an inactive candidate unless ``set_active`` is true
        (operator use); a promote is what normally makes it live.
        """
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
            # Both files reach the disk BEFORE the rename makes them a version.
            _fsync(staging / "adapter.safetensors")
            _fsync(staging / "metadata.json")
            _fsync(staging)
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
        _fsync(adapter_dir)        # the rename that named vN
        _fsync(self.adapters_dir)  # the adapter dir itself, on its first save
        # The version exists once its metadata row commits. If that fails, the
        # published directory goes too, so no unregistered version is left.
        try:
            with self.repo.transaction():
                self.repo.add_version(meta)
                if set_active:
                    self.repo.set_active(name, version)
        except BaseException:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
        return meta


# --------------------------------------------------------------------------- #
# serialization helpers — thin wrappers over the contracts' own JSON codecs
# (contracts v1.1), used for the metadata.json sidecar and the API responses.
# --------------------------------------------------------------------------- #
def _metadata_to_dict(meta: AdapterMetadata) -> dict:
    return meta.to_json()


def _promotion_decision_to_dict(decision: PromotionDecision) -> dict:
    return decision.to_json()
