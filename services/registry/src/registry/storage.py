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
from dataclasses import asdict
from pathlib import Path

from contracts import (
    AdapterKind,
    AdapterMetadata,
    AdapterRef,
    AggregationMethod,
    LoRAHyperParams,
    PrivacySpec,
    PromotionAction,
    PromotionDecision,
    utcnow_iso,
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ST_MAGIC_MAX_HEADER = 100_000_000  # sanity bound on safetensors header length


class StorageError(Exception):
    """Base class for registry storage errors."""


class AdapterNotFound(StorageError):
    pass


class VersionExists(StorageError):
    """Refused to overwrite an existing immutable version (D9)."""


def _validate_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise StorageError(f"invalid adapter name: {name!r}")
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

    def previous_version(self, name: str, version: int) -> int | None:
        """The version immediately before ``version`` in this adapter's history.

        Used by the D5 rule as the rollback target. None if ``version`` is the
        first version (nothing to fall back to).
        """
        vs = self.list_versions(name)
        earlier = [v for v in vs if v < version]
        return max(earlier) if earlier else None

    # -- promotion audit trail (D5/D9) --------------------------------------- #
    def record_promotion(self, name: str, decision: PromotionDecision) -> None:
        """Append a PromotionDecision to this adapter's audit log (never rewritten)."""
        log = self._adapter_dir(name) / "promotions.jsonl"
        with log.open("a") as f:
            f.write(json.dumps(_promotion_decision_to_dict(decision)) + "\n")

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
    ) -> AdapterMetadata:
        """Write a new immutable version. Assigns the next version number."""
        _validate_name(name)
        if not is_safetensors(payload):
            raise StorageError("payload is not a valid safetensors blob")

        version = (self.latest_version(name) if self._adapter_dir(name).exists() else 0) + 1
        vdir = self._version_dir(name, version)
        try:
            # M4: mkdir is the atomic claim on a version. If a concurrent save
            # already took it, FileExistsError -> VersionExists (a clean 409),
            # never a silent overwrite (D9) or an unhandled 500.
            vdir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            raise VersionExists(f"{name} v{version} already exists") from e

        (vdir / "adapter.safetensors").write_bytes(payload)
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
        )
        (vdir / "metadata.json").write_text(json.dumps(_metadata_to_dict(meta), indent=2))
        if set_active:
            self.set_active(name, version)
        return meta


# --------------------------------------------------------------------------- #
# (de)serialization helpers — dataclasses <-> JSON-safe dicts
# --------------------------------------------------------------------------- #
def _metadata_to_dict(meta: AdapterMetadata) -> dict:
    d = asdict(meta)
    d["ref"]["kind"] = meta.ref.kind.value
    d["aggregation"] = meta.aggregation.value if meta.aggregation else None
    # tuples -> lists for JSON
    d["hparams"]["target_modules"] = list(meta.hparams.target_modules)
    d["source_clients"] = list(meta.source_clients)
    return d


def _metadata_from_dict(d: dict) -> AdapterMetadata:
    ref = d["ref"]
    return AdapterMetadata(
        ref=AdapterRef(
            name=ref["name"],
            version=ref["version"],
            kind=AdapterKind(ref["kind"]),
            cluster_id=ref.get("cluster_id"),
        ),
        hparams=LoRAHyperParams(
            rank=d["hparams"]["rank"],
            lora_alpha=d["hparams"]["lora_alpha"],
            dropout=d["hparams"]["dropout"],
            target_modules=tuple(d["hparams"]["target_modules"]),
            alpha=d["hparams"]["alpha"],
            beta=d["hparams"]["beta"],
        ),
        privacy=PrivacySpec(**d["privacy"]),
        aggregation=AggregationMethod(d["aggregation"]) if d.get("aggregation") else None,
        round=d.get("round"),
        seed=d["seed"],
        sha256=d["sha256"],
        num_bytes=d["num_bytes"],
        source_clients=tuple(d.get("source_clients", ())),
        created_at=d["created_at"],
        contracts_version=d.get("contracts_version", "1.0.0"),
    )


def _promotion_decision_to_dict(decision: PromotionDecision) -> dict:
    d = asdict(decision)
    d["action"] = decision.action.value
    d["adapter"]["kind"] = decision.adapter.kind.value
    return d


def _promotion_decision_from_dict(d: dict) -> PromotionDecision:
    ref = d["adapter"]
    return PromotionDecision(
        adapter=AdapterRef(
            name=ref["name"],
            version=ref["version"],
            kind=AdapterKind(ref["kind"]),
            cluster_id=ref.get("cluster_id"),
        ),
        action=PromotionAction(d["action"]),
        active_version_after=d["active_version_after"],
        reason=d["reason"],
        timestamp=d["timestamp"],
    )
