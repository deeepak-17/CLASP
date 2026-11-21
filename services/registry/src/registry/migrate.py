"""Import a volume written by the file-based registry into the metadata database.

Before the database, a volume held everything in files::

    adapters/<name>/vN/metadata.json   # one AdapterMetadata per version
    adapters/<name>/active             # the active version number
    adapters/<name>/promotions.jsonl   # one PromotionDecision per line

``import_volume`` reads all three into the repository in one transaction:
parts before composites (a composite's part rows need the parts), versions in
order, then pointers, then decisions. Anything unreadable raises a
``StorageError`` naming the file, and the transaction leaves the database
empty. Weight files are not touched; ``metadata.json`` stays as a sidecar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from contracts import AdapterKind, AdapterMetadata, PromotionDecision

from .errors import StorageError
from .repository import MetadataRepository

_VERSION_DIR = re.compile(r"v(\d+)")


def _adapter_dirs(store_root: Path) -> list[Path]:
    adapters_dir = Path(store_root) / "adapters"
    if not adapters_dir.is_dir():
        return []
    return sorted(p for p in adapters_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _read_metadata(path: Path, name: str, version: int) -> AdapterMetadata:
    try:
        meta = AdapterMetadata.from_json(json.loads(path.read_text()))
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise StorageError(f"cannot import {path}: {e}") from e
    if (meta.ref.name, meta.ref.version) != (name, version):
        raise StorageError(
            f"cannot import {path}: it describes {meta.ref.name} v{meta.ref.version}"
        )
    return meta


def _read_versions(adapter_dir: Path) -> list[AdapterMetadata]:
    found = []
    for p in adapter_dir.iterdir():
        m = _VERSION_DIR.fullmatch(p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), p))
    return [_read_metadata(p / "metadata.json", adapter_dir.name, v) for v, p in sorted(found)]


def _read_active(adapter_dir: Path) -> int | None:
    path = adapter_dir / "active"
    if not path.exists():
        return None
    raw = path.read_text().strip()
    try:
        return int(raw)
    except ValueError as e:
        raise StorageError(f"corrupt active pointer in {path}: {raw!r}") from e


def _read_decisions(adapter_dir: Path) -> list[PromotionDecision]:
    path = adapter_dir / "promotions.jsonl"
    if not path.exists():
        return []
    decisions = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decisions.append(PromotionDecision.from_json(json.loads(line)))
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise StorageError(f"cannot import {path}:{lineno}: {e}") from e
    return decisions


def import_volume(store_root: Path, repo: MetadataRepository) -> dict:
    """Import every version, pointer and decision of a file-based volume.

    Returns how many of each were imported. Idempotent: a repository that
    already holds adapters is left alone and the counts are all zero.
    """
    counts = {"adapters": 0, "versions": 0, "pointers": 0, "decisions": 0}
    with repo.transaction():
        if repo.list_adapters():
            return counts
        dirs = _adapter_dirs(Path(store_root))
        metas = {d.name: _read_versions(d) for d in dirs}
        # parts first: every composite's part rows must find their versions
        ordered = sorted(
            (m for ms in metas.values() for m in ms),
            key=lambda m: (m.ref.kind is AdapterKind.COMPOSITE, m.ref.name, m.ref.version),
        )
        for meta in ordered:
            repo.add_version(meta)
        counts["versions"] = len(ordered)
        counts["adapters"] = sum(1 for ms in metas.values() if ms)
        for d in dirs:
            if not metas[d.name]:
                continue  # a directory with no version holds nothing to import
            active = _read_active(d)
            if active is not None:
                try:
                    repo.set_active(d.name, active)
                except StorageError as e:
                    raise StorageError(f"cannot import {d / 'active'}: {e}") from e
                counts["pointers"] += 1
            for decision in _read_decisions(d):
                repo.record_decision(d.name, decision)
                counts["decisions"] += 1
    return counts
