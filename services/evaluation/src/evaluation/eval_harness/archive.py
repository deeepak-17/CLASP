"""Results archive — a checksummed inventory of everything P5 reports.

:func:`build_manifest` records the sha256 and size of every file under the
archived directories (results, reports, the D1 partition definition, the
report figures), so a later reader can confirm that the numbers they are
looking at are the ones that were reported. :func:`verify_manifest` re-hashes
and lists what changed, disappeared or appeared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluation.utils.io_utils import sha256_file
from evaluation.utils.timing import utc_timestamp

#: Directories (relative to the module root) the archive covers.
ARCHIVED_DIRS: tuple[str, ...] = (
    "results",
    "reports",
    "datasets/partitions",
    "datasets/metadata",
    "docs/report",
    "configs",
)

MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
_SKIP_NAMES = {MANIFEST_NAME, ".gitkeep", ".DS_Store"}


def _files(root: Path, dirs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for rel in dirs:
        base = root / rel
        if base.is_dir():
            out.extend(p for p in base.rglob("*") if p.is_file() and p.name not in _SKIP_NAMES)
    return sorted(out)


def build_manifest(root: Path, dirs: Sequence[str] = ARCHIVED_DIRS, *, label: str = "") -> dict[str, Any]:
    # Fingerprint every file: sha256 + size, keyed by path relative to services/evaluation.
    files = {
        str(p.relative_to(root)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in _files(root, dirs)
    }
    return {
        "label": label,
        "created_at": utc_timestamp(),
        "directories": list(dirs),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
    }


def verify_manifest(manifest: dict[str, Any], root: Path) -> dict[str, list[str]]:
    """``{"changed": [...], "missing": [...], "added": [...]}`` relative to ``manifest``."""
    recorded: dict[str, dict[str, Any]] = manifest["files"]
    current = {str(p.relative_to(root)): p for p in _files(root, manifest["directories"])}
    # Re-hash what is on disk now and compare with what was archived.
    changed = [rel for rel, meta in recorded.items() if rel in current and sha256_file(current[rel]) != meta["sha256"]]
    return {
        "changed": sorted(changed),
        "missing": sorted(set(recorded) - set(current)),
        "added": sorted(set(current) - set(recorded)),
    }
