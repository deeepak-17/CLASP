"""Durable file writes shared by the adapter store and the result store.

Atomic renames stop a crash from leaving a half-written file; fsync stops a
power cut from losing a write that was already acknowledged.
"""
from __future__ import annotations

import os
from pathlib import Path

_IS_WINDOWS = os.name == "nt"


def fsync_path(path: Path) -> None:
    """Force a file's contents, or a directory's entries, onto the disk.

    Atomic renames stop a crash from leaving a half-written version; this stops
    a power cut from losing a write the registry already acknowledged. Windows
    cannot open a directory handle (NTFS journals its own metadata), so
    directories are skipped there; files need a writable handle to flush.
    """
    is_dir = path.is_dir()
    if is_dir and _IS_WINDOWS:
        return
    fd = os.open(path, os.O_RDWR if _IS_WINDOWS else os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``path`` so it is either the old content or the new, and durable."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    fsync_path(tmp)
    os.replace(tmp, path)
    fsync_path(path.parent)
