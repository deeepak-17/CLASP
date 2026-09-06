"""Filesystem helpers: atomic writes, JSON/JSONL streaming, and hashing.

Atomicity matters here for a concrete reason: P4's State Registry and the
dashboard both *read* artefacts this repository *writes*. A partially written
``manifest.json`` observed by a reader is an integration bug, so every write
goes through a temporary file plus :func:`os.replace`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from utils.errors import ClaspP5Error

_ENCODING = "utf-8"
_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write ``text`` to ``path`` atomically, creating parents as required."""
    target = Path(path).expanduser()
    ensure_dir(target.parent)

    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=_ENCODING, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def write_json(path: Path | str, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Serialise ``payload`` as UTF-8 JSON and write it atomically."""
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=_json_default)
    return atomic_write_text(path, text + "\n")


def read_json(path: Path | str) -> Any:
    """Read and parse a JSON document.

    Raises:
        ClaspP5Error: If the file is missing or is not valid JSON.
    """
    target = Path(path).expanduser()
    try:
        with target.open("r", encoding=_ENCODING) as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ClaspP5Error(f"JSON file not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ClaspP5Error(f"Malformed JSON in {target}: {exc}") from exc


def write_jsonl(path: Path | str, records: Iterable[Any]) -> int:
    """Write ``records`` as newline-delimited JSON. Returns the record count.

    The whole file is buffered in memory before the atomic swap. Corpus sizes
    in this project are on the order of thousands of records, so this is a
    deliberate simplicity/robustness trade-off rather than an oversight.
    """
    lines = [json.dumps(record, ensure_ascii=False, default=_json_default) for record in records]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return len(lines)


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Lazily yield parsed records from a newline-delimited JSON file.

    Blank lines are skipped. Malformed lines raise, with the line number
    included so bad corpus rows can be located immediately.
    """
    target = Path(path).expanduser()
    try:
        handle = target.open("r", encoding=_ENCODING)
    except FileNotFoundError as exc:
        raise ClaspP5Error(f"JSONL file not found: {target}") from exc

    with handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ClaspP5Error(f"Malformed JSONL at {target}:{lineno}: {exc}") from exc


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode(_ENCODING)).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    target = Path(path).expanduser()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ClaspP5Error(f"Cannot hash missing file: {target}") from exc
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Fallback encoder for types ``json`` cannot serialise natively."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
