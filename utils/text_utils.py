"""Source-text helpers used while normalising the collected code corpus."""

from __future__ import annotations

from pathlib import Path

_NULL_BYTE = b"\x00"
_BINARY_SNIFF_BYTES = 8192


def normalise_newlines(text: str) -> str:
    """Convert CRLF/CR line endings to LF.

    Line-ending noise would otherwise change content hashes and inflate
    duplicate counts across repositories authored on different platforms.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def looks_binary(path: Path | str) -> bool:
    """Heuristically detect a binary file by sniffing for NUL bytes."""
    try:
        with Path(path).open("rb") as handle:
            return _NULL_BYTE in handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True


def read_source_text(path: Path | str) -> str | None:
    """Read a UTF-8 source file, returning ``None`` if it is not decodable.

    Returning ``None`` rather than raising lets the collector skip and count
    unusable files without aborting a multi-thousand-file crawl.
    """
    target = Path(path)
    if looks_binary(target):
        return None
    try:
        return normalise_newlines(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, OSError):
        return None


def count_lines(text: str) -> int:
    """Total number of lines, counting a trailing newline as a terminator."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def count_code_lines(text: str) -> int:
    """Count non-blank, non-comment-only lines (a crude Python SLOC).

    Deliberately does not attempt to strip docstrings: it is a partition
    balancing signal, not a code-metrics product.
    """
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        total += 1
    return total


def truncate(text: str, limit: int = 120) -> str:
    """Single-line, length-capped rendering of ``text`` for log messages."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"
