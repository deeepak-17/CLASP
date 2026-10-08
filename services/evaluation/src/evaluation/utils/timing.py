"""Timestamp and duration helpers.

All timestamps written to disk are ISO-8601 in UTC with an explicit ``Z``
suffix. P4's Registry sorts snapshots by timestamp, so a consistent,
lexicographically sortable format across modules is a contract requirement,
not a stylistic preference.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def file_timestamp() -> str:
    """Return a filename-safe UTC timestamp: ``YYYYMMDDTHHMMSSZ``."""
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


@dataclass
class Stopwatch(AbstractContextManager["Stopwatch"]):
    """Measure wall-clock duration of a block.

    Example:
        >>> with Stopwatch() as sw:
        ...     pass
        >>> sw.elapsed_seconds >= 0.0
        True
    """

    label: str = "elapsed"
    _start: float = field(default=0.0, init=False, repr=False)
    _end: float | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed; live value while the block is still running."""
        end = self._end if self._end is not None else time.perf_counter()
        return round(end - self._start, 6)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.label}={self.elapsed_seconds:.3f}s"
