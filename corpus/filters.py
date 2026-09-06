"""File selection for the D1 corpus.

Filtering is separated from acquisition so the rules are unit-testable against
synthetic trees with no repositories present, and so the reason a file was
dropped is recorded rather than inferred. Every rejection increments a named
counter which flows into the corpus manifest and the collection report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from corpus.models import FilterConfig
from utils.logging_utils import get_logger
from utils.text_utils import count_code_lines, count_lines, read_source_text

_LOG = get_logger(__name__)


class SkipReason(str, Enum):
    """Why a discovered file did not make it into the corpus."""

    EXCLUDED_GLOB = "excluded_glob"
    TOO_LARGE = "too_large"
    UNDECODABLE = "undecodable"
    TOO_FEW_CODE_LINES = "too_few_code_lines"
    DUPLICATE_CONTENT = "duplicate_content"
    EMPTY = "empty"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class AcceptedFile:
    """A file that passed every filter, with its measured properties."""

    path: Path
    relative_path: str
    content: str
    num_lines: int
    num_code_lines: int
    num_bytes: int
    content_sha256: str


def matches_any(relative_path: str, patterns: list[str]) -> bool:
    """Whether a POSIX-style relative path matches any glob in ``patterns``.

    ``PurePosixPath.full_match`` (Python 3.13+) is used when available because
    it implements ``**`` correctly; the fallback approximates it by also
    testing each path suffix, since :meth:`PurePath.match` anchors at the tail.
    """
    candidate = PurePosixPath(relative_path)
    for pattern in patterns:
        full_match = getattr(candidate, "full_match", None)
        if full_match is not None:
            if full_match(pattern):
                return True
            continue
        if candidate.match(pattern):  # pragma: no cover - Python < 3.13 path
            return True
    return False


class FileSelector:
    """Applies :class:`~corpus.models.FilterConfig` to a directory tree.

    Deduplication state is held per instance and shared across sources, so a
    file vendored into two projects is kept once and counted as a duplicate
    the second time. That matters directly for CLASP's isolation claim: the
    same bytes appearing in two clusters would make "Project A's code never
    reaches B" false by construction.
    """

    def __init__(self, config: FilterConfig) -> None:
        self._config = config
        self._seen_hashes: dict[str, str] = {}
        self._skips: Counter[str] = Counter()
        self._totals: Counter[str] = Counter()

    # --- public API --------------------------------------------------------
    def discover(self, root: Path, subpaths: list[str] | None = None) -> list[Path]:
        """List candidate files under ``root``, restricted to ``subpaths``.

        Results are sorted so collection order — and therefore every downstream
        hash — is deterministic.
        """
        search_roots: list[Path] = []
        for sub in subpaths or []:
            candidate = root / sub
            if candidate.is_dir():
                search_roots.append(candidate)
            else:
                _LOG.warning("Configured subpath does not exist, skipping: %s", candidate)
        if not search_roots:
            search_roots = [root]

        discovered: set[Path] = set()
        for search_root in search_roots:
            for pattern in self._config.include_globs:
                discovered.update(p for p in search_root.glob(pattern) if p.is_file())
        return sorted(discovered)

    def evaluate(self, path: Path, relative_path: str) -> AcceptedFile | SkipReason:
        """Apply every filter to one file.

        Returns:
            An :class:`AcceptedFile` on success, or the :class:`SkipReason`
            that rejected it. Rejections are counted internally.
        """
        if matches_any(relative_path, self._config.exclude_globs):
            return self._skip(SkipReason.EXCLUDED_GLOB, relative_path)

        try:
            size = path.stat().st_size
        except OSError:
            return self._skip(SkipReason.UNREADABLE, relative_path)

        if size == 0:
            return self._skip(SkipReason.EMPTY, relative_path)
        if size > self._config.max_file_bytes:
            return self._skip(SkipReason.TOO_LARGE, relative_path)

        text = read_source_text(path)
        if text is None:
            return self._skip(SkipReason.UNDECODABLE, relative_path)

        code_lines = count_code_lines(text)
        if code_lines < self._config.min_code_lines:
            return self._skip(SkipReason.TOO_FEW_CODE_LINES, relative_path)

        # Hash the normalised text, not the raw bytes, so CRLF/LF variants of
        # the same file deduplicate correctly.
        from utils.io_utils import sha256_text  # local import: avoids a cycle at module load

        digest = sha256_text(text)
        if self._config.drop_duplicate_content and digest in self._seen_hashes:
            _LOG.debug(
                "Duplicate content: %s already seen as %s", relative_path, self._seen_hashes[digest]
            )
            return self._skip(SkipReason.DUPLICATE_CONTENT, relative_path)
        self._seen_hashes[digest] = relative_path

        return AcceptedFile(
            path=path,
            relative_path=relative_path,
            content=text,
            num_lines=count_lines(text),
            num_code_lines=code_lines,
            num_bytes=len(text.encode("utf-8")),
            content_sha256=digest,
        )

    # --- statistics --------------------------------------------------------
    def skip_counts(self) -> dict[str, int]:
        """Rejection counts for the current source, keyed by reason."""
        return dict(self._skips)

    def total_skip_counts(self) -> dict[str, int]:
        """Rejection counts across every source seen by this selector."""
        return dict(self._totals)

    def reset_skip_counts(self) -> dict[str, int]:
        """Return and clear the per-source counters. Totals are unaffected."""
        counts = dict(self._skips)
        self._skips.clear()
        return counts

    @property
    def duplicates_dropped(self) -> int:
        """Total duplicate-content rejections across all sources."""
        return self._totals.get(SkipReason.DUPLICATE_CONTENT.value, 0)

    @property
    def unique_content_count(self) -> int:
        """Number of distinct content hashes admitted so far."""
        return len(self._seen_hashes)

    # --- internals ---------------------------------------------------------
    def _skip(self, reason: SkipReason, relative_path: str) -> SkipReason:
        self._skips[reason.value] += 1
        self._totals[reason.value] += 1
        _LOG.debug("Skipped %s (%s)", relative_path, reason.value)
        return reason
