"""A tiny Markdown report builder.

Several Week-1/Week-2 deliverables are *documents* (dataset survey, dry-run
report, partition validation report). Generating them from the same data
structures the code operates on keeps the write-ups from drifting away from
what the pipeline actually did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.io_utils import atomic_write_text
from utils.timing import utc_timestamp


def _cell(value: Any) -> str:
    """Render a table cell, escaping pipes so the table cannot break."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


@dataclass
class MarkdownReport:
    """Accumulates Markdown blocks and writes them out atomically."""

    title: str
    subtitle: str | None = None
    _blocks: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._blocks.append(f"# {self.title}")
        header = [f"**Generated:** {utc_timestamp()}"]
        if self.subtitle:
            header.insert(0, f"_{self.subtitle}_")
        self._blocks.append("  \n".join(header))

    # --- block builders -------------------------------------------------
    def heading(self, text: str, level: int = 2) -> "MarkdownReport":
        self._blocks.append(f"{'#' * max(1, min(level, 6))} {text}")
        return self

    def paragraph(self, text: str) -> "MarkdownReport":
        self._blocks.append(text.strip())
        return self

    def bullets(self, items: Sequence[str]) -> "MarkdownReport":
        if items:
            self._blocks.append("\n".join(f"- {item}" for item in items))
        return self

    def key_values(self, mapping: dict[str, Any]) -> "MarkdownReport":
        if mapping:
            self._blocks.append("\n".join(f"- **{k}:** {_cell(v)}" for k, v in mapping.items()))
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> "MarkdownReport":
        """Append a pipe table. A no-op when ``rows`` is empty."""
        if not rows:
            self._blocks.append("_No rows._")
            return self
        head = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
        self._blocks.append("\n".join([head, sep, *body]))
        return self

    def code(self, text: str, language: str = "") -> "MarkdownReport":
        self._blocks.append(f"```{language}\n{text.rstrip()}\n```")
        return self

    def rule(self) -> "MarkdownReport":
        self._blocks.append("---")
        return self

    def status_line(self, ok: bool, message: str) -> "MarkdownReport":
        self._blocks.append(f"**{'PASS' if ok else 'FAIL'}** — {message}")
        return self

    # --- output ----------------------------------------------------------
    def render(self) -> str:
        return "\n\n".join(block for block in self._blocks if block) + "\n"

    def write(self, path: Path | str) -> Path:
        """Write the report to ``path`` and return the resolved path."""
        return atomic_write_text(path, self.render())
