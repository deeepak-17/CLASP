"""Canonical filesystem locations for the CLASP-P5 module (``services/evaluation``).

Every module resolves paths through :func:`project_paths` rather than
hard-coding relative strings, so scripts behave identically regardless of the
working directory they are invoked from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_ENV_ROOT = "CLASP_P5_ROOT"


#: The ``evaluation`` package directory (``services/evaluation/src/evaluation``).
_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _discover_root() -> Path:
    """Locate the module root (``services/evaluation``).

    Resolution order:

    1. ``$CLASP_P5_ROOT`` if set (used by CI and container runs).
    2. The nearest ancestor of this file holding both ``configs/`` and
       ``pyproject.toml`` — the module root in a source checkout or an
       editable install.
    3. Three levels above the package (``src/evaluation`` -> module root), as a
       last resort.
    """
    # Module root = services/evaluation; every relative path in configs resolves from here.
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "configs").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return _PACKAGE_DIR.parent.parent


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable view of the directories CLASP-P5 reads from and writes to."""

    root: Path

    # --- configuration -------------------------------------------------
    @property
    def configs(self) -> Path:
        return self.root / "configs"

    # --- data ----------------------------------------------------------
    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def raw_data(self) -> Path:
        """Untouched source material (cloned repositories, downloads)."""
        return self.datasets / "raw"

    @property
    def processed_data(self) -> Path:
        """Normalised corpus records ready for partitioning."""
        return self.datasets / "processed"

    @property
    def dataset_metadata(self) -> Path:
        """Corpus manifests and provenance records."""
        return self.datasets / "metadata"

    @property
    def partitions(self) -> Path:
        """Partition shards and the partition manifest (Week 2 output)."""
        return self.datasets / "partitions"

    # --- evaluation ------------------------------------------------------
    @property
    def evaluation(self) -> Path:
        """The harness package; benchmark task files live beside its adapters."""
        return _PACKAGE_DIR / "eval_harness"

    @property
    def eval_results(self) -> Path:
        return self.root / "results"

    # --- documentation & artefacts ---------------------------------------
    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def schemas(self) -> Path:
        return _PACKAGE_DIR / "interfaces" / "schemas"

    def relative(self, path: Path | str) -> str:
        """Render ``path`` relative to the module root for readable logging."""
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)


@lru_cache(maxsize=1)
def project_paths() -> ProjectPaths:
    """Return the cached :class:`ProjectPaths` for this checkout."""
    return ProjectPaths(root=_discover_root())
