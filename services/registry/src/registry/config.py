"""Shared configuration helpers for the registry."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "CLASP_REGISTRY_DATA"
DEFAULT_DATA_ROOT = "/data/registry"


def data_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Explicit root, else ``CLASP_REGISTRY_DATA``, else ``/data/registry``.

    An empty string counts as unset.
    """
    return Path(root or os.environ.get(DATA_ENV) or DEFAULT_DATA_ROOT)
