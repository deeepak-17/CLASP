"""Shared configuration helpers for the registry."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "CLASP_REGISTRY_DATA"
DEFAULT_DATA_ROOT = "/data/registry"

#: D7 (U3): a client's whole federated run may spend at most this epsilon at
#: this delta. The numbers ``security.dp.ledger`` enforces on the edge and the
#: cluster; the registry records what was spent and reports it against them.
RUN_EPSILON_BUDGET = 8.0
RUN_DELTA = 1e-5
RUN_ROUNDS = 2


def data_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Explicit root, else ``CLASP_REGISTRY_DATA``, else ``/data/registry``.

    An empty string counts as unset.
    """
    return Path(root or os.environ.get(DATA_ENV) or DEFAULT_DATA_ROOT)
