"""Make the monorepo's shared packages importable from P5's flat layout.

P5's own packages (``utils``, ``interfaces``, ``corpus``, ``partitions``,
``eval_harness``) are imported from the repository root, which the root
``conftest.py`` and ``scripts/_bootstrap.py`` put on ``sys.path``. Two things
P5 depends on live in the ``services/<name>/src`` layout instead:

* ``contracts`` (``contracts/src``) — the frozen v1.0 wire types the registry
  parses;
* ``evaluation`` (``services/evaluation/src``) — P5's installable package,
  home of ``evaluation.completion``, the in-project metric the edge lane and
  the integration tests import.

The team convention is ``pip install -e contracts -e services/evaluation``
(CI does exactly that). :func:`ensure_shared_packages` is the fallback for a
checkout where they are not installed: it *appends* the ``src`` directories,
so an installed copy always wins and nothing is ever shadowed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: import name -> source directory, relative to the repository root.
SHARED_PACKAGES: dict[str, str] = {
    "contracts": "contracts/src",
    "evaluation": "services/evaluation/src",
}


def _is_regular_package(name: str) -> bool:
    """True if ``name`` resolves to a real package, not a bare namespace directory.

    The repository root contains a ``contracts/`` directory with no
    ``__init__.py``; with the root on ``sys.path`` that is enough for
    ``find_spec`` to report a namespace package, which is not importable in
    any useful sense.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin is not None


def ensure_shared_packages(root: Path | str) -> list[str]:
    """Append the ``src`` dir of each shared package that is not already importable.

    Returns the directories that were appended (empty when everything is
    installed), so callers can log what they did.
    """
    added: list[str] = []
    for name, relative in SHARED_PACKAGES.items():
        if _is_regular_package(name):
            continue
        src = str(Path(root) / relative)
        if Path(src).is_dir() and src not in sys.path:
            sys.path.append(src)
            added.append(src)
    return added
