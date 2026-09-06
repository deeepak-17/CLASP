"""Root conftest — makes the repository importable without installation.

CLASP-P5 uses a flat top-level package layout (``utils/``, ``interfaces/``,
``corpus/``, ``partitions/``, ``evaluation/``) as specified by the module
plan. Those names are deliberately *not* installed into ``site-packages``:
``utils`` and ``interfaces`` are common enough that installing them globally
would risk shadowing another package in a teammate's environment.

Instead the repository root is placed on ``sys.path`` here (for pytest) and by
``scripts/_bootstrap.py`` (for the CLIs). ``make`` targets set ``PYTHONPATH``
for the same reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
