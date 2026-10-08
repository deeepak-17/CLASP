"""Make the ``evaluation`` package importable when a script is run directly.

Imported first by every CLI in ``scripts/``. The normal setup is
``pip install -e contracts -e services/evaluation`` (what CI does), in which
case this is a no-op. In a bare checkout it *appends* the two ``src``
directories, so an installed copy always wins and nothing is shadowed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_ROOT.parent.parent

#: import name -> source directory of a package the scripts need.
_SOURCES = {
    "evaluation": MODULE_ROOT / "src",
    "contracts": REPO_ROOT / "contracts" / "src",
}

for _name, _src in _SOURCES.items():
    if importlib.util.find_spec(_name) is None and _src.is_dir() and str(_src) not in sys.path:
        sys.path.append(str(_src))
