"""Make the repository importable when a script is run directly.

Imported first by every CLI in ``scripts/``. See the root ``conftest.py`` for
why CLASP-P5's flat top-level packages are not installed into site-packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
