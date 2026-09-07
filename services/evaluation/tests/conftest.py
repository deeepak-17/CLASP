"""Keep this CI leg pointed at the *installed* ``evaluation`` distribution.

The repository-root ``conftest.py`` prepends the CLASP-P5 source tree to
``sys.path`` so P5's flat top-level packages (``utils/``, ``evaluation/`` …)
import without being installed. That is right for the P5 test suite, but it
also makes ``import evaluation`` resolve to the source directory — which pulls
in the full harness and its dependencies — instead of the lightweight
``services/evaluation`` package this leg installs and is meant to smoke-test.

Drop the repo root back off ``sys.path`` here (rootdir conftests load first,
so this runs after the injection) so ``import evaluation`` resolves to the
installed distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
sys.path[:] = [entry for entry in sys.path if entry not in ("", _REPO_ROOT)]
