import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The CLASP checkout this package runs from (services/edge/src/edge -> repo).
# CLASP_REPO_ROOT overrides it, e.g. in a container laid out differently.
REPO_ROOT = Path(os.environ.get("CLASP_REPO_ROOT") or Path(__file__).resolve().parents[4])


def portable_path(path) -> Optional[str]:
    """A path as a manifest should record it (D9): repo-relative and POSIX.

    Paths inside the checkout become relative to it, so a manifest means the
    same thing on every machine. Anything outside it is recorded relative to
    the home directory ("~/..."), so no user's absolute layout is committed;
    non-path strings (e.g. "stub:random") pass through unchanged.
    """
    if path is None:
        return None
    s = str(path)
    if ":" in s and not Path(s).drive:      # a tag such as stub:random
        return s
    p = Path(s).expanduser()
    if not p.is_absolute():
        return p.as_posix()
    p = p.resolve()
    for root, prefix in ((REPO_ROOT.resolve(), ""), (Path.home().resolve(), "~/")):
        try:
            return prefix + p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.as_posix()


@dataclass
class ModelProfile:
    name: str
    model_id: str
    max_new_tokens: int = 256

PROFILES = {
    "dev": ModelProfile(
        name='dev',
        model_id='deepseek-ai/deepseek-coder-1.3b-base'
    ),
    "target": ModelProfile(
        name='target',
        model_id='deepseek-ai/deepseek-coder-6.7b-base'
    )
}
