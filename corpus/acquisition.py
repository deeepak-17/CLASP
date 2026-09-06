"""Obtaining source material for the D1 corpus.

Three strategies behind one :class:`SourceAcquirer` protocol:

``git``
    Shallow-clone each repository at its pinned ref. The default and the only
    mode that produces a corpus suitable for training.

``local``
    Use an existing checkout under ``datasets/raw``. For air-gapped machines
    and CI runners with a pre-seeded cache.

``synthetic``
    Generate a deterministic stand-in tree. Development and testing only —
    it lets the entire Week-1/Week-2 pipeline execute with no network and no
    disk-heavy clone, and every artefact derived from it is stamped
    ``synthetic: true`` so it can never be mistaken for the real corpus.

Separating acquisition from collection is what makes the pipeline testable:
:mod:`corpus.collector` never shells out, and the tests never touch a network.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from corpus.models import AcquisitionConfig, SourceConfig, SyntheticConfig
from utils.errors import CorpusError
from utils.io_utils import ensure_dir
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)

#: Refs that look like a moving branch rather than a pinned commit/tag.
_MOVING_REFS = {"main", "master", "develop", "HEAD", "trunk"}


@dataclass(frozen=True)
class AcquiredSource:
    """A source repository made available on the local filesystem."""

    cluster_id: str
    root: Path
    resolved_ref: str | None
    synthetic: bool = False


@runtime_checkable
class SourceAcquirer(Protocol):
    """Strategy for making a configured source available locally."""

    def acquire(self, source: SourceConfig, raw_dir: Path) -> AcquiredSource:
        """Materialise ``source`` under ``raw_dir`` and describe the result."""
        ...


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------
class GitAcquirer:
    """Shallow-clones each source at its pinned ref.

    Existing checkouts are reused rather than re-cloned; the resolved commit
    SHA is recorded either way so the manifest carries exact provenance even
    when a ``ref`` names a tag or a branch.
    """

    def __init__(self, config: AcquisitionConfig) -> None:
        self._config = config
        if shutil.which("git") is None:
            raise CorpusError(
                "acquisition.mode='git' requires the `git` executable, which was not found on PATH. "
                "Install git, or switch to mode 'local' / 'synthetic'."
            )

    def acquire(self, source: SourceConfig, raw_dir: Path) -> AcquiredSource:
        if not source.repo_url:
            raise CorpusError(f"Source '{source.cluster_id}' has no repo_url")

        target = source.checkout_dir(ensure_dir(raw_dir))

        if source.ref in _MOVING_REFS:
            _LOG.warning(
                "Source '%s' pins the moving ref '%s'; the corpus will not be reproducible "
                "across time. Prefer a tag or commit SHA.",
                source.cluster_id,
                source.ref,
            )

        if (target / ".git").is_dir():
            _LOG.info("Reusing existing checkout for '%s' at %s", source.cluster_id, target)
        else:
            self._clone(source, target)

        return AcquiredSource(
            cluster_id=source.cluster_id,
            root=target,
            resolved_ref=self._resolve_head(target) or source.ref,
        )

    # --- internals ---------------------------------------------------------
    def _clone(self, source: SourceConfig, target: Path) -> None:
        if target.exists():
            # A non-git directory sitting at the target path means a previous
            # run failed part-way. Refuse rather than mixing two trees.
            raise CorpusError(
                f"{target} exists but is not a git checkout. Remove it and retry, "
                f"or set acquisition.mode='local' to use it as-is."
            )

        command = ["git", "clone", "--quiet"]
        if self._config.git_depth > 0:
            command += ["--depth", str(self._config.git_depth)]
        if source.ref:
            command += ["--branch", source.ref]
        command += [source.repo_url or "", str(target)]

        _LOG.info("Cloning %s @ %s -> %s", source.repo_url, source.ref or "default", target)
        self._run(command, cwd=None, what=f"clone {source.cluster_id}")

    def _resolve_head(self, target: Path) -> str | None:
        """Return the checked-out commit SHA, or ``None`` if unavailable."""
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _LOG.warning("Could not resolve HEAD in %s: %s", target, exc)
            return None
        if completed.returncode != 0:
            _LOG.warning("git rev-parse failed in %s: %s", target, completed.stderr.strip())
            return None
        return completed.stdout.strip() or None

    def _run(self, command: list[str], *, cwd: Path | None, what: str) -> None:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CorpusError(f"Timed out after {self._config.timeout_seconds}s during {what}") from exc
        except OSError as exc:
            raise CorpusError(f"Failed to execute git during {what}: {exc}") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise CorpusError(f"git failed during {what} (exit {completed.returncode}): {detail}")


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------
class LocalAcquirer:
    """Uses a checkout that is already present on disk."""

    def acquire(self, source: SourceConfig, raw_dir: Path) -> AcquiredSource:
        target = source.checkout_dir(raw_dir)
        if not target.is_dir():
            raise CorpusError(
                f"acquisition.mode='local' but no checkout found for '{source.cluster_id}' at {target}. "
                f"Populate it, or switch to mode 'git'."
            )
        _LOG.info("Using local checkout for '%s' at %s", source.cluster_id, target)
        return AcquiredSource(cluster_id=source.cluster_id, root=target, resolved_ref=source.ref)


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------
class SyntheticAcquirer:
    """Generates a deterministic, offline stand-in tree per source.

    The generated modules are structurally varied per cluster (different base
    class names, method counts and docstring style) so that project-level
    partitioning has *something* real to separate, and so balance metrics are
    not trivially uniform. It is a plumbing fixture, not training data.
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config

    def acquire(self, source: SourceConfig, raw_dir: Path) -> AcquiredSource:
        root = ensure_dir(source.checkout_dir(ensure_dir(raw_dir)))
        package = ensure_dir(root / source.cluster_id)

        for index in range(self._config.files_per_source):
            path = package / f"module_{index:02d}.py"
            path.write_text(self._render(source, index), encoding="utf-8")

        _LOG.info(
            "Generated %d synthetic file(s) for '%s' at %s",
            self._config.files_per_source,
            source.cluster_id,
            root,
        )
        return AcquiredSource(
            cluster_id=source.cluster_id,
            root=root,
            resolved_ref=f"synthetic-{self._config.seed}",
            synthetic=True,
        )

    def _render(self, source: SourceConfig, index: int) -> str:
        """Deterministically render one synthetic module."""
        digest = hashlib.sha256(
            f"{self._config.seed}|{source.cluster_id}|{index}".encode("utf-8")
        ).hexdigest()
        # Vary shape per file so shard sizes are unequal, exercising the
        # balance metrics rather than trivially satisfying them.
        method_count = 2 + (int(digest[:2], 16) % 5)
        label = source.project_label

        lines = [
            f'"""Synthetic {label} module {index}.',
            "",
            "Generated by corpus.acquisition.SyntheticAcquirer for offline pipeline",
            "testing. NOT real source code and NOT valid training data.",
            f'Fingerprint: {digest[:16]}',
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "",
            f"class {label}Component{index:02d}:",
            f'    """Placeholder component {index} for cluster {source.cluster_id}."""',
            "",
            f"    fingerprint = {digest[:16]!r}",
            "",
        ]
        for method_index in range(method_count):
            lines += [
                f"    def operation_{method_index}(self, value: int) -> int:",
                f'        """Deterministic placeholder operation {method_index}."""',
                f"        total = value * {method_index + 1}",
                f"        if total > {100 * (method_index + 1)}:",
                f"            total -= {method_index + 1}",
                "        return total",
                "",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_acquirer(
    acquisition: AcquisitionConfig, synthetic: SyntheticConfig
) -> SourceAcquirer:
    """Return the acquirer matching ``acquisition.mode``."""
    if acquisition.mode == "git":
        return GitAcquirer(acquisition)
    if acquisition.mode == "local":
        return LocalAcquirer()
    if acquisition.mode == "synthetic":
        return SyntheticAcquirer(synthetic)
    raise CorpusError(f"Unsupported acquisition mode '{acquisition.mode}'")  # pragma: no cover
