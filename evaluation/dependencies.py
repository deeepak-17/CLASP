"""Harness dependency preflight — Week 1, Thursday.

"Set up HumanEval/MBPP harness dependencies."

Rather than a bare ``requirements.txt``, dependencies are declared here as
data: what each package is *for*, which week first needs it, and whether its
absence is fatal now or merely blocks later work. Running
``scripts/check_dependencies.py`` then answers a question a requirements file
cannot — "is this machine ready for the work scheduled this week?" — and gives
a precise install command when it is not.

The distinction that matters: the Week-1 dry run must run on a laptop with no
GPU and no model weights, so ``torch``/``transformers``/``peft`` are declared
**optional** with the week they become required. Reporting them as hard
failures in Week 1 would be noise.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Final

from utils.logging_utils import get_logger
from utils.reporting import MarkdownReport

_LOG = get_logger(__name__)

#: Minimum interpreter this repository is developed and tested against.
MIN_PYTHON: Final[tuple[int, int]] = (3, 11)


@dataclass(frozen=True)
class DependencySpec:
    """One declared dependency and why the harness needs it."""

    package: str
    import_name: str
    purpose: str
    required_from_week: int
    required_now: bool
    install_hint: str = ""
    min_version: str | None = None

    def hint(self) -> str:
        return self.install_hint or f"pip install {self.package}"


@dataclass(frozen=True)
class DependencyStatus:
    """Observed state of one dependency on this machine."""

    spec: DependencySpec
    installed: bool
    version: str | None
    version_ok: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Satisfied, or not needed yet."""
        if not self.spec.required_now:
            return True
        return self.installed and self.version_ok

    @property
    def state(self) -> str:
        if not self.installed:
            return "MISSING" if self.spec.required_now else "not installed (not needed yet)"
        if not self.version_ok:
            return f"OUTDATED (need >= {self.spec.min_version})"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.spec.package,
            "purpose": self.spec.purpose,
            "required_from_week": self.spec.required_from_week,
            "required_now": self.spec.required_now,
            "installed": self.installed,
            "version": self.version,
            "state": self.state,
            "ok": self.ok,
        }


#: The harness dependency set. ``required_now`` reflects Week-1/2 scope.
DEPENDENCIES: Final[tuple[DependencySpec, ...]] = (
    DependencySpec(
        package="PyYAML",
        import_name="yaml",
        purpose="Parse configs/*.yaml. Every entry point depends on it.",
        required_from_week=1,
        required_now=True,
        min_version="6.0",
    ),
    DependencySpec(
        package="pytest",
        import_name="pytest",
        purpose="Run the unit and integration test suite.",
        required_from_week=1,
        required_now=True,
        min_version="8.0",
    ),
    DependencySpec(
        package="jsonschema",
        import_name="jsonschema",
        purpose=(
            "Structurally validate partition manifests and eval results against the "
            "interface contracts (Week-2 Friday sign-off)."
        ),
        required_from_week=2,
        required_now=True,
        min_version="4.21",
    ),
    DependencySpec(
        package="datasets",
        import_name="datasets",
        purpose=(
            "Fetch the published HumanEval/MBPP task sets from the Hugging Face Hub. "
            "The Week-1 dry run uses bundled fixtures and does not need it."
        ),
        required_from_week=3,
        required_now=False,
        install_hint="pip install datasets",
        min_version=None,
    ),
    DependencySpec(
        package="human-eval",
        import_name="human_eval",
        purpose=(
            "Reference Pass@k implementation, used to cross-check P5's own scorer "
            "when it lands in Week 3."
        ),
        required_from_week=3,
        required_now=False,
        install_hint="pip install human-eval",
    ),
    DependencySpec(
        package="torch",
        import_name="torch",
        purpose="Tensor runtime for P1's Edge Layer. Needed once the harness calls a real model.",
        required_from_week=3,
        required_now=False,
        install_hint="see https://pytorch.org/get-started/locally/ for the correct CUDA build",
    ),
    DependencySpec(
        package="transformers",
        import_name="transformers",
        purpose="Loads DeepSeek-Coder-6.7B. Owned by P1; P5 needs it only to call the merged model.",
        required_from_week=3,
        required_now=False,
    ),
    DependencySpec(
        package="peft",
        import_name="peft",
        purpose="LoRA adapter loading for the merged-model eval path. Owned by P1.",
        required_from_week=3,
        required_now=False,
    ),
)


def _parse_version(value: str) -> tuple[int, ...]:
    """Parse a dotted version into a comparable tuple, ignoring suffixes."""
    parts: list[int] = []
    for chunk in value.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def check_dependency(spec: DependencySpec) -> DependencyStatus:
    """Determine whether ``spec`` is satisfied in the current interpreter.

    Namespace packages are treated as *not installed*. This is not pedantry:
    the repository has a data-only ``datasets/`` directory, and with the repo
    root on ``sys.path`` a bare ``import datasets`` succeeds as an empty
    namespace package. Reporting that as "the Hugging Face ``datasets``
    library is present" would be a false pass.

    (At runtime the shadow is harmless — Python prefers a real package found
    later on the path over a namespace portion found earlier — so the fix
    belongs here in detection, not in the repository layout.)
    """
    try:
        found = importlib.util.find_spec(spec.import_name)
    except (ImportError, ValueError):
        found = None

    if found is None:
        return DependencyStatus(spec=spec, installed=False, version=None, version_ok=False)

    if found.origin is None:  # namespace package: directory without real code
        return DependencyStatus(
            spec=spec,
            installed=False,
            version=None,
            version_ok=False,
            error=f"resolved to a namespace package at {list(found.submodule_search_locations or [])}",
        )

    try:
        importlib.import_module(spec.import_name)
    except ImportError:
        return DependencyStatus(spec=spec, installed=False, version=None, version_ok=False)
    except Exception as exc:  # a broken install imports but raises
        return DependencyStatus(
            spec=spec, installed=False, version=None, version_ok=False, error=str(exc)
        )

    try:
        version = importlib.metadata.version(spec.package)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(importlib.import_module(spec.import_name), "__version__", None)

    version_ok = True
    if spec.min_version and version:
        version_ok = _parse_version(version) >= _parse_version(spec.min_version)

    return DependencyStatus(spec=spec, installed=True, version=version, version_ok=version_ok)


@dataclass(frozen=True)
class DependencyCheckResult:
    """Outcome of a full preflight."""

    python_version: str
    python_ok: bool
    platform: str
    statuses: list[DependencyStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.python_ok and all(status.ok for status in self.statuses)

    @property
    def blocking(self) -> list[DependencyStatus]:
        """Dependencies needed now that are not satisfied."""
        return [status for status in self.statuses if not status.ok]

    @property
    def deferred(self) -> list[DependencyStatus]:
        """Not-yet-required dependencies that are absent."""
        return [s for s in self.statuses if not s.spec.required_now and not s.installed]

    def install_command(self) -> str | None:
        """A single pip command that would resolve every blocking dependency."""
        packages = [s.spec.package for s in self.blocking if not s.spec.install_hint]
        if not packages:
            return None
        return "pip install " + " ".join(sorted(packages))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "python_version": self.python_version,
            "python_ok": self.python_ok,
            "platform": self.platform,
            "dependencies": [status.to_dict() for status in self.statuses],
        }


def check_dependencies(
    specs: tuple[DependencySpec, ...] = DEPENDENCIES,
) -> DependencyCheckResult:
    """Run the full preflight and log a one-line verdict."""
    python_ok = sys.version_info[:2] >= MIN_PYTHON
    if not python_ok:
        _LOG.error(
            "Python %s is below the required %d.%d",
            platform.python_version(),
            *MIN_PYTHON,
        )

    statuses = [check_dependency(spec) for spec in specs]
    result = DependencyCheckResult(
        python_version=platform.python_version(),
        python_ok=python_ok,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        statuses=statuses,
    )

    for status in statuses:
        log = _LOG.info if status.ok else _LOG.error
        log("  %-14s %-12s %s", status.spec.package, status.version or "-", status.state)

    if result.ok:
        _LOG.info("Dependency preflight PASSED for Week-1/Week-2 scope")
    else:
        _LOG.error("Dependency preflight FAILED: %d blocking issue(s)", len(result.blocking))
    return result


def render_dependency_report(result: DependencyCheckResult) -> MarkdownReport:
    """Render the Week-1 Thursday dependency status as Markdown."""
    report = MarkdownReport(
        title="CLASP-P5 · Evaluation Harness Dependency Report",
        subtitle="Week 1 · Thursday deliverable — HumanEval/MBPP harness dependencies",
    )

    report.heading("1. Verdict")
    report.status_line(
        result.ok,
        "environment satisfies every dependency required for Week-1/Week-2 scope"
        if result.ok
        else f"{len(result.blocking)} blocking dependency issue(s)",
    )
    report.key_values(
        {
            "Python": f"{result.python_version} (minimum {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
            "Platform": result.platform,
            "Required now": sum(1 for s in result.statuses if s.spec.required_now),
            "Deferred to later weeks": sum(1 for s in result.statuses if not s.spec.required_now),
        }
    )
    command = result.install_command()
    if command:
        report.paragraph("Resolve the blocking issues with:")
        report.code(command, "bash")

    report.heading("2. Dependency matrix")
    report.table(
        ["Package", "Required from", "Needed now", "Installed", "Version", "State", "Purpose"],
        [
            [
                status.spec.package,
                f"Week {status.spec.required_from_week}",
                status.spec.required_now,
                status.installed,
                status.version or "—",
                status.state,
                status.spec.purpose,
            ]
            for status in result.statuses
        ],
    )

    if result.deferred:
        report.heading("3. Deferred dependencies")
        report.paragraph(
            "Absent but not yet needed. These become required when the harness starts calling "
            "P1's real merged model and scoring Pass@k (Week 3), so they are declared now to "
            "avoid a surprise at integration time."
        )
        report.bullets(
            [f"`{s.spec.package}` — {s.spec.purpose} (`{s.spec.hint()}`)" for s in result.deferred]
        )

    report.rule()
    report.paragraph(
        "Generated by `evaluation/dependencies.py`. Regenerate with "
        "`python scripts/check_dependencies.py`."
    )
    return report
