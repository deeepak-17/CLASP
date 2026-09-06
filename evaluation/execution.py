"""Sandboxed execution of assembled programs — Week 3 support module.

The harness and the adapters already assemble a runnable program per
completion (``BenchmarkAdapter.assemble_program``); this module is the one
place that actually executes model-generated code to decide pass/fail.

Isolation model — read this before enabling ``scoring.execution_enabled``
--------------------------------------------------------------------------
Every program runs as its own **subprocess**, in a throwaway temporary
directory, with:

* a hard wall-clock timeout (``scoring.execution_timeout_seconds``);
* a restricted environment (no inherited secrets, minimal ``PATH``);
* best-effort POSIX resource limits (CPU seconds, address space, no core
  dumps, capped open-file count) applied via ``preexec_fn`` where the
  ``resource`` module is available.

This is **process-level** isolation, not container- or VM-level isolation:
it stops a runaway loop or an accidental fork bomb from hanging the harness,
and it stops the executed code from writing outside its temp directory by
convention, but it does **not** stop a deliberately malicious payload from
reading the filesystem, opening a socket, or otherwise using any syscall the
harness process itself is permitted to make. The Week-2 architecture notes
this in :mod:`evaluation.harness`: "executing model-generated code
additionally needs a sandbox policy P3 has to sign off on." That sign-off
has not happened as of Week 3 — P3's mTLS/DP-SGD workstream is scoped to the
training path, not eval execution. Running this against completions from a
trusted or mock generator (as every run in this repository does) is fine;
running it against arbitrary untrusted input without a stronger sandbox
(container, gVisor, seccomp profile) would not be.

``scoring.execution_enabled`` therefore defaults to ``false`` and must be
turned on deliberately.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logging_utils import get_logger

_LOG = get_logger(__name__)

#: Output captured beyond this many characters is truncated. Generated code
#: that misbehaves can print unboundedly; the harness must not OOM because of it.
_MAX_CAPTURED_OUTPUT = 4096

#: Best-effort ceiling on address space for the child process (POSIX only).
_MAX_ADDRESS_SPACE_BYTES = 1 << 30  # 1 GiB


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of running one assembled program."""

    passed: bool
    timed_out: bool
    exit_code: int | None
    duration_seconds: float
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "timed_out": self.timed_out,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "stderr_tail": self.stderr_tail,
        }


def _preexec() -> None:  # pragma: no cover - exercised only on POSIX child processes
    """Apply best-effort resource limits in the child, before exec.

    Runs inside the forked child, so any exception here would surface as an
    opaque subprocess failure; every call is therefore individually guarded.
    POSIX-only (``resource`` is unavailable on Windows, where this function
    is never installed as ``preexec_fn`` — see :func:`_preexec_fn_for_platform`).
    """
    import resource

    for limit, value in (
        (resource.RLIMIT_CPU, (10, 10)),
        (resource.RLIMIT_CORE, (0, 0)),
        (resource.RLIMIT_NOFILE, (64, 64)),
    ):
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_MAX_ADDRESS_SPACE_BYTES, _MAX_ADDRESS_SPACE_BYTES))
    except (ValueError, OSError):
        # Some platforms (notably macOS for certain processes) refuse RLIMIT_AS;
        # the timeout and CPU limit above are still in effect.
        pass


def _preexec_fn_for_platform():
    """Return :func:`_preexec`, or ``None`` where it cannot apply (non-POSIX)."""
    try:
        import resource  # noqa: F401

        return _preexec
    except ImportError:
        return None


def execute_program(source: str, *, timeout_seconds: float = 10.0) -> ExecutionOutcome:
    """Run ``source`` as a standalone Python program and report pass/fail.

    "Passed" means the process exited with status 0 within the timeout — the
    same signal HumanEval's own ``check()`` harness and MBPP's bare
    ``assert`` statements use: a failed ``assert`` or an uncaught exception
    exits non-zero, and the assembled program's tests are what determine
    that, not this function.

    Args:
        source: A complete, self-contained Python program (prompt +
            completion + tests, as produced by
            ``BenchmarkAdapter.assemble_program``).
        timeout_seconds: Wall-clock limit. Mirrors
            ``scoring.execution_timeout_seconds``.

    Returns:
        An :class:`ExecutionOutcome`. Never raises for a failure *of the
        executed program* — a syntax error, an infinite loop (via timeout)
        and a failed assertion are all ordinary "did not pass" outcomes, not
        exceptions from this function.
    """
    with tempfile.TemporaryDirectory(prefix="clasp-p5-exec-") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(source, encoding="utf-8")

        # Restricted environment: no inherited API keys/tokens, minimal PATH
        # so the candidate cannot invoke arbitrary tools found via a broad PATH.
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_preexec_fn_for_platform(),
            )
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            return ExecutionOutcome(
                passed=False,
                timed_out=True,
                exit_code=None,
                duration_seconds=round(time.perf_counter() - start, 6),
                stderr_tail=_truncate(stderr or f"execution exceeded {timeout_seconds}s"),
            )
        except OSError as exc:  # interpreter missing, permission error, etc.
            return ExecutionOutcome(
                passed=False,
                timed_out=False,
                exit_code=None,
                duration_seconds=round(time.perf_counter() - start, 6),
                stderr_tail=_truncate(f"{type(exc).__name__}: {exc}"),
            )
        duration = round(time.perf_counter() - start, 6)

    return ExecutionOutcome(
        passed=completed.returncode == 0,
        timed_out=False,
        exit_code=completed.returncode,
        duration_seconds=duration,
        stderr_tail=_truncate(completed.stderr),
    )


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CAPTURED_OUTPUT:
        return text
    return text[-_MAX_CAPTURED_OUTPUT:]
