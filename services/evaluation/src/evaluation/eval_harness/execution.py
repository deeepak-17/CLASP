"""Sandboxed execution of assembled programs.

This is the one place that executes model-generated code (HumanEval/MBPP
programs assembled by ``BenchmarkAdapter.assemble_program``) to decide
pass/fail. Two isolation backends, chosen by ``scoring.sandbox``:

``docker`` — the isolation D5's guard needs for untrusted model output.
    Each program runs in a fresh ``python:3.11-slim`` container with **no
    network** (``--network none``), a read-only root filesystem with the
    program mounted read-only, all Linux capabilities dropped,
    ``no-new-privileges``, an unprivileged user, and memory / CPU / process
    limits; a ``timeout`` inside the container kills the program at the
    wall-clock limit and the container is force-removed if the client is
    stuck. See :func:`docker_command`.

``process`` — the fallback where Docker is unavailable. A subprocess in a
    throwaway temp directory with a wall-clock timeout, a stripped environment
    and best-effort POSIX resource limits. It stops runaway loops and fork
    bombs from hanging the harness but does **not** stop a deliberately
    malicious program from reading files or opening sockets.

``auto`` (the configured default) uses ``docker`` when a Docker daemon
answers and ``process`` otherwise, and every outcome records which backend
actually ran (:attr:`ExecutionOutcome.isolation`), so a result can always be
traced to the isolation it was produced under.

``scoring.execution_enabled`` still defaults to ``false`` and must be turned
on deliberately.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.utils.errors import EvaluationError
from evaluation.utils.logging_utils import get_logger

_LOG = get_logger(__name__)

#: Output captured beyond this many characters is truncated. Generated code
#: that misbehaves can print unboundedly; the harness must not OOM because of it.
_MAX_CAPTURED_OUTPUT = 4096

#: Best-effort ceiling on address space for the child process (POSIX only).
_MAX_ADDRESS_SPACE_BYTES = 1 << 30  # 1 GiB

#: ``scoring.sandbox`` values.
SANDBOX_MODES = ("auto", "docker", "process")

#: Image every docker-sandboxed program runs in (has ``timeout`` from coreutils).
DOCKER_IMAGE = "python:3.11-slim"

#: Extra seconds the docker client may take beyond the program's own limit
#: (container start-up and teardown) before the container is force-removed.
_DOCKER_GRACE_SECONDS = 30.0


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of running one assembled program."""

    passed: bool
    timed_out: bool
    exit_code: int | None
    duration_seconds: float
    stderr_tail: str = ""
    #: The backend that actually ran the program: ``"docker"`` or ``"process"``.
    isolation: str = "process"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "timed_out": self.timed_out,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "stderr_tail": self.stderr_tail,
            "isolation": self.isolation,
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


@functools.lru_cache(maxsize=1)
def docker_available() -> bool:
    """Whether a Docker daemon answers (cached for the process lifetime)."""
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0 and bool(probe.stdout.strip())


def resolve_sandbox(mode: str) -> str:
    """``auto`` -> ``docker`` if available else ``process``; ``docker`` requires Docker."""
    if mode not in SANDBOX_MODES:
        raise EvaluationError(f"sandbox must be one of {SANDBOX_MODES}, got {mode!r}")
    if mode == "process":
        return "process"
    if docker_available():
        return "docker"
    if mode == "docker":
        raise EvaluationError("sandbox 'docker' requested but no Docker daemon is reachable")
    _warn_process_fallback()
    return "process"


@functools.lru_cache(maxsize=1)
def _warn_process_fallback() -> None:
    _LOG.warning(
        "No Docker daemon reachable: executing generated code with process-level isolation only "
        "(no network or filesystem confinement). Set scoring.sandbox to 'docker' to require it."
    )


def docker_command(host_dir: Path | str, timeout_seconds: float, *, name: str, image: str = DOCKER_IMAGE) -> list[str]:
    """The ``docker run`` invocation for one program in ``host_dir/candidate.py``."""
    return [
        "docker", "run", "--rm", "--name", name,
        # No network: generated code cannot download or send anything.
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1",
        "--pids-limit", "64",
        "--ulimit", "nofile=64:64",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{Path(host_dir).resolve()}:/sandbox:ro",
        "-w", "/tmp",
        image,
        # `timeout` inside the container kills a runaway program at the wall-clock limit.
        "timeout", "-s", "KILL", f"{timeout_seconds:g}",
        "python", "/sandbox/candidate.py",
    ]


def execute_program(source: str, *, timeout_seconds: float = 10.0, sandbox: str = "process") -> ExecutionOutcome:
    """Run ``source`` as a standalone Python program and report pass/fail.

    "Passed" means the program exited with status 0 within the timeout — the
    same signal HumanEval's own ``check()`` harness and MBPP's bare
    ``assert`` statements use.

    Args:
        source: A complete, self-contained Python program (prompt +
            completion + tests, as produced by
            ``BenchmarkAdapter.assemble_program``).
        timeout_seconds: Wall-clock limit for the program itself. Mirrors
            ``scoring.execution_timeout_seconds``.
        sandbox: ``"docker"``, ``"process"`` or ``"auto"`` (see the module
            docstring). Mirrors ``scoring.sandbox``.

    Returns:
        An :class:`ExecutionOutcome`. Never raises for a failure *of the
        executed program* — a syntax error, an infinite loop (via timeout)
        and a failed assertion are all ordinary "did not pass" outcomes.
    """
    # Pick the isolation once per call: 'docker' (no network) or 'process' (fallback).
    backend = resolve_sandbox(sandbox)
    with tempfile.TemporaryDirectory(prefix="clasp-p5-exec-") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(source, encoding="utf-8")
        if backend == "docker":
            # The container runs as user 'nobody', so the mounted program must be world-readable.
            script_path.chmod(0o644)
            Path(tmpdir).chmod(0o755)
            return _run_docker(tmpdir, timeout_seconds)
        return _run_process(script_path, tmpdir, timeout_seconds)


def _run_process(script_path: Path, tmpdir: str, timeout_seconds: float) -> ExecutionOutcome:
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
    return ExecutionOutcome(
        passed=completed.returncode == 0,
        timed_out=False,
        exit_code=completed.returncode,
        duration_seconds=round(time.perf_counter() - start, 6),
        stderr_tail=_truncate(completed.stderr),
    )


def _run_docker(tmpdir: str, timeout_seconds: float) -> ExecutionOutcome:
    name = f"clasp-p5-exec-{uuid.uuid4().hex[:12]}"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            docker_command(tmpdir, timeout_seconds, name=name),
            capture_output=True,
            text=True,
            timeout=timeout_seconds + _DOCKER_GRACE_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # The docker client itself hung: force-remove the container so nothing keeps running.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
        return ExecutionOutcome(
            passed=False,
            timed_out=True,
            exit_code=None,
            duration_seconds=round(time.perf_counter() - start, 6),
            stderr_tail=f"docker client exceeded {timeout_seconds + _DOCKER_GRACE_SECONDS}s; container removed",
            isolation="docker",
        )
    except OSError as exc:
        return ExecutionOutcome(
            passed=False,
            timed_out=False,
            exit_code=None,
            duration_seconds=round(time.perf_counter() - start, 6),
            stderr_tail=_truncate(f"{type(exc).__name__}: {exc}"),
            isolation="docker",
        )
    duration = round(time.perf_counter() - start, 6)
    # 137 = SIGKILL: `timeout -s KILL` at the wall-clock limit, or the memory
    # limit's OOM killer. Elapsed time tells them apart.
    killed = completed.returncode == 137
    timed_out = killed and duration >= timeout_seconds
    note = ""
    if killed:
        note = f"execution exceeded {timeout_seconds}s" if timed_out else "killed (memory limit)"
    return ExecutionOutcome(
        passed=completed.returncode == 0,
        timed_out=timed_out,
        exit_code=completed.returncode,
        duration_seconds=duration,
        stderr_tail=_truncate(completed.stderr or note),
        isolation="docker",
    )


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CAPTURED_OUTPUT:
        return text
    return text[-_MAX_CAPTURED_OUTPUT:]
