"""Sandboxing of model-generated code — docker isolation flags, backend resolution, provenance."""

from __future__ import annotations

import pytest

from evaluation.eval_harness import execution
from evaluation.eval_harness.execution import docker_command, execute_program, resolve_sandbox
from evaluation.eval_harness.guard_anchor import SampleOutcome, _isolation_summary
from evaluation.eval_harness.models import ScoringConfig
from evaluation.utils.errors import ConfigError, EvaluationError


def test_docker_command_isolates_network_filesystem_and_resources(tmp_path) -> None:
    cmd = docker_command(tmp_path, 7, name="clasp-p5-exec-test")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--pids-limit 64",
                 "--memory 512m", "--memory-swap 512m", "--cpus 1", "--user 65534:65534", "--rm"):
        assert flag in joined, flag
    assert f"{tmp_path.resolve()}:/sandbox:ro" in cmd
    assert cmd[-6:] == ["timeout", "-s", "KILL", "7", "python", "/sandbox/candidate.py"]


@pytest.mark.parametrize(("mode", "available", "expected"), [
    ("process", True, "process"),
    ("auto", True, "docker"),
    ("auto", False, "process"),
    ("docker", True, "docker"),
])
def test_resolve_sandbox(monkeypatch, mode: str, available: bool, expected: str) -> None:
    monkeypatch.setattr(execution, "docker_available", lambda: available)
    assert resolve_sandbox(mode) == expected


def test_docker_mode_refuses_to_fall_back(monkeypatch) -> None:
    monkeypatch.setattr(execution, "docker_available", lambda: False)
    with pytest.raises(EvaluationError, match="no Docker daemon"):
        resolve_sandbox("docker")
    with pytest.raises(EvaluationError):
        resolve_sandbox("chroot")


def test_process_backend_records_its_isolation() -> None:
    outcome = execute_program("assert 1 + 1 == 2\n", sandbox="process")
    assert outcome.passed and outcome.isolation == "process"
    assert outcome.to_dict()["isolation"] == "process"


def test_scoring_config_validates_sandbox() -> None:
    assert ScoringConfig().sandbox == "auto"
    with pytest.raises(ConfigError):
        ScoringConfig(sandbox="none")


def test_anchor_states_the_isolation_actually_used() -> None:
    outcomes = [SampleOutcome("t1", True, False, True, isolation="docker"),
                SampleOutcome("t2", False, False, True, isolation="docker")]
    assert _isolation_summary(outcomes).startswith("docker container per program: no network")
    mixed = outcomes + [SampleOutcome("t3", True, False, True, isolation="process")]
    assert "no network confinement" in _isolation_summary(mixed)


# --- what counts as a pass (review on #22: sys.exit(0) must not pass) ---
_HUMANEVAL_SHAPED = """
def f(x):
{body}

def check(candidate):
    assert candidate(2) == 4

check(f)
"""


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("    return x * 2", True),                                   # correct
        ("    return None", False),                                   # wrong answer
        ("    import sys; sys.exit(0)", False),                       # exits 0 before the tests finish
        ("    import os; os._exit(0)", False),                        # hard exit, same idea
        ("    print('__CLASP_PASSED_x__'); import sys; sys.exit(0)", False),  # guessed sentinel
    ],
)
def test_pass_requires_reaching_the_end_of_the_tests(body: str, expected: bool) -> None:
    outcome = execute_program(_HUMANEVAL_SHAPED.format(body=body), sandbox="process")
    assert outcome.passed is expected


def test_spawned_child_does_not_outlive_the_timeout(tmp_path) -> None:
    marker = tmp_path / "alive"
    program = (
        "import subprocess, time\n"
        f"subprocess.Popen(['/bin/sh', '-c', 'sleep 2; echo alive > {marker}'])\n"
        "time.sleep(30)\n"
    )
    outcome = execute_program(program, timeout_seconds=1, sandbox="process")
    assert outcome.timed_out
    import time

    time.sleep(2.5)
    assert not marker.exists()
