"""Evaluation domain model and configuration schema.

Task representation
-------------------
:class:`EvalTask` is benchmark-agnostic on purpose. HumanEval and MBPP differ
in prompt style and test format, but the harness only needs "a prompt to
complete" plus "a test program to run", so both adapters normalise into this
one type and the runner stays free of per-benchmark branching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from interfaces.contracts import BenchmarkName
from utils.errors import ConfigError, EvaluationError


@dataclass(frozen=True)
class EvalTask:
    """One benchmark problem, normalised across HumanEval and MBPP.

    Attributes:
        task_id: Benchmark-unique identifier (``HumanEval/0``, ``MBPP/2``).
        benchmark: Which benchmark it came from.
        prompt: Text handed to the model. For HumanEval, the function signature
            and docstring; for MBPP, the natural-language description plus, if
            configured, the first assert as a specification example.
        test_code: Test program appended after the completion during scoring.
        entry_point: Function the tests call.
        canonical_solution: Reference solution when the benchmark supplies one.
            Used only for harness self-tests, never shown to the model.
        metadata: Benchmark-specific extras retained for traceability.
    """

    task_id: str
    benchmark: BenchmarkName
    prompt: str
    test_code: str
    entry_point: str
    canonical_solution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise EvaluationError("EvalTask.task_id must be non-empty")
        if not self.prompt.strip():
            raise EvaluationError(f"Task {self.task_id} has an empty prompt")
        if not self.entry_point:
            raise EvaluationError(f"Task {self.task_id} has no entry_point")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["benchmark"] = self.benchmark.value
        return payload


@dataclass(frozen=True)
class TaskOutcome:
    """Result of running one task through the harness.

    Week 1/2 populates generation fields only. ``passed`` stays ``None``
    until Week 3 wires execution-based scoring — deliberately tri-state, so
    "not scored" can never be misread as "failed".
    """

    task_id: str
    benchmark: BenchmarkName
    completions: list[str]
    passed: list[bool] | None = None
    error: str | None = None
    latency_ms: float | None = None

    @property
    def ok(self) -> bool:
        """Whether generation succeeded (independent of correctness)."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark.value,
            "num_completions": len(self.completions),
            "completions": self.completions,
            "passed": self.passed,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class BenchmarkRunSummary:
    """Aggregate outcome for one benchmark within a run."""

    benchmark: BenchmarkName
    tasks_total: int
    tasks_attempted: int
    tasks_succeeded: int
    tasks_failed: int
    completions_generated: int
    scored: bool
    pass_at_k: dict[int, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark.value,
            "tasks_total": self.tasks_total,
            "tasks_attempted": self.tasks_attempted,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "completions_generated": self.completions_generated,
            "scored": self.scored,
            "pass_at_k": {str(k): v for k, v in sorted(self.pass_at_k.items())},
            "elapsed_seconds": self.elapsed_seconds,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Configuration schema (mirrors configs/evaluation.yaml)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """Top-level run behaviour."""

    mode: str = "dry_run"
    run_id_prefix: str = "p5"
    limit: int | None = 5
    num_samples_per_task: int = 1
    continue_on_task_error: bool = True
    results_dir: Path = Path("evaluation/results")

    VALID_MODES = ("dry_run", "full")

    def __post_init__(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise ConfigError(f"run.mode must be one of {self.VALID_MODES}, got '{self.mode}'")
        if self.limit is not None and self.limit < 1:
            raise ConfigError("run.limit must be >= 1 or null")
        if self.num_samples_per_task < 1:
            raise ConfigError("run.num_samples_per_task must be >= 1")


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters passed through to the Edge Layer."""

    max_new_tokens: int = 384
    temperature: float = 0.2
    stop_sequences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ConfigError("generation.max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ConfigError("generation.temperature must be >= 0")


@dataclass(frozen=True)
class BackendConfig:
    """Which inference backend the harness drives."""

    kind: str = "mock"
    model_id: str = "mock/deepseek-coder-6.7b-base"

    VALID_KINDS = ("mock", "edge")

    def __post_init__(self) -> None:
        if self.kind not in self.VALID_KINDS:
            raise ConfigError(f"backend.kind must be one of {self.VALID_KINDS}, got '{self.kind}'")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings shared by every benchmark adapter."""

    enabled: bool = True
    tasks_path: Path | None = None
    sample_tasks_path: Path | None = None
    include_test_example_in_prompt: bool = True


@dataclass(frozen=True)
class ScoringConfig:
    """Pass@k scoring. Validated in Week 1; acted on in Week 3."""

    pass_at_k: list[int] = field(default_factory=lambda: [1])
    execution_enabled: bool = False
    execution_timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not self.pass_at_k:
            raise ConfigError("scoring.pass_at_k must list at least one k")
        for k in self.pass_at_k:
            if k < 1:
                raise ConfigError(f"scoring.pass_at_k values must be >= 1, got {k}")
        if self.execution_timeout_seconds <= 0:
            raise ConfigError("scoring.execution_timeout_seconds must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    """Root of ``configs/evaluation.yaml`` under the ``evaluation`` key."""

    run: RunConfig = field(default_factory=RunConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    benchmarks: dict[str, BenchmarkConfig] = field(default_factory=dict)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def __post_init__(self) -> None:
        known = {b.value.lower() for b in BenchmarkName}
        unknown = set(self.benchmarks) - known
        if unknown:
            raise ConfigError(
                f"Unknown benchmark(s) in configs/evaluation.yaml: {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}"
            )
        if self.run.num_samples_per_task < max(self.scoring.pass_at_k, default=1):
            # Not fatal: legitimate during a Week-1 dry run, but pass@10 from
            # one sample per task is not a number anyone should quote.
            pass

    def benchmark(self, name: BenchmarkName) -> BenchmarkConfig:
        """Config for ``name``, defaulting to an enabled benchmark with no paths."""
        return self.benchmarks.get(name.value.lower(), BenchmarkConfig())

    def enabled_benchmarks(self) -> list[BenchmarkName]:
        """Benchmarks switched on in config, in declaration order."""
        return [name for name in BenchmarkName if self.benchmark(name).enabled]


def parse_benchmark_name(value: str) -> BenchmarkName:
    """Parse a benchmark name case-insensitively."""
    for name in BenchmarkName:
        if name.value.lower() == value.lower():
            return name
    valid = ", ".join(b.value for b in BenchmarkName)
    raise EvaluationError(f"Unknown benchmark '{value}'. Valid: {valid}")


def task_from_dict(data: Mapping[str, Any]) -> EvalTask:
    """Rebuild an :class:`EvalTask` from its serialised form."""
    try:
        return EvalTask(
            task_id=str(data["task_id"]),
            benchmark=parse_benchmark_name(str(data["benchmark"])),
            prompt=str(data["prompt"]),
            test_code=str(data.get("test_code", "")),
            entry_point=str(data["entry_point"]),
            canonical_solution=data.get("canonical_solution"),
            metadata=dict(data.get("metadata", {})),
        )
    except KeyError as exc:
        raise EvaluationError(f"Task record missing field: {exc}") from exc
