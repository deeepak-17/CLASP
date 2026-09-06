"""Evaluation harness — Week 1 scaffold, Week 3 scoring.

Week 1 Wed: "Scaffold eval-harness repo structure"
Week 1 Fri: "Dry-run harness on a tiny sample; fix path/config issues"
Week 3 Mon/Tue: "Wire the HumanEval/MBPP runner to call the merged model"
Week 3 Wed: "Implement Pass@k scoring function"

What this does
--------------
For every enabled benchmark: load tasks, build prompts, request completions
from the configured inference client (mock, or P1's real Edge Layer once
registered — see :mod:`evaluation.registry`), truncate at stop sequences,
assemble the runnable program, and write a run artefact plus a Markdown
report.

When ``scoring.execution_enabled`` is true (see :mod:`evaluation.execution`
for the isolation model and its limits), every assembled program is executed
and Pass@k is computed via :mod:`evaluation.scoring` for each ``k`` in
``scoring.pass_at_k``. ``TaskOutcome.passed`` is a tri-state by design: it
stays ``None`` when a task never ran generation (an error) or when
``execution_enabled`` is false, is populated with one bool per completion
once scored — "not scored" can never be misread as "failed".

``execution_enabled`` defaults to ``false``. A dry run (the Week-1 default)
still proves config resolution, path resolution, task parsing, prompt
construction, the client protocol, stop-sequence truncation, program
assembly and artefact writing all work, deterministically, with no sandbox
and no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from evaluation.base import BenchmarkAdapter
from evaluation.execution import execute_program
from evaluation.models import (
    BenchmarkRunSummary,
    EvalTask,
    EvaluationConfig,
    TaskOutcome,
)
from evaluation.registry import build_adapters, build_inference_client
from evaluation.scoring import AggregatePassAtK, aggregate_pass_at_k, pass_at_k
from interfaces.edge_client import EdgeInferenceClient, GenerationRequest
from interfaces.security_client import NullPrivacyAccountant, PrivacyAccountant
from utils.config import load_config
from utils.errors import ClaspP5Error, EvaluationError
from utils.io_utils import ensure_dir, write_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.timing import Stopwatch, file_timestamp, utc_timestamp

_LOG = get_logger(__name__)


def load_evaluation_config(path: Path | str | None = None) -> EvaluationConfig:
    """Load ``configs/evaluation.yaml`` into an :class:`EvaluationConfig`."""
    target = path or (project_paths().configs / "evaluation.yaml")
    return load_config(EvaluationConfig, target, section="evaluation")


@dataclass
class HarnessRun:
    """Everything one harness invocation produced."""

    run_id: str
    mode: str
    model_id: str
    created_at: str
    scored: bool
    summaries: list[BenchmarkRunSummary] = field(default_factory=list)
    outcomes: dict[str, list[TaskOutcome]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    artifact_path: Path | None = None
    privacy_label: str = "DP off (baseline)"

    @property
    def ok(self) -> bool:
        """Whether every attempted task generated without error."""
        return all(summary.tasks_failed == 0 for summary in self.summaries)

    @property
    def total_tasks(self) -> int:
        return sum(summary.tasks_attempted for summary in self.summaries)

    @property
    def total_completions(self) -> int:
        return sum(summary.completions_generated for summary in self.summaries)

    def to_dict(self, *, include_completions: bool = True) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "scored": self.scored,
            "ok": self.ok,
            "elapsed_seconds": self.elapsed_seconds,
            "privacy": self.privacy_label,
            "totals": {
                "tasks": self.total_tasks,
                "completions": self.total_completions,
            },
            "benchmarks": [summary.to_dict() for summary in self.summaries],
            "outcomes": (
                {
                    benchmark: [outcome.to_dict() for outcome in outcomes]
                    for benchmark, outcomes in self.outcomes.items()
                }
                if include_completions
                else {}
            ),
            "notes": (
                [
                    "scoring.execution_enabled is true: 'passed' and 'pass_at_k' below are "
                    "computed by executing every assembled program (evaluation.execution) and "
                    "scoring with evaluation.scoring.pass_at_k.",
                    "This artefact still is NOT an EvalResult contract document — it is the raw "
                    "per-run record. See evaluation/results/results.json for the dashboard-facing "
                    "EvalResult rollup (evaluation.results_store).",
                ]
                if self.scored
                else [
                    "scoring.execution_enabled is false for this run; 'scored' is false and "
                    "'passed' is null for every task. Pass@k scoring exists in "
                    "evaluation.scoring/evaluation.execution but was not invoked.",
                    "This artefact is NOT an EvalResult contract document. P5 emits "
                    "EvalResult only once scoring is real.",
                ]
            ),
        }


class EvaluationHarness:
    """Drives benchmark adapters against an inference client.

    Args:
        config: Parsed ``evaluation`` config section.
        client: Inference client. Defaults to whatever ``backend.kind``
            selects; injected directly by tests.
        accountant: P3 privacy accountant used to label the run. Defaults to
            :class:`~interfaces.security_client.NullPrivacyAccountant`.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        client: EdgeInferenceClient | None = None,
        accountant: PrivacyAccountant | None = None,
    ) -> None:
        self._config = config
        self._client = client or build_inference_client(config)
        self._accountant = accountant or NullPrivacyAccountant()
        self._adapters: list[BenchmarkAdapter] = build_adapters(config)

    # --- public API --------------------------------------------------------
    def run(self, *, limit: int | None = None, write_artifact: bool = True) -> HarnessRun:
        """Execute the harness across every enabled benchmark.

        Args:
            limit: Per-benchmark task cap, overriding ``run.limit``.
            write_artifact: Write the run JSON to ``evaluation/results/``.
        """
        effective_limit = limit if limit is not None else self._config.run.limit
        run_id = f"{self._config.run.run_id_prefix}-{file_timestamp()}"

        run = HarnessRun(
            run_id=run_id,
            mode=self._config.run.mode,
            model_id=self._client.model_id,
            created_at=utc_timestamp(),
            scored=self._config.scoring.execution_enabled,
            privacy_label=self._accountant.current_budget().label(),
        )

        _LOG.info(
            "Harness run %s | mode=%s | backend=%s | limit=%s | samples/task=%d",
            run_id,
            run.mode,
            self._config.backend.kind,
            effective_limit if effective_limit is not None else "all",
            self._config.run.num_samples_per_task,
        )

        with Stopwatch("harness") as watch:
            for adapter in self._adapters:
                summary, outcomes = self._run_benchmark(adapter, effective_limit)
                run.summaries.append(summary)
                run.outcomes[adapter.name.value] = outcomes
        run.elapsed_seconds = watch.elapsed_seconds

        if write_artifact:
            run.artifact_path = self._write_artifact(run)

        _LOG.info(
            "Harness run %s finished in %.3fs: %d task(s), %d completion(s), ok=%s",
            run_id,
            run.elapsed_seconds,
            run.total_tasks,
            run.total_completions,
            run.ok,
        )
        return run

    # --- internals ---------------------------------------------------------
    def _run_benchmark(
        self, adapter: BenchmarkAdapter, limit: int | None
    ) -> tuple[BenchmarkRunSummary, list[TaskOutcome]]:
        """Run every task of one benchmark."""
        _LOG.info("--- %s ---", adapter.name.value)
        tasks = adapter.load_tasks(limit=limit)

        outcomes: list[TaskOutcome] = []
        errors: list[str] = []

        with Stopwatch(adapter.name.value) as watch:
            for task in tasks:
                outcome = self._run_task(adapter, task)
                outcomes.append(outcome)
                if outcome.error:
                    errors.append(f"{task.task_id}: {outcome.error}")
                    if not self._config.run.continue_on_task_error:
                        raise EvaluationError(
                            f"Task {task.task_id} failed and run.continue_on_task_error is false: "
                            f"{outcome.error}"
                        )

        succeeded = sum(1 for outcome in outcomes if outcome.ok)
        scoring = self._config.scoring
        aggregate: AggregatePassAtK | None = None
        if scoring.execution_enabled:
            aggregate = aggregate_pass_at_k(outcomes, scoring.pass_at_k)
            for k, skipped in aggregate.tasks_skipped.items():
                if skipped:
                    _LOG.warning(
                        "%s: pass@%d skipped %d/%d task(s): %s",
                        adapter.name.value,
                        k,
                        len(skipped),
                        len(outcomes),
                        "; ".join(skipped[:3]) + (" ..." if len(skipped) > 3 else ""),
                    )

        summary = BenchmarkRunSummary(
            benchmark=adapter.name,
            tasks_total=len(tasks),
            tasks_attempted=len(outcomes),
            tasks_succeeded=succeeded,
            tasks_failed=len(outcomes) - succeeded,
            completions_generated=sum(len(outcome.completions) for outcome in outcomes),
            scored=aggregate is not None,
            pass_at_k=dict(aggregate.pass_at_k) if aggregate else {},
            elapsed_seconds=watch.elapsed_seconds,
            errors=errors,
        )
        _LOG.info(
            "%s: %d/%d task(s) generated, %d completion(s), %.3fs%s",
            adapter.name.value,
            summary.tasks_succeeded,
            summary.tasks_attempted,
            summary.completions_generated,
            summary.elapsed_seconds,
            f", pass@k={summary.pass_at_k}" if summary.scored else "",
        )
        return summary, outcomes

    def _run_task(self, adapter: BenchmarkAdapter, task: EvalTask) -> TaskOutcome:
        """Generate, truncate, assemble and (if enabled) execute one task.

        Never raises for task-local faults — a broken client or a candidate
        program that crashes must not abort the whole benchmark run.
        """
        generation = self._config.generation
        scoring = self._config.scoring
        try:
            request = GenerationRequest(
                task_id=task.task_id,
                prompt=adapter.build_prompt(task),
                max_new_tokens=generation.max_new_tokens,
                temperature=generation.temperature,
                stop_sequences=tuple(generation.stop_sequences),
                num_samples=self._config.run.num_samples_per_task,
            )
            result = self._client.generate(request)

            completions = [
                adapter.truncate_completion(text, generation.stop_sequences)
                for text in result.completions
            ]

            # Assemble every program now. It is cheap, and it surfaces
            # adapter/format mismatches during the dry run rather than at
            # execution time when a sandbox failure would be ambiguous.
            programs = [adapter.assemble_program(task, completion) for completion in completions]

            passed: list[bool] | None = None
            if scoring.execution_enabled:
                passed = [
                    execute_program(program, timeout_seconds=scoring.execution_timeout_seconds).passed
                    for program in programs
                ]

            return TaskOutcome(
                task_id=task.task_id,
                benchmark=adapter.name,
                completions=completions,
                passed=passed,
                latency_ms=result.latency_ms,
            )

        except ClaspP5Error as exc:
            _LOG.warning("Task %s failed: %s", task.task_id, exc)
            return TaskOutcome(
                task_id=task.task_id,
                benchmark=adapter.name,
                completions=[],
                error=str(exc),
            )
        except Exception as exc:  # a broken client must not abort the whole run
            _LOG.exception("Task %s raised an unexpected error", task.task_id)
            return TaskOutcome(
                task_id=task.task_id,
                benchmark=adapter.name,
                completions=[],
                error=f"{type(exc).__name__}: {exc}",
            )

    def _write_artifact(self, run: HarnessRun) -> Path:
        """Write the run JSON under ``run.results_dir``."""
        results_dir = ensure_dir(self._config.run.results_dir)
        path = results_dir / f"{run.run_id}.json"
        write_json(path, run.to_dict())
        _LOG.info("Wrote run artefact -> %s", project_paths().relative(path))
        return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_dry_run_report(run: HarnessRun, config: EvaluationConfig) -> MarkdownReport:
    """Render the Week-1 Friday dry-run outcome as Markdown."""
    report = MarkdownReport(
        title="CLASP-P5 · Evaluation Harness Dry Run",
        subtitle="Week 1 · Friday deliverable — tiny-sample dry run and path/config verification",
    )

    report.heading("1. Verdict")
    report.status_line(
        run.ok,
        f"{run.total_tasks} task(s) across {len(run.summaries)} benchmark(s) produced "
        f"{run.total_completions} completion(s) with no generation errors"
        if run.ok
        else "one or more tasks failed to generate",
    )
    report.key_values(
        {
            "Run id": f"`{run.run_id}`",
            "Mode": run.mode,
            "Backend": f"{config.backend.kind} (`{run.model_id}`)",
            "Samples per task": config.run.num_samples_per_task,
            "Task limit": config.run.limit if config.run.limit is not None else "all",
            "Elapsed": f"{run.elapsed_seconds:.3f}s",
            "Privacy": run.privacy_label,
            "Artefact": f"`{project_paths().relative(run.artifact_path)}`" if run.artifact_path else "—",
        }
    )

    report.heading("2. Per-benchmark results")
    report.table(
        ["Benchmark", "Tasks", "Generated", "Failed", "Completions", "Scored", "Elapsed (s)"],
        [
            [
                summary.benchmark.value,
                summary.tasks_total,
                summary.tasks_succeeded,
                summary.tasks_failed,
                summary.completions_generated,
                summary.scored,
                f"{summary.elapsed_seconds:.3f}",
            ]
            for summary in run.summaries
        ],
    )

    report.heading("3. What this dry run verifies")
    report.bullets(
        [
            "`configs/evaluation.yaml` parses and every declared path resolves.",
            "Both benchmark task files load and parse into the normalised `EvalTask` type.",
            "Prompt construction succeeds for every task in both adapters.",
            "The `EdgeInferenceClient` protocol round-trips request → completion.",
            "Stop-sequence truncation runs over every completion.",
            "`assemble_program` produces a program for every task/completion pair.",
            "The run artefact is written to `evaluation/results/` in the expected shape.",
        ]
    )

    report.heading("4. Explicitly out of scope this week")
    report.bullets(
        [
            "**Pass@k scoring** — Week 3 (`Implement Pass@k scoring function`). "
            "`scored` is false and every `passed` field is null.",
            "**Real model inference** — Week 3 (`Wire HumanEval runner to call the merged model`). "
            "Completions come from `MockEdgeInferenceClient` and are placeholders.",
            "**Sandboxed execution** — generated code is assembled but never executed; "
            "the isolation policy needs P3 sign-off first.",
            "**Published benchmark data** — the bundled `sample_tasks.jsonl` fixtures are "
            "original, format-compatible tasks, not HumanEval/MBPP proper.",
        ]
    )

    errors = [error for summary in run.summaries for error in summary.errors]
    if errors:
        report.heading("5. Errors")
        report.bullets(errors)

    report.rule()
    report.paragraph(
        "Generated by `evaluation/harness.py`. Regenerate with "
        "`python scripts/dry_run_harness.py`."
    )
    return report


def default_dry_run_report_path() -> Path:
    """Canonical location of the generated dry-run report."""
    return project_paths().reports / "dry_run_report.md"


# ---------------------------------------------------------------------------
# WEEK 3 — Pass@k scoring for one task, across every configured k
# ---------------------------------------------------------------------------
def compute_pass_at_k(n: int, c: int, ks: Sequence[int]) -> dict[int, float]:
    """Pass@k for one task at every k in ``ks``. **Week-3 deliverable — implemented.**

    Thin wrapper around :func:`evaluation.scoring.pass_at_k` kept here because
    this is the call site and config key (``scoring.pass_at_k``) the Week-1/2
    scaffold fixed in advance. Any ``k`` that exceeds ``n`` (more samples
    requested than were generated) is silently omitted from the result
    rather than raising — the harness-level aggregate
    (:func:`evaluation.scoring.aggregate_pass_at_k`, used by
    :meth:`EvaluationHarness._run_benchmark`) is what reports *that*
    omission per task; this single-task convenience function is for callers
    (tests, notebooks, the sanity-check script) that just want the numbers
    that could be computed.

    Args:
        n: Samples generated for the task.
        c: Of those, how many passed.
        ks: k values to compute, e.g. ``config.scoring.pass_at_k``.

    Returns:
        ``{k: pass@k}`` for every ``k`` that was computable.
    """
    computed: dict[int, float] = {}
    for k in ks:
        try:
            computed[k] = pass_at_k(n, c, k)
        except EvaluationError:
            continue
    return computed
