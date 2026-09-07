"""Benchmark adapter base class.

An adapter's job is narrow: turn a benchmark's on-disk format into
:class:`~evaluation.models.EvalTask` objects, build the prompt, and know how
to stitch a completion back into a runnable program. Everything else —
generation, retries, artefact writing — belongs to
:class:`~evaluation.harness.EvaluationHarness`.

That split is what lets the Week-3 "wire the runner to the merged model" task
touch one file instead of two benchmark implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.models import BenchmarkConfig, EvalTask, task_from_dict
from interfaces.contracts import BenchmarkName
from utils.errors import EvaluationError
from utils.io_utils import read_jsonl
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)


class BenchmarkAdapter(ABC):
    """Loads and prepares tasks for one code-generation benchmark."""

    #: Benchmark this adapter serves.
    name: BenchmarkName

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    # --- required of subclasses -------------------------------------------
    @abstractmethod
    def build_prompt(self, task: EvalTask) -> str:
        """Return the text handed to the model for ``task``."""

    @abstractmethod
    def assemble_program(self, task: EvalTask, completion: str) -> str:
        """Stitch ``completion`` into a runnable program with the task's tests.

        Not executed in Weeks 1-2 — Pass@k scoring is a Week-3 deliverable —
        but implemented now so the Week-1 dry run can verify that every task
        assembles into syntactically plausible source before any sandbox work
        begins.
        """

    # --- provided ----------------------------------------------------------
    def load_tasks(self, limit: int | None = None) -> list[EvalTask]:
        """Load tasks from the configured path, honouring ``limit``.

        Falls back to the bundled sample fixture when the configured path is
        missing, and logs the substitution loudly: a dry run that silently
        evaluated fixtures instead of the real benchmark would be worse than
        one that failed.
        """
        path = self._resolve_tasks_path()
        tasks = [self._parse(record, path) for record in read_jsonl(path)]

        if not tasks:
            raise EvaluationError(f"No tasks loaded for {self.name.value} from {path}")

        if limit is not None:
            tasks = tasks[:limit]

        _LOG.info(
            "%s: loaded %d task(s) from %s",
            self.name.value,
            len(tasks),
            project_paths().relative(path),
        )
        return tasks

    def prepare(self, tasks: Iterable[EvalTask]) -> list[tuple[EvalTask, str]]:
        """Pair each task with its built prompt."""
        return [(task, self.build_prompt(task)) for task in tasks]

    @staticmethod
    def truncate_completion(completion: str, stop_sequences: Sequence[str]) -> str:
        """Cut ``completion`` at the earliest stop sequence.

        Models frequently continue past the function body into the next
        definition or a ``__main__`` block; leaving that in place would append
        junk after the tests and produce spurious failures at scoring time.
        """
        earliest = len(completion)
        for marker in stop_sequences:
            index = completion.find(marker)
            if index != -1:
                earliest = min(earliest, index)
        return completion[:earliest]

    # --- internals ---------------------------------------------------------
    def _resolve_tasks_path(self) -> Path:
        configured = self._config.tasks_path
        if configured and Path(configured).is_file():
            return Path(configured)

        fallback = self._config.sample_tasks_path or self.default_sample_path()
        if configured:
            _LOG.warning(
                "%s: configured tasks_path %s not found; falling back to the bundled sample "
                "fixture. Results are NOT comparable to published benchmark numbers.",
                self.name.value,
                configured,
            )
        if not Path(fallback).is_file():
            raise EvaluationError(
                f"{self.name.value}: neither tasks_path ({configured}) nor the sample fixture "
                f"({fallback}) exists. Check configs/evaluation.yaml."
            )
        return Path(fallback)

    def default_sample_path(self) -> Path:
        """Bundled fixture location for this benchmark."""
        return project_paths().evaluation / self.name.value.lower() / "sample_tasks.jsonl"

    def _parse(self, record: dict, path: Path) -> EvalTask:
        task = task_from_dict(record)
        if task.benchmark is not self.name:
            raise EvaluationError(
                f"{path}: task '{task.task_id}' declares benchmark "
                f"'{task.benchmark.value}' but was loaded by the {self.name.value} adapter"
            )
        return task
