"""HumanEval guard anchors — scoring the edge lane's samples so D5's guard can run.

The D5 rule (``registry.promotion.decide``) is two-sided: it also needs
HumanEval pass@1 for the candidate and the baseline adapter. The edge lane
(``edge.humaneval_baseline``) already *generates* those completions on the
GPU and writes them as evalplus samples — one ``{"task_id", "solution"}`` line
per problem, ``solution`` being the prompt plus the truncated completion. What
it could not do was *score* them: evalplus imports the Unix-only ``resource``
module, so on the Windows demo machine the guard stayed unavailable and every
promotion decision was a provisional ROLLBACK.

This module is the scoring half, on P5's executor:

* each sample is assembled into ``solution + test + check(entry_point)`` —
  the program HumanEval's own harness runs — against the **published**
  HumanEval tests in ``eval_harness/humaneval/tasks.jsonl``;
* each program runs through :func:`eval_harness.execution.execute_program`
  (own subprocess, temp dir, timeout, POSIX rlimits);
* pass@1 is the unbiased estimator from :mod:`eval_harness.scoring`.

The output is an anchor JSON in the shape ``edge.promote.resolve_guard``
reads (``base_pass_at_1`` plus ``subset.n``), with the generation manifest
merged in when supplied, exactly as ``edge.humaneval_baseline`` merges
``{**manifest, **score}``.

What it is not
--------------
* Not evalplus. Only the original HumanEval tests run, so ``plus_pass_at_1``
  is ``null``, and the ``scorer`` block says which harness produced the number.
  Candidate and baseline must therefore be scored by the same tool — this one,
  or evalplus for both — never one of each.
* Not a sandbox for hostile code. See :mod:`eval_harness.execution`: the
  isolation is process-level, adequate for completions from the team's own
  model, not for arbitrary untrusted input.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.eval_harness.execution import execute_program
from evaluation.eval_harness.models import EvalTask
from evaluation.eval_harness.scoring import pass_at_k
from evaluation.utils.errors import EvaluationError
from evaluation.utils.timing import utc_timestamp

#: What ``scorer.name`` says in every anchor this module writes.
SCORER_NAME = "clasp-p5 eval_harness.execution"


@dataclass(frozen=True)
class SampleOutcome:
    """One executed sample."""

    task_id: str
    passed: bool
    timed_out: bool
    entry_point_defined: bool
    stderr_tail: str = ""
    isolation: str = "process"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "entry_point_defined": self.entry_point_defined,
            "stderr_tail": self.stderr_tail,
            "isolation": self.isolation,
        }


def load_samples(path: Path | str) -> list[dict[str, str]]:
    """Read an evalplus samples JSONL (``task_id`` plus ``solution`` or ``completion``)."""
    samples: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_no} is not JSON: {exc}") from exc
            if "task_id" not in record or not ({"solution", "completion"} & record.keys()):
                raise EvaluationError(f"{path}:{line_no} needs task_id and solution (or completion)")
            samples.append(record)
    if not samples:
        raise EvaluationError(f"No samples in {path}")
    return samples


def assemble_program(task: EvalTask, sample: Mapping[str, str]) -> str:
    """``solution + test + check(entry_point)`` — HumanEval's own program shape.

    A ``solution`` already contains the prompt (evalplus convention); a bare
    ``completion`` is appended to the task's prompt first.
    """
    if "solution" in sample:
        source = sample["solution"]
    else:
        prompt = task.prompt if task.prompt.endswith("\n") else task.prompt + "\n"
        source = prompt + sample["completion"]
    if not source.endswith("\n"):
        source += "\n"
    return f"{source}\n{task.test_code}\ncheck({task.entry_point})\n"


def score_samples(
    samples: Sequence[Mapping[str, str]],
    tasks: Mapping[str, EvalTask],
    *,
    timeout_seconds: float = 10.0,
    sandbox: str = "auto",
) -> list[SampleOutcome]:
    """Execute every sample against its task's tests. Unknown task ids are an error."""
    # Every sample must belong to a known HumanEval task, otherwise the anchor would be meaningless.
    unknown = sorted({s["task_id"] for s in samples} - tasks.keys())
    if unknown:
        raise EvaluationError(
            f"{len(unknown)} sample task id(s) are not in the HumanEval task file: {unknown[:5]}"
        )
    outcomes: list[SampleOutcome] = []
    for sample in samples:
        task = tasks[sample["task_id"]]
        # Build prompt + completion + HumanEval's check() call, then run it in the sandbox.
        program = assemble_program(task, sample)
        result = execute_program(program, timeout_seconds=timeout_seconds, sandbox=sandbox)
        outcomes.append(
            SampleOutcome(
                task_id=task.task_id,
                passed=result.passed,
                timed_out=result.timed_out,
                entry_point_defined=f"def {task.entry_point}" in program,
                stderr_tail=result.stderr_tail[-400:],
                isolation=result.isolation,
            )
        )
    return outcomes


_ISOLATION_TEXT = {
    "docker": "docker container per program: no network, read-only fs, no capabilities, mem/cpu/pid limits",
    "process": "subprocess per program, temp dir, wall-clock timeout, POSIX rlimits (no network confinement)",
}


def _isolation_summary(outcomes: Sequence[SampleOutcome]) -> str:
    kinds = sorted({o.isolation for o in outcomes})
    return "; ".join(_ISOLATION_TEXT.get(k, k) for k in kinds)


def anchor_result_record(
    anchor: Mapping[str, Any],
    *,
    model_checkpoint: str,
    generation: Mapping[str, Any],
    provenance_note: str,
    raw_artifact_path: str | None = None,
):
    """A REAL ``results.json`` record for a scored anchor (HumanEval pass@1 on its subset).

    ``run_id`` is derived from the samples' sha256, so re-scoring the same
    samples replaces the record instead of adding a duplicate.
    """
    from evaluation import __version__
    from evaluation.eval_harness.results_store import GenerationSnapshot, ResultRecord
    from evaluation.interfaces.contracts import AdapterKind, AdapterRef, BenchmarkName, EvalResult

    n = int(anchor["n_problems"])
    n_samples = int(anchor["samples"]["n_samples"])
    return ResultRecord(
        eval_result=EvalResult(
            adapter=AdapterRef(name="baseline", version=0, kind=AdapterKind.CLIENT),
            benchmark=BenchmarkName.HUMANEVAL,
            pass_at_k={1: float(anchor["base_pass_at_1"])},
            num_tasks=n,
            num_samples_per_task=max(1, n_samples // n),
            created_at=anchor["scored_at"],
            # Same samples -> same run_id, so re-scoring replaces the record instead of duplicating it.
            run_id=f"anchor-{anchor['samples']['sha256'][:12]}",
        ),
        model_checkpoint=model_checkpoint,
        dataset_split=f"frozen subset: first {n} HumanEval tasks by id",
        num_problems=n,
        seed=generation.get("seed"),
        generation=GenerationSnapshot(
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            stop_sequences=list(generation["stop_sequences"]),
        ),
        provenance="REAL",
        provenance_note=provenance_note,
        raw_artifact_path=raw_artifact_path,
        benchmark_data_source="original HumanEval check() tests (src/evaluation/eval_harness/humaneval/tasks.jsonl)",
        harness_version=__version__,
    )


def _task_order(task_id: str) -> tuple[int, str]:
    tail = task_id.rsplit("/", 1)[-1]
    return (int(tail), task_id) if tail.isdigit() else (1 << 30, task_id)


def build_anchor(
    outcomes: Sequence[SampleOutcome],
    *,
    samples_path: Path | str,
    generation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Anchor JSON for ``edge.promote.resolve_guard``.

    pass@1 is averaged per task with the unbiased estimator, so several
    samples per task are handled correctly; with one greedy sample per task it
    is simply the fraction solved. When the generation manifest lists the
    frozen subset, the samples must cover exactly that subset.
    """
    if not outcomes:
        raise EvaluationError("No outcomes to build an anchor from")
    by_task: dict[str, list[SampleOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_task[outcome.task_id].append(outcome)
    task_ids = sorted(by_task, key=_task_order)

    manifest = dict(generation_manifest or {})
    subset = manifest.get("subset") or {"n": len(task_ids), "task_ids": task_ids}
    declared = subset.get("task_ids")
    if declared is not None and sorted(declared, key=_task_order) != task_ids:
        missing = sorted(set(declared) - set(task_ids), key=_task_order)
        extra = sorted(set(task_ids) - set(declared), key=_task_order)
        raise EvaluationError(
            f"Samples do not match the manifest's frozen subset (missing {missing[:5]}, extra {extra[:5]})"
        )

    # pass@1 per task with the unbiased estimator, then averaged over tasks.
    per_task_p1 = [pass_at_k(len(v), sum(o.passed for o in v), 1) for v in (by_task[t] for t in task_ids)]
    solved = sum(1 for t in task_ids if any(o.passed for o in by_task[t]))
    samples_bytes = Path(samples_path).read_bytes()

    return {
        **manifest,
        "subset": subset,
        "n_problems": len(task_ids),
        "base_pass_at_1": round(sum(per_task_p1) / len(task_ids), 4),
        "base_solved": solved,
        "plus_pass_at_1": None,
        "plus_solved": None,
        "scorer": {
            "name": SCORER_NAME,
            "tests": "original HumanEval check() tests (src/evaluation/eval_harness/humaneval/tasks.jsonl)",
            "isolation": _isolation_summary(outcomes),
            "plus_note": "EvalPlus extended tests not run — plus_pass_at_1 is null, not zero",
            "comparability": "compare only with an anchor scored by the same scorer",
        },
        "samples": {
            "path": str(samples_path),
            "sha256": hashlib.sha256(samples_bytes).hexdigest(),
            "n_samples": len(outcomes),
        },
        "generation_manifest_supplied": generation_manifest is not None,
        "scored_at": utc_timestamp(),
        "per_task": [o.to_dict() for o in sorted(outcomes, key=lambda o: _task_order(o.task_id))],
    }
