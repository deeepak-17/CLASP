"""Pass@k scoring — Week 3, Wednesday.

"Implement Pass@k scoring function."

The unbiased estimator
-----------------------
Chen et al., 2021 ("Evaluating Large Language Models Trained on Code",
https://arxiv.org/abs/2107.03374, eq. 1) show that the naive Monte-Carlo
estimate of "would at least one of k samples pass" is a biased estimator of
pass@k when it is computed from a *fixed* pool of ``n`` generated samples
(``n`` is usually > ``k`` so several k values can be read off one generation
run). Generating exactly ``k`` samples and checking whether any pass has
high variance; the fix is to generate ``n >= k`` samples, count how many of
them pass (``c``), and use::

    pass@k := E[1 - C(n-c, k) / C(n, k)]

where ``C(a, b)`` is "a choose b". Variables, matching the paper and this
module's signature:

``n``
    Total number of samples generated for one task.
``c``
    Number of those samples that passed (compiled, ran, and satisfied every
    test in the task).
``k``
    The k in "pass@k" — the number of submission attempts a user is modelled
    as getting.

Intuition: ``C(n-c, k) / C(n, k)`` is the probability that a random draw of
``k`` samples *without replacement* from the ``n`` generated contains zero
passing samples. One minus that is the probability at least one of the ``k``
drawn samples passes — exactly what "pass@k" means, and unbiased because it
marginalises over every possible k-subset of the n samples actually drawn,
rather than over a single fresh draw of k.

This module implements only the estimator and its aggregation; running the
generated code to decide pass/fail is :mod:`evaluation.execution`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from evaluation.models import TaskOutcome
from utils.errors import EvaluationError
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)


class ScoringError(EvaluationError):
    """Pass@k could not be computed for the given (n, c, k)."""


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased pass@k estimate for one task.

    Args:
        n: Total samples generated for the task. Must be >= 1.
        c: Samples, of the ``n``, that passed every test. ``0 <= c <= n``.
        k: The k in pass@k. Must satisfy ``1 <= k <= n``.

    Returns:
        A value in ``[0.0, 1.0]``. ``0.0`` when ``c == 0`` (nothing passed,
        so no k-subset can contain a pass); ``1.0`` when ``c == n`` (every
        sample passed, so every k-subset trivially contains one).

    Raises:
        ScoringError: For any input outside the domain the estimator is
            defined on — negative counts, ``c > n``, ``k < 1``, ``n == 0``
            (no samples were generated, so pass@k is undefined rather than
            zero — a task that errored out must not silently read as "0%
            correct"), or ``k > n`` ("insufficient samples": you cannot
            estimate pass@10 from 3 generated samples; generate more samples
            or ask for a smaller k).

    The formula computes ``C(n-c, k) / C(n, k)`` directly via
    :func:`math.comb` rather than the numerically-optimised running-product
    form used by some reference implementations. ``n`` here is at most a few
    hundred (``run.num_samples_per_task``), so exact integer arithmetic is
    both simpler to audit against the written formula and immune to the
    floating-point cancellation the product form exists to avoid at large n.
    """
    if k < 1:
        raise ScoringError(f"pass@k requires k >= 1, got k={k}")
    if n < 0 or c < 0:
        raise ScoringError(f"n and c must be non-negative, got n={n}, c={c}")
    if c > n:
        raise ScoringError(f"c cannot exceed n (c={c} > n={n}); c counts passes out of n samples")
    if n == 0:
        raise ScoringError(
            "pass@k is undefined for n=0 (no samples were generated for this task); "
            "a task with zero completions must be excluded from the aggregate, not scored as 0"
        )
    if k > n:
        raise ScoringError(
            f"pass@{k} requires at least {k} sample(s) but only n={n} were generated "
            f"(insufficient samples); lower k or raise run.num_samples_per_task"
        )
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass(frozen=True)
class TaskPassAtK:
    """Per-task pass@k outcome for every requested k.

    ``computed`` holds the k values that could be scored; ``skipped`` records
    the k values that could not be (with why), so a run with
    ``num_samples_per_task < max(pass_at_k)`` reports that fact instead of
    silently omitting numbers.
    """

    task_id: str
    n: int
    c: int
    computed: dict[int, float]
    skipped: dict[int, str]


def task_pass_at_k(outcome: TaskOutcome, ks: Sequence[int]) -> TaskPassAtK:
    """Compute pass@k for every ``k`` in ``ks`` for one scored task.

    Args:
        outcome: A :class:`~evaluation.models.TaskOutcome` whose ``passed``
            field has been populated by :mod:`evaluation.execution` (a list
            of one bool per completion — *not* ``None``).
        ks: The k values to compute (e.g. ``[1, 10]`` from
            ``configs/evaluation.yaml``'s ``scoring.pass_at_k``).

    Raises:
        ScoringError: If ``outcome.passed`` is ``None`` (the task was never
            executed) — this is a caller bug, not a data condition, so it is
            not folded into ``skipped``.
    """
    if outcome.passed is None:
        raise ScoringError(
            f"Task {outcome.task_id} has not been executed (passed=None); "
            f"run evaluation.execution before scoring"
        )
    n = len(outcome.passed)
    c = sum(1 for p in outcome.passed if p)

    computed: dict[int, float] = {}
    skipped: dict[int, str] = {}
    for k in ks:
        try:
            computed[k] = pass_at_k(n, c, k)
        except ScoringError as exc:
            skipped[k] = str(exc)
    return TaskPassAtK(task_id=outcome.task_id, n=n, c=c, computed=computed, skipped=skipped)


@dataclass(frozen=True)
class AggregatePassAtK:
    """Mean pass@k across every scorable task in a benchmark run.

    ``tasks_included`` and ``tasks_skipped`` are keyed by k because a task
    can be skipped for one k (e.g. pass@10 with 3 samples) while still
    contributing to another (pass@1). Reporting per-k counts is what keeps
    this "honest" per the Week-3 requirement: a shrunken denominator is
    visible, never silently absorbed into the mean.
    """

    pass_at_k: dict[int, float]
    tasks_included: dict[int, int]
    tasks_skipped: dict[int, list[str]]
    per_task: list[TaskPassAtK]

    def to_dict(self) -> dict:
        return {
            "pass_at_k": {str(k): v for k, v in sorted(self.pass_at_k.items())},
            "tasks_included": {str(k): v for k, v in sorted(self.tasks_included.items())},
            "tasks_skipped": {str(k): v for k, v in sorted(self.tasks_skipped.items())},
        }


def aggregate_pass_at_k(outcomes: Sequence[TaskOutcome], ks: Sequence[int]) -> AggregatePassAtK:
    """Mean pass@k, per k, across every *executed* task in ``outcomes``.

    Tasks whose ``passed`` is ``None`` (never executed — usually because
    generation itself failed for that task) are excluded from every k and
    counted in that k's ``tasks_skipped`` list with a clear reason, matching
    the "no silent hiding of failures" requirement: a benchmark run with
    generation failures reports a smaller ``tasks_included`` count rather
    than quietly averaging over fewer tasks than the headline suggests.
    """
    per_task: list[TaskPassAtK] = []
    sums: dict[int, float] = {k: 0.0 for k in ks}
    counts: dict[int, int] = {k: 0 for k in ks}
    skipped: dict[int, list[str]] = {k: [] for k in ks}

    for outcome in outcomes:
        if outcome.passed is None:
            for k in ks:
                skipped[k].append(f"{outcome.task_id}: not executed (error={outcome.error!r})")
            continue
        result = task_pass_at_k(outcome, ks)
        per_task.append(result)
        for k in ks:
            if k in result.computed:
                sums[k] += result.computed[k]
                counts[k] += 1
            else:
                skipped[k].append(f"{outcome.task_id}: {result.skipped[k]}")

    means = {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in ks}
    if any(counts[k] == 0 for k in ks):
        empty = [k for k in ks if counts[k] == 0]
        _LOG.warning("pass@%s has zero scorable tasks; reporting 0.0 rather than omitting it", empty)

    return AggregatePassAtK(
        pass_at_k=means,
        tasks_included=counts,
        tasks_skipped=skipped,
        per_task=per_task,
    )


def pass_at_k_from_mapping(counts: Mapping[str, tuple[int, int]], k: int) -> dict[str, float]:
    """Convenience: ``{task_id: (n, c)}`` -> ``{task_id: pass@k}``.

    Used by the sanity-check script and tests to compute pass@k without
    constructing full :class:`TaskOutcome` objects.
    """
    return {task_id: pass_at_k(n, c, k) for task_id, (n, c) in counts.items()}
