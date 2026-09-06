"""In-project completion metric — P5's "in-project eval v1" (D5 primary metric).

Why this exists
---------------
The D5 two-sided promotion rule the State Registry runs needs a *primary*
signal — "did this adapter get better at the code it will actually be used
on?" — not just the HumanEval/MBPP regression guard (:mod:`evaluation.scoring`).
Perplexity alone, which is all P1 measures during training, cannot fill
:class:`~interfaces.contracts.InProjectMetrics` honestly. This module measures
the other two fields.

What it measures
----------------
**Next-line completion** over each federated client's *held-out* ``.py`` files
(the deterministic 10% slice :func:`partitions.materialize.split_held_out`
carves off so no client is ever scored on a file it trained on):

* ``exact_match``     — fraction of held-out lines the model reproduces exactly
  (trailing whitespace ignored, leading indentation kept).
* ``edit_similarity`` — mean character-level normalised edit similarity
  ``1 - lev(pred, target) / max(len(pred), len(target))`` — the CodeXGLUE
  code-completion convention. Lies in ``[0, 1]``; 1.0 is a perfect line.

``perplexity`` stays :data:`None` here: P5 has no logits. In the integrated
pipeline P1 hands its held-out perplexity across and it is merged into the
:class:`~interfaces.contracts.InProjectMetrics` at that seam.

The noise band
--------------
:func:`noise_band` turns *N* repeated baseline evaluations into
``EvalResult.baseline_noise_band`` — the population standard deviation of the
held-out edit similarity across the repeats. It is the smallest improvement
D5 should treat as real rather than run-to-run jitter. With a deterministic
backend (the mock, or greedy decoding) every repeat is identical and the band
is 0.0 — correct, and exactly the placeholder the panel notes flag: it only
becomes a gate once a stochastic backend makes the repeats differ.

Backends
--------
Driven through :class:`~interfaces.edge_client.EdgeInferenceClient`, same as
the benchmark harness. Against :class:`~interfaces.edge_client.MockEdgeInferenceClient`
the numbers are near-zero by construction (the mock never emits real code) and
must be labelled ``DEMO_TEST`` — see ``scripts/run_in_project_eval.py``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from corpus.models import CorpusRecord
from interfaces.contracts import InProjectMetrics, PartitionManifest
from interfaces.edge_client import EdgeInferenceClient, GenerationRequest
from partitions.materialize import split_held_out
from partitions.partitioner import load_shard_records
from utils.errors import ClaspP5Error, EvaluationError
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)

_TRIPLE_QUOTES = ('"""', "'''")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InProjectConfig:
    """Parameters for one in-project evaluation.

    Defaults mirror ``configs/materialize_client.yaml`` for the held-out cut
    (``seed``/``held_out_fraction``) so this scores exactly the files P1 was
    told to hold out, and use short greedy decoding for the completion itself.
    """

    seed: int = 20260616
    held_out_fraction: float = 0.10
    #: Upper bound on completion examples drawn per client. Held-out sets are
    #: small (a handful of files); this caps the busy ones so one large client
    #: does not dominate the aggregate.
    max_examples_per_client: int = 60
    #: A candidate line needs at least this many non-blank lines of context
    #: above it, so the model is completing with real preceding code.
    min_prefix_lines: int = 3
    #: Skip pathologically long target lines (generated tables, vendored data).
    max_target_chars: int = 200
    max_new_tokens: int = 48
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.held_out_fraction < 1.0:
            raise EvaluationError("in_project.held_out_fraction must lie strictly between 0 and 1")
        if self.max_examples_per_client < 1:
            raise EvaluationError("in_project.max_examples_per_client must be >= 1")
        if self.min_prefix_lines < 1:
            raise EvaluationError("in_project.min_prefix_lines must be >= 1")
        if self.max_target_chars < 1:
            raise EvaluationError("in_project.max_target_chars must be >= 1")
        if self.max_new_tokens < 1:
            raise EvaluationError("in_project.max_new_tokens must be >= 1")
        if self.temperature < 0.0:
            raise EvaluationError("in_project.temperature must be >= 0")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompletionExample:
    """One held-out line to predict, with the file context that precedes it."""

    example_id: str          # "<file_id>#L<n>"
    file_id: str
    relative_path: str
    line_number: int          # 1-based, the line being predicted
    prefix: str               # every line above it, newline-terminated
    target: str               # the held-out line itself (verbatim)


@dataclass(frozen=True)
class ExampleScore:
    """Per-example outcome."""

    example_id: str
    line_number: int
    target: str
    prediction: str
    edit_similarity: float
    exact_match: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "line_number": self.line_number,
            "target": self.target,
            "prediction": self.prediction,
            "edit_similarity": round(self.edit_similarity, 6),
            "exact_match": self.exact_match,
            "error": self.error,
        }


@dataclass(frozen=True)
class InProjectEvalResult:
    """Aggregate in-project metrics for one client, plus per-example detail."""

    client_id: str
    cluster_id: str
    edit_similarity: float
    exact_match: float
    n_examples: int
    n_held_out_files: int
    seed: int
    perplexity: float | None = None
    generation_errors: int = 0
    per_example: tuple[ExampleScore, ...] = field(default_factory=tuple)

    def to_metrics(self) -> InProjectMetrics:
        """The contract object the D5 rule consumes."""
        return InProjectMetrics(
            edit_similarity=self.edit_similarity,
            exact_match=self.exact_match,
            n_examples=self.n_examples,
            perplexity=self.perplexity,
        )

    def to_dict(self, *, include_examples: bool = True) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "cluster_id": self.cluster_id,
            "edit_similarity": round(self.edit_similarity, 6),
            "exact_match": round(self.exact_match, 6),
            "n_examples": self.n_examples,
            "n_held_out_files": self.n_held_out_files,
            "generation_errors": self.generation_errors,
            "perplexity": self.perplexity,
            "seed": self.seed,
            "examples": [s.to_dict() for s in self.per_example] if include_examples else [],
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings (iterative, O(len(a)*len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,        # deletion
                    current[j - 1] + 1,     # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def char_edit_similarity(prediction: str, target: str) -> float:
    """``1 - lev(prediction, target) / max(len)``, clamped to ``[0, 1]``.

    Both empty is a perfect match (1.0) — a model correctly predicting an
    empty completion for an empty target should not be punished.
    """
    if not prediction and not target:
        return 1.0
    denom = max(len(prediction), len(target))
    return 1.0 - levenshtein(prediction, target) / denom


def line_exact_match(prediction: str, target: str) -> bool:
    """Exact-match a predicted line against the target, ignoring trailing whitespace."""
    return prediction.rstrip() == target.rstrip()


def noise_band(values: Sequence[float]) -> float:
    """Population standard deviation of ``values`` — the D5 baseline noise band.

    Returns 0.0 for fewer than two values: with a single baseline evaluation
    there is no observable run-to-run variation to band.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------
def _file_lines(content: str) -> list[str]:
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # a trailing newline is a terminator, not an empty line
    return lines


def _is_scorable_target(line: str, *, max_target_chars: int) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    if stripped.startswith(_TRIPLE_QUOTES):
        return False
    if len(line) > max_target_chars:
        return False
    return True


def _selection_key(seed: int, file_id: str, line_index: int) -> str:
    """Deterministic hash order for choosing which lines to score.

    Same ``sha256(seed:key:index)`` construction as
    :func:`partitions.materialize._holdout_sort_key`, for the same reason:
    a hash order avoids the artefact of always scoring the first lines of
    every file.
    """
    return hashlib.sha256(f"{seed}:{file_id}:{line_index}".encode("utf-8")).hexdigest()


def build_completion_examples(
    records: Sequence[CorpusRecord], *, config: InProjectConfig
) -> list[CompletionExample]:
    """Turn held-out files into a capped, deterministic set of completion tasks.

    Every eligible ``(file, line)`` pair across all ``records`` is pooled, hash
    -ordered by ``(seed, file_id, line_index)``, and the first
    ``config.max_examples_per_client`` are kept, then restored to
    ``(file_id, line_number)`` order. A file with no content, or with no line
    that clears :func:`_is_scorable_target` given enough preceding context, is
    skipped.
    """
    candidates: list[CompletionExample] = []
    for record in sorted(records, key=lambda r: r.file_id):
        if not record.content:
            continue
        lines = _file_lines(record.content)
        for index in range(config.min_prefix_lines, len(lines)):
            if not _is_scorable_target(lines[index], max_target_chars=config.max_target_chars):
                continue
            prefix_lines = lines[:index]
            if not any(pl.strip() for pl in prefix_lines):
                continue
            candidates.append(
                CompletionExample(
                    example_id=f"{record.file_id}#L{index + 1}",
                    file_id=record.file_id,
                    relative_path=record.relative_path,
                    line_number=index + 1,
                    prefix="\n".join(prefix_lines) + "\n",
                    target=lines[index],
                )
            )

    candidates.sort(key=lambda ex: _selection_key(config.seed, ex.file_id, ex.line_number))
    kept = candidates[: config.max_examples_per_client]
    kept.sort(key=lambda ex: (ex.file_id, ex.line_number))
    return kept


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class InProjectEvaluator:
    """Drives an :class:`EdgeInferenceClient` over a set of completion examples."""

    def __init__(self, client: EdgeInferenceClient, config: InProjectConfig | None = None) -> None:
        self._client = client
        self._config = config or InProjectConfig()

    def score_examples(self, examples: Sequence[CompletionExample]) -> list[ExampleScore]:
        """Generate and score one completion per example. Never raises per-example."""
        scores: list[ExampleScore] = []
        for example in examples:
            prediction, error = self._complete(example)
            scores.append(
                ExampleScore(
                    example_id=example.example_id,
                    line_number=example.line_number,
                    target=example.target,
                    prediction=prediction,
                    edit_similarity=char_edit_similarity(prediction.rstrip(), example.target.rstrip()),
                    exact_match=line_exact_match(prediction, example.target),
                    error=error,
                )
            )
        return scores

    def evaluate(
        self,
        *,
        client_id: str,
        cluster_id: str,
        examples: Sequence[CompletionExample],
        n_held_out_files: int,
        perplexity: float | None = None,
    ) -> InProjectEvalResult:
        """Score ``examples`` and roll them up into an :class:`InProjectEvalResult`."""
        if not examples:
            raise EvaluationError(
                f"Client '{client_id}' produced no eligible completion examples "
                "(held-out files too short or all comments/blank/docstring lines)"
            )
        scores = self.score_examples(examples)
        n = len(scores)
        return InProjectEvalResult(
            client_id=client_id,
            cluster_id=cluster_id,
            edit_similarity=math.fsum(s.edit_similarity for s in scores) / n,
            exact_match=math.fsum(1.0 for s in scores if s.exact_match) / n,
            n_examples=n,
            n_held_out_files=n_held_out_files,
            seed=self._config.seed,
            perplexity=perplexity,
            generation_errors=sum(1 for s in scores if s.error is not None),
            per_example=tuple(scores),
        )

    # --- internals --------------------------------------------------------
    def _complete(self, example: CompletionExample) -> tuple[str, str | None]:
        request = GenerationRequest(
            task_id=example.example_id,
            prompt=example.prefix,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            stop_sequences=("\n",),
            num_samples=1,
        )
        try:
            result = self._client.generate(request)
        except ClaspP5Error as exc:
            _LOG.warning("in-project completion failed for %s: %s", example.example_id, exc)
            return "", str(exc)
        except Exception as exc:  # a broken client must not abort the sweep
            _LOG.exception("in-project completion raised for %s", example.example_id)
            return "", f"{type(exc).__name__}: {exc}"

        raw = result.completions[0] if result.completions else ""
        # First line only: clients that ignore stop_sequences (the mock) still
        # get scored on a single predicted line.
        return raw.split("\n", 1)[0].rstrip(), None


def evaluate_client_in_project(
    manifest: PartitionManifest,
    client_id: str,
    client: EdgeInferenceClient,
    *,
    config: InProjectConfig | None = None,
    perplexity: float | None = None,
) -> InProjectEvalResult:
    """End-to-end: load a client's shard, hold out a slice, score next-line completion.

    Reuses :func:`partitions.materialize.split_held_out` so the held-out set is
    byte-identical to the one materialized for P1 at the same ``seed`` and
    ``held_out_fraction`` — the client is never scored on a file it trained on.
    """
    config = config or InProjectConfig()
    records = load_shard_records(manifest, client_id)  # raises ContractViolationError if unknown
    missing = [r.file_id for r in records if r.content is None]
    if missing:
        raise EvaluationError(
            f"Shard '{client_id}' has {len(missing)} record(s) with no content; "
            "cannot run in-project completion (re-materialize the shard with content)"
        )

    _, held_out = split_held_out(
        records, client_id=client_id, seed=config.seed, fraction=config.held_out_fraction
    )
    examples = build_completion_examples(held_out, config=config)
    _LOG.info(
        "in-project eval %s: %d held-out file(s) -> %d completion example(s)",
        client_id,
        len(held_out),
        len(examples),
    )
    shard = manifest.shard_for(client_id)
    return InProjectEvaluator(client, config).evaluate(
        client_id=client_id,
        cluster_id=shard.cluster_id,
        examples=examples,
        n_held_out_files=len(held_out),
        perplexity=perplexity,
    )
