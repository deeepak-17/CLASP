"""In-project completion metric — P5's "in-project eval v1" (D5 primary metric).

Why this exists
---------------
The D5 two-sided promotion rule the State Registry runs needs a *primary*
signal — "did this adapter get better at the code it will actually be used
on?" — not just the HumanEval/MBPP regression guard (:mod:`eval_harness.scoring`).
Perplexity alone, which is all P1 measures during training, cannot fill
:class:`~interfaces.contracts.InProjectMetrics` honestly. This module measures
the other two fields.

What it measures
----------------
**Next-line completion** over each federated client's *held-out* ``.py`` files
(the deterministic 10% slice :func:`partitions.materialize.split_held_out`
carves off so no client is ever scored on a file it trained on):

* ``exact_match``     — fraction of held-out lines the model reproduces exactly
  (both sides ``strip()``-ed, so indentation is not scored).
* ``edit_similarity`` — mean character-level normalised edit similarity
  ``1 - lev(pred, target) / max(len(pred), len(target))`` on the stripped
  lines — the RepoBench / CodeXGLUE convention. Lies in ``[0, 1]``.

One definition, not two
-----------------------
The metric itself — target filter, example selection, Levenshtein ratio,
exact match, aggregation, example-set fingerprint and noise band — is
:mod:`evaluation.completion` (``services/evaluation``), the module the edge
lane's ``edge.completion_eval`` and the four-seam integration test import.
This module only adapts it to P5's inputs: it builds the examples from a
partition shard's held-out *records* instead of a materialized ``held_out/``
directory, and the selection is arranged so both routes yield the identical
example set (same ``examples_sha256``) — so a number from this harness and a
number from the live round are directly comparable.

``perplexity`` stays :data:`None` here: P5 has no logits. In the integrated
pipeline P1 hands its held-out perplexity across and it is merged into the
:class:`~interfaces.contracts.InProjectMetrics` at that seam.

The noise band
--------------
:func:`noise_band` turns *N* repeated baseline evaluations into
``EvalResult.baseline_noise_band`` — the spread (max − min) of the held-out
edit similarity across the repeats, exactly as
:func:`evaluation.completion.noise_band` computes it. It is the smallest
improvement D5 should treat as real rather than run-to-run jitter. With a
deterministic backend (the mock, or greedy decoding) every repeat is
identical and the band is 0.0 — a true statement about decoding, not the
training-seed band D5 ultimately wants.

Backends
--------
Driven through :class:`~interfaces.edge_client.EdgeInferenceClient`, same as
the benchmark harness. Against :class:`~interfaces.edge_client.MockEdgeInferenceClient`
the numbers are near-zero by construction (the mock never emits real code) and
must be labelled ``DEMO_TEST`` — see ``scripts/run_in_project_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Sequence

from evaluation.corpus.models import CorpusRecord
from evaluation import completion as _canonical
from evaluation.completion import edit_similarity, examples_fingerprint, levenshtein
from evaluation.interfaces.contracts import InProjectMetrics, PartitionManifest
from evaluation.interfaces.edge_client import EdgeInferenceClient, GenerationRequest
from evaluation.partitions.materialize import split_held_out
from evaluation.partitions.partitioner import load_shard_records
from evaluation.utils.errors import ClaspP5Error, EvaluationError
from evaluation.utils.logging_utils import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "CompletionExample",
    "ExampleScore",
    "InProjectConfig",
    "InProjectEvalResult",
    "InProjectEvaluator",
    "build_completion_examples",
    "char_edit_similarity",
    "edit_similarity",
    "evaluate_client_in_project",
    "levenshtein",
    "line_exact_match",
    "noise_band",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InProjectConfig:
    """Parameters for one in-project evaluation.

    Defaults mirror ``configs/materialize_client.yaml`` for the held-out cut
    (``seed``/``held_out_fraction``) so this scores exactly the files P1 was
    told to hold out. ``max_examples_per_client``, ``stride`` and
    ``max_new_tokens`` match ``edge.completion_eval``'s defaults (60 / 7 / 48,
    greedy) so the two lanes score the same problems the same way.
    """

    seed: int = 20260616
    held_out_fraction: float = 0.10
    #: Upper bound on completion examples drawn per client (edge: ``--max-examples``).
    max_examples_per_client: int = 60
    #: Take every ``stride``-th usable line of a file (edge: ``--stride``), so
    #: the picks spread through the file instead of clustering on imports.
    stride: int = 7
    max_new_tokens: int = 48
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.held_out_fraction < 1.0:
            raise EvaluationError("in_project.held_out_fraction must lie strictly between 0 and 1")
        if self.max_examples_per_client < 1:
            raise EvaluationError("in_project.max_examples_per_client must be >= 1")
        if self.stride < 1:
            raise EvaluationError("in_project.stride must be >= 1")
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

    # The two attributes ``evaluation.completion.examples_fingerprint`` reads,
    # so the fingerprint is computed by the canonical function, unchanged.
    @property
    def file(self) -> str:
        return self.relative_path

    @property
    def line_no(self) -> int:
        """0-based, as in ``evaluation.completion.CompletionExample``."""
        return self.line_number - 1


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
    #: ``evaluation.completion.examples_fingerprint`` of the scored set — the
    #: same field ``edge.completion_eval`` records as ``examples_sha256``.
    examples_sha256: str = ""
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
            "examples_sha256": self.examples_sha256,
            "examples": [s.to_dict() for s in self.per_example] if include_examples else [],
        }


# ---------------------------------------------------------------------------
# Metrics — thin names over evaluation.completion (the single implementation)
# ---------------------------------------------------------------------------
def char_edit_similarity(prediction: str, target: str) -> float:
    """Edit similarity of one predicted line, as the D5 metric scores it.

    Both sides are ``strip()``-ed and passed to
    :func:`evaluation.completion.edit_similarity` — the same comparison
    :func:`evaluation.completion.score` applies.
    """
    return edit_similarity(prediction.strip(), target.strip())


def line_exact_match(prediction: str, target: str) -> bool:
    """Exact match on stripped lines, as :func:`evaluation.completion.score` counts it."""
    return prediction.strip() == target.strip()


def noise_band(values: Sequence[float]) -> float:
    """D5 baseline noise band: spread (max − min) of repeated edit similarities.

    Delegates to :func:`evaluation.completion.noise_band`. Returns 0.0 for
    fewer than two values: one evaluation shows no run-to-run variation.
    """
    band, _note = _canonical.noise_band([{"edit_similarity": float(v)} for v in values])
    return float(band)


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------
def build_completion_examples(
    records: Sequence[CorpusRecord], *, config: InProjectConfig
) -> list[CompletionExample]:
    """Next-line problems from held-out records — identical to the directory route.

    ``edge.completion_eval`` calls :func:`evaluation.completion.collect_examples`
    on the materialized ``held_out/`` directory, which
    :func:`partitions.materialize.materialize_client_repo` writes from exactly
    these records. This walks the records in the order ``collect_examples``
    walks the files (``.py`` only, sorted path-component-wise like
    ``sorted(Path.rglob(...))``), with the same per-file quota, the same early
    stop and :func:`evaluation.completion.extract_examples` doing the
    selection — so both routes produce the same example set and the same
    :func:`evaluation.completion.examples_fingerprint`.
    """
    files = sorted(
        (r for r in records if r.content and r.relative_path.endswith(".py")),
        key=lambda r: PurePosixPath(r.relative_path).parts,
    )
    if not files:
        return []
    max_examples = config.max_examples_per_client
    per_file = max(1, -(-max_examples // len(files)) * 2)  # collect_examples' quota

    examples: list[CompletionExample] = []
    for record in files:
        for ex in _canonical.extract_examples(
            record.content, record.relative_path, max_per_file=per_file, stride=config.stride
        ):
            examples.append(
                CompletionExample(
                    example_id=f"{record.file_id}#L{ex.line_no + 1}",
                    file_id=record.file_id,
                    relative_path=record.relative_path,
                    line_number=ex.line_no + 1,
                    prefix=ex.prompt,
                    target=ex.target,
                )
            )
        if len(examples) >= max_examples:
            break
    return examples[:max_examples]


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
            # One model call per example; a failed call scores as an empty prediction, it never aborts the run.
            prediction, error = self._complete(example)
            scores.append(
                ExampleScore(
                    example_id=example.example_id,
                    line_number=example.line_number,
                    target=example.target,
                    prediction=prediction,
                    edit_similarity=char_edit_similarity(prediction, example.target),
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
        # The aggregate is computed by the canonical scorer, not re-derived.
        scored = _canonical.score([s.prediction for s in scores], examples)
        return InProjectEvalResult(
            client_id=client_id,
            cluster_id=cluster_id,
            edit_similarity=float(scored["edit_similarity"]),
            exact_match=float(scored["exact_match"]),
            n_examples=int(scored["n_examples"]),
            n_held_out_files=n_held_out_files,
            seed=self._config.seed,
            perplexity=perplexity,
            generation_errors=sum(1 for s in scores if s.error is not None),
            examples_sha256=examples_fingerprint(examples),
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
        # First line only, cut exactly as edge.completion_eval._first_line does;
        # clients that ignore stop_sequences (the mock) still get one line.
        return raw.split("\n", 1)[0], None


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
    # Step 1: load every file in this client's partition shard.
    records = load_shard_records(manifest, client_id)  # raises ContractViolationError if unknown
    missing = [r.file_id for r in records if r.content is None]
    if missing:
        raise EvaluationError(
            f"Shard '{client_id}' has {len(missing)} record(s) with no content; "
            "cannot run in-project completion (re-materialize the shard with content)"
        )

    # Step 2: take the same 10% held-out files the edge lane never trained on (same seed + fraction).
    _, held_out = split_held_out(
        records, client_id=client_id, seed=config.seed, fraction=config.held_out_fraction
    )
    # Step 3: turn held-out files into next-line completion problems.
    examples = build_completion_examples(held_out, config=config)
    _LOG.info(
        "in-project eval %s: %d held-out file(s) -> %d completion example(s)",
        client_id,
        len(held_out),
        len(examples),
    )
    shard = manifest.shard_for(client_id)
    # Step 4: ask the model for each next line and score it (edit similarity + exact match).
    return InProjectEvaluator(client, config).evaluate(
        client_id=client_id,
        cluster_id=shard.cluster_id,
        examples=examples,
        n_held_out_files=len(held_out),
        perplexity=perplexity,
    )
