"""Dashboard-facing results index — Week 3, Friday.

"Store results in a simple results.json the dashboard can read."

Two artefacts, two audiences
-----------------------------
:mod:`evaluation.harness` already writes one JSON file per run to
``evaluation/results/<run_id>.json``, containing every raw completion. That
document is deliberately **not** the dashboard's contract: it is large (every
sample of every task), it is not shaped like
:class:`~interfaces.contracts.EvalResult`, and there is one per run rather
than one place a dashboard can read to answer "show me every scored result".

This module writes the other artefact: ``evaluation/results/results.json``,
an append-only *index* of :class:`ResultRecord` — one entry per scored
benchmark run, each wrapping a schema-valid ``EvalResult`` plus the extra
metadata a dashboard (or a panel) needs that the frozen P4 contract does not
carry:

* which dataset/split was evaluated and how many problems it had;
* the generation configuration and seed, for reproducibility;
* whether the model backend was P1's real merged model or the offline mock
  (``provenance``) — the field this whole exercise exists to keep honest;
* a pointer back to the big raw-completions file, per "keep huge raw
  completions separate from the dashboard-facing results".

The ``eval_result`` sub-object of every record still validates against
``interfaces/schemas/eval_result.schema.json`` unmodified — extending the
results architecture, not replacing or loosening the frozen contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from interfaces.contracts import CONTRACT_VERSION, EvalResult
from interfaces.validation import validate_document
from utils.errors import ClaspP5Error
from utils.io_utils import read_json, write_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.timing import utc_timestamp

_LOG = get_logger(__name__)

#: Version of the results.json *index* document shape (independent of
#: CONTRACT_VERSION, which versions only the nested eval_result).
RESULTS_INDEX_VERSION = "1.0.0"

Provenance = Literal["REAL", "DEMO_TEST"]


class ResultsStoreError(ClaspP5Error):
    """A results.json record failed to build or validate."""


@dataclass(frozen=True)
class GenerationSnapshot:
    """The decoding configuration a run used, for reproducibility."""

    max_new_tokens: int
    temperature: float
    stop_sequences: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stop_sequences": list(self.stop_sequences),
        }


@dataclass(frozen=True)
class ResultRecord:
    """One dashboard-facing entry: an ``EvalResult`` plus P5-local run metadata.

    Answers, per the Week-3 Friday requirement, every question the dashboard
    needs and the frozen ``EvalResult`` contract does not carry on its own:
    model/checkpoint, dataset/split, problem/sample counts, seed, generation
    config, timestamp, and REAL vs DEMO/TEST provenance.
    """

    eval_result: EvalResult
    model_checkpoint: str
    dataset_split: str
    num_problems: int
    seed: int | None
    generation: GenerationSnapshot
    provenance: Provenance
    provenance_note: str
    raw_artifact_path: str | None = None
    benchmark_data_source: str | None = None

    def __post_init__(self) -> None:
        if self.provenance not in ("REAL", "DEMO_TEST"):
            raise ResultsStoreError(f"provenance must be REAL or DEMO_TEST, got {self.provenance!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_result": self.eval_result.to_dict(),
            "run_metadata": {
                "model_checkpoint": self.model_checkpoint,
                "dataset_split": self.dataset_split,
                "num_problems": self.num_problems,
                "seed": self.seed,
                "generation": self.generation.to_dict(),
                "provenance": self.provenance,
                "provenance_note": self.provenance_note,
                "raw_artifact_path": self.raw_artifact_path,
                "benchmark_data_source": self.benchmark_data_source,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultRecord":
        try:
            meta = data["run_metadata"]
            return cls(
                eval_result=EvalResult.from_dict(data["eval_result"]),
                model_checkpoint=str(meta["model_checkpoint"]),
                dataset_split=str(meta["dataset_split"]),
                num_problems=int(meta["num_problems"]),
                seed=meta.get("seed"),
                generation=GenerationSnapshot(**meta["generation"]),
                provenance=meta["provenance"],
                provenance_note=str(meta.get("provenance_note", "")),
                raw_artifact_path=meta.get("raw_artifact_path"),
                benchmark_data_source=meta.get("benchmark_data_source"),
            )
        except (KeyError, TypeError) as exc:
            raise ResultsStoreError(f"Malformed results.json record: {exc}") from exc

    def validate(self) -> None:
        """Raise if the nested ``eval_result`` does not satisfy its contract."""
        report = validate_document(
            "eval_result", self.eval_result.to_dict(), document_name=f"ResultRecord({self.eval_result.run_id})"
        )
        report.raise_for_status()


@dataclass
class ResultsIndex:
    """The full contents of ``results.json``."""

    index_version: str
    updated_at: str
    records: list[ResultRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "updated_at": self.updated_at,
            "contract_version": CONTRACT_VERSION,
            "results": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultsIndex":
        return cls(
            index_version=str(data.get("index_version", RESULTS_INDEX_VERSION)),
            updated_at=str(data.get("updated_at", "")),
            records=[ResultRecord.from_dict(r) for r in data.get("results", [])],
        )


def default_results_path() -> Path:
    """Canonical location of the dashboard-facing results index."""
    return project_paths().eval_results / "results.json"


def load_results_index(path: Path | str | None = None) -> ResultsIndex:
    """Load ``results.json``, returning an empty index if it does not exist yet."""
    target = Path(path) if path else default_results_path()
    if not target.is_file():
        return ResultsIndex(index_version=RESULTS_INDEX_VERSION, updated_at=utc_timestamp())
    return ResultsIndex.from_dict(read_json(target))


def append_results(records: list[ResultRecord], path: Path | str | None = None) -> Path:
    """Validate and append ``records`` to ``results.json``, then write it atomically.

    Records for the same ``run_id`` + ``benchmark`` replace the prior entry
    (re-running the baseline eval updates its own record rather than
    accumulating duplicates); anything else is appended.
    """
    target = Path(path) if path else default_results_path()
    for record in records:
        record.validate()

    index = load_results_index(target)
    by_key = {(r.eval_result.run_id, r.eval_result.benchmark.value): i for i, r in enumerate(index.records)}
    for record in records:
        key = (record.eval_result.run_id, record.eval_result.benchmark.value)
        if key in by_key:
            index.records[by_key[key]] = record
        else:
            by_key[key] = len(index.records)
            index.records.append(record)

    index.updated_at = utc_timestamp()
    write_json(target, index.to_dict())
    _LOG.info(
        "results.json: %d record(s) total (%d added/updated this run) -> %s",
        len(index.records),
        len(records),
        project_paths().relative(target),
    )
    return target
