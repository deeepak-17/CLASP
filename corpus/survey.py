"""Dataset survey and selection — Week 1, Monday.

"Survey candidate datasets (multi-project code corpora)."

The survey is expressed as data (``configs/dataset_survey.yaml``) and scored
by code, rather than written as prose. That buys three things:

* the recorded decision and the corpus the pipeline actually builds cannot
  silently diverge;
* re-weighting a criterion re-runs the whole comparison in a second;
* an override — selecting something other than the top scorer — is detectable
  and is reported as a warning rather than passing unnoticed.

Scores are editorial judgements captured in the config with justifications.
This module supplies the arithmetic and the audit trail, not the opinions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.config import load_config
from utils.errors import ConfigError, CorpusError
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport

_LOG = get_logger(__name__)

#: Permitted score range for every criterion.
SCORE_MIN, SCORE_MAX = 1, 5

#: Tolerance when checking that criterion weights sum to 1.0.
_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Criterion:
    """One weighted requirement the corpus must satisfy."""

    key: str
    label: str
    weight: float
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise ConfigError(f"Criterion '{self.key}' weight must lie in (0, 1], got {self.weight}")


@dataclass(frozen=True)
class Candidate:
    """One corpus option under consideration."""

    key: str
    name: str
    kind: str
    scores: dict[str, int]
    url: str = ""
    approx_size: str = ""
    licensing: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for criterion_key, value in self.scores.items():
            if not SCORE_MIN <= int(value) <= SCORE_MAX:
                raise ConfigError(
                    f"Candidate '{self.key}' score for '{criterion_key}' must be "
                    f"{SCORE_MIN}-{SCORE_MAX}, got {value}"
                )


@dataclass(frozen=True)
class SurveyConfig:
    """Root of ``configs/dataset_survey.yaml`` under the ``survey`` key."""

    name: str
    owner: str
    criteria: list[Criterion]
    candidates: list[Candidate]
    selected: str
    selection_rationale: str = ""
    known_limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ConfigError("survey.criteria must not be empty")
        if not self.candidates:
            raise ConfigError("survey.candidates must not be empty")

        total_weight = sum(c.weight for c in self.criteria)
        if abs(total_weight - 1.0) > _WEIGHT_TOLERANCE:
            raise ConfigError(f"survey.criteria weights must sum to 1.0, got {total_weight:.6f}")

        criterion_keys = {c.key for c in self.criteria}
        if len(criterion_keys) != len(self.criteria):
            raise ConfigError("survey.criteria contains duplicate keys")

        for candidate in self.candidates:
            missing = criterion_keys - set(candidate.scores)
            if missing:
                raise ConfigError(
                    f"Candidate '{candidate.key}' is missing score(s) for: {', '.join(sorted(missing))}"
                )
            extra = set(candidate.scores) - criterion_keys
            if extra:
                raise ConfigError(
                    f"Candidate '{candidate.key}' scores unknown criterion/criteria: "
                    f"{', '.join(sorted(extra))}"
                )

        if self.selected not in {c.key for c in self.candidates}:
            raise ConfigError(
                f"survey.selected='{self.selected}' is not among the declared candidates"
            )

    def candidate(self, key: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.key == key:
                return candidate
        raise CorpusError(f"No candidate '{key}' in survey")  # pragma: no cover - guarded above


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with its weighted total and per-criterion contributions."""

    candidate: Candidate
    weighted_score: float
    contributions: dict[str, float]
    rank: int
    is_selected: bool

    @property
    def key(self) -> str:
        return self.candidate.key

    @property
    def normalised_score(self) -> float:
        """Weighted score rescaled to 0-100 for readability."""
        return round(100.0 * self.weighted_score / SCORE_MAX, 1)


@dataclass(frozen=True)
class SurveyOutcome:
    """Result of scoring every candidate in the survey."""

    config: SurveyConfig
    ranking: list[ScoredCandidate]
    selected: ScoredCandidate
    top_scorer: ScoredCandidate
    warnings: list[str] = field(default_factory=list)

    @property
    def selection_is_top_scorer(self) -> bool:
        return self.selected.key == self.top_scorer.key


def load_survey_config(path: Path | str | None = None) -> SurveyConfig:
    """Load ``configs/dataset_survey.yaml`` into a :class:`SurveyConfig`."""
    target = path or (project_paths().configs / "dataset_survey.yaml")
    return load_config(SurveyConfig, target, section="survey")


def score_survey(config: SurveyConfig) -> SurveyOutcome:
    """Compute the weighted ranking and validate the recorded selection."""
    scored: list[tuple[Candidate, float, dict[str, float]]] = []
    for candidate in config.candidates:
        contributions = {
            criterion.key: round(criterion.weight * float(candidate.scores[criterion.key]), 4)
            for criterion in config.criteria
        }
        scored.append((candidate, round(sum(contributions.values()), 4), contributions))

    # Ties break on key so the ranking is stable across runs.
    scored.sort(key=lambda item: (-item[1], item[0].key))

    ranking = [
        ScoredCandidate(
            candidate=candidate,
            weighted_score=total,
            contributions=contributions,
            rank=index + 1,
            is_selected=candidate.key == config.selected,
        )
        for index, (candidate, total, contributions) in enumerate(scored)
    ]

    selected = next(item for item in ranking if item.is_selected)
    top = ranking[0]

    warnings: list[str] = []
    if selected.key != top.key:
        warnings.append(
            f"Selected candidate '{selected.key}' (rank {selected.rank}, "
            f"score {selected.weighted_score}) is not the top scorer "
            f"'{top.key}' (score {top.weighted_score}). This is a documented override — "
            f"confirm survey.selection_rationale justifies it."
        )
        _LOG.warning(warnings[-1])
    else:
        _LOG.info(
            "Selected candidate '%s' is the top scorer (%.4f / %d)",
            selected.key,
            selected.weighted_score,
            SCORE_MAX,
        )

    return SurveyOutcome(
        config=config,
        ranking=ranking,
        selected=selected,
        top_scorer=top,
        warnings=warnings,
    )


def render_survey_report(outcome: SurveyOutcome) -> MarkdownReport:
    """Render the Week-1 Monday deliverable as a Markdown report."""
    config = outcome.config
    report = MarkdownReport(
        title="CLASP-P5 · Dataset Survey (D1 Corpus Selection)",
        subtitle=f"Data & Demo · {config.owner}",
    )

    report.heading("1. Decision")
    report.key_values(
        {
            "Survey": config.name,
            "Selected corpus": f"{outcome.selected.candidate.name} (`{outcome.selected.key}`)",
            "Weighted score": f"{outcome.selected.weighted_score} / {SCORE_MAX} "
            f"({outcome.selected.normalised_score}%)",
            "Rank": f"{outcome.selected.rank} of {len(outcome.ranking)}",
            "Licensing": outcome.selected.candidate.licensing,
            "Approx. size": outcome.selected.candidate.approx_size,
        }
    )
    report.paragraph(config.selection_rationale.strip())
    if outcome.warnings:
        report.heading("Override notice", level=3)
        report.bullets(outcome.warnings)

    report.heading("2. Evaluation criteria")
    report.paragraph(
        "Weights encode CLASP's actual constraints: an unambiguous project boundary is "
        "the precondition for cluster-level LoRA, and per-file licence certainty is the "
        "precondition for the on-premise enterprise framing."
    )
    report.table(
        ["Criterion", "Weight", "Why it matters"],
        [[c.label, f"{c.weight:.2f}", " ".join(c.description.split())] for c in config.criteria],
    )

    report.heading("3. Candidate ranking")
    headers = ["Rank", "Candidate", "Kind", *[c.label for c in config.criteria], "Weighted", "Selected"]
    rows: list[list[Any]] = []
    for item in outcome.ranking:
        rows.append(
            [
                item.rank,
                item.candidate.name,
                item.candidate.kind,
                *[item.candidate.scores[c.key] for c in config.criteria],
                f"{item.weighted_score:.3f}",
                "yes" if item.is_selected else "",
            ]
        )
    report.table(headers, rows)
    report.paragraph(f"Raw scores are {SCORE_MIN}–{SCORE_MAX} (higher is better).")

    report.heading("4. Candidate notes")
    for item in outcome.ranking:
        report.heading(f"{item.rank}. {item.candidate.name}", level=3)
        report.key_values(
            {
                "Key": f"`{item.candidate.key}`",
                "Kind": item.candidate.kind,
                "Source": item.candidate.url or "—",
                "Approx. size": item.candidate.approx_size or "—",
                "Licensing": item.candidate.licensing or "—",
                "Weighted score": f"{item.weighted_score:.3f} ({item.normalised_score}%)",
            }
        )
        if item.candidate.notes:
            report.paragraph(" ".join(item.candidate.notes.split()))

    report.heading("5. Known limitations of the selected corpus")
    report.paragraph(
        "Recorded here so they can be cited as threats to validity in the Phase-II report "
        "rather than discovered at review time."
    )
    report.bullets(config.known_limitations)

    report.rule()
    report.paragraph(
        "Generated by `corpus/survey.py` from `configs/dataset_survey.yaml`. "
        "Edit the config and re-run `python scripts/survey_datasets.py` to regenerate; "
        "do not hand-edit this file."
    )
    return report
