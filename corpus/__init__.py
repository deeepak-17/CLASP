"""D1 corpus: survey, sourcing, acquisition and collection.

Week-1 scope (Mon–Tue of the CLASP daily plan):

* :mod:`corpus.survey` — score candidate multi-project code corpora and record
  the selection with its justification.
* :mod:`corpus.models` — configuration schema and corpus record types.
* :mod:`corpus.acquisition` — git / local / synthetic sourcing strategies.
* :mod:`corpus.filters` — deterministic file selection and deduplication.
* :mod:`corpus.collector` — the pipeline that produces ``corpus.jsonl`` and
  its provenance manifest.

Downstream, :mod:`partitions` consumes ``corpus.jsonl``; nothing in this
package knows partitioning exists.
"""

from corpus.collector import (
    CollectionResult,
    CorpusCollector,
    load_corpus_config,
    load_corpus_records,
)
from corpus.models import CorpusConfig, CorpusManifest, CorpusRecord, SourceConfig, SourceStats
from corpus.survey import SurveyOutcome, load_survey_config, render_survey_report, score_survey

__all__ = [
    "CollectionResult",
    "CorpusCollector",
    "CorpusConfig",
    "CorpusManifest",
    "CorpusRecord",
    "SourceConfig",
    "SourceStats",
    "SurveyOutcome",
    "load_corpus_config",
    "load_corpus_records",
    "load_survey_config",
    "render_survey_report",
    "score_survey",
]
