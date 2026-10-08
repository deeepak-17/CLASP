# CLASP-P5 · Week 1 Completion Checklist

## Monday: Dataset Survey
- [x] Survey candidate datasets (`configs/dataset_survey.yaml`)
- [x] Weighted scoring against 6 criteria
- [x] Record selection with justification
- [x] `corpus/survey.py` — scorer + report generator
- [x] `scripts/survey_datasets.py` — CLI
- [x] Report: `reports/dataset_survey.md`

## Tuesday: Corpus Collection
- [x] Declarative source config (`configs/dataset.yaml`)
- [x] Three acquisition modes: git / local / synthetic
- [x] `corpus/acquisition.py` — GitAcquirer, LocalAcquirer, SyntheticAcquirer
- [x] `corpus/filters.py` — glob filtering, size bounds, SLOC floor, deduplication
- [x] `corpus/collector.py` — pipeline
- [x] `scripts/collect_dataset.py` — CLI
- [x] Artefact: `datasets/processed/corpus.jsonl` (135 files, 45,922 code lines)
- [x] Artefact: `datasets/metadata/corpus_manifest.json` (provenance)
- [x] Report: `reports/corpus_collection_report.md`

## Wednesday: Eval Harness Scaffold
- [x] `evaluation/models.py` — normalised task/outcome types
- [x] `evaluation/base.py` — BenchmarkAdapter base class
- [x] `evaluation/humaneval/adapter.py` — HumanEvalAdapter
- [x] `evaluation/mbpp/adapter.py` — MbppAdapter
- [x] `evaluation/registry.py` — adapter registry + backend factory
- [x] `evaluation/harness.py` — runner (scoring deferred to Week 3)
- [x] Bundled sample fixtures (5 tasks per benchmark, format-compatible)

## Thursday: Dependency Preflight
- [x] `evaluation/dependencies.py` — preflight checks
- [x] `scripts/check_dependencies.py` — CLI
- [x] Week-1/2 requirements satisfied (PyYAML, pytest, jsonschema)
- [x] Week-3+ requirements declared but not yet blocking
- [x] Report: `reports/dependency_report.md`

## Friday: Harness Dry Run
- [x] `scripts/dry_run_harness.py` — CLI
- [x] Load tasks from both benchmarks
- [x] Build prompts (HumanEval verbatim, MBPP with example + stub)
- [x] Generate completions via `MockEdgeInferenceClient`
- [x] Truncate at stop sequences
- [x] Assemble runnable programs (syntactically valid)
- [x] Write run artefact to `evaluation/results/<run_id>.json`
- [x] Pass@k scoring explicitly deferred to Week 3
- [x] Report: `reports/dry_run_report.md`

## Cross-cutting Week-1 Deliverables
- [x] `utils/` — paths, I/O, logging, config, timing, text, reporting
- [x] `interfaces/contracts.py` — contract mirror (AdapterRef, PartitionManifest, EvalResult)
- [x] `interfaces/edge_client.py` — P1 protocol + mock
- [x] `interfaces/security_client.py` — P3 privacy accountant placeholder
- [x] `configs/logging.yaml` — centralised logging config
- [x] `conftest.py` — pytest fixtures
- [x] Tests: utils, corpus, evaluation (93 passing at end of Week 1)

**Week 1 status: COMPLETE**
