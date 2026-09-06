# CLASP-P5 · Evaluation Harness (HumanEval / MBPP / Pass@k)

Reference doc for the Week-1 scaffold and the Week-3 real implementation. Referenced from `evaluation/humaneval/adapter.py` and `evaluation/mbpp/adapter.py`.

## 1. Two supported routes to task data

Every benchmark adapter (`evaluation/base.py::BenchmarkAdapter.load_tasks`) resolves its input in this order:

1. **`tasks_path`** (`configs/evaluation.yaml`) — the real, published benchmark, fetched by `scripts/fetch_benchmark_data.py` (see §2). This is what every Week-3 script uses.
2. **`sample_tasks_path`** — the bundled 5-task fixture (`evaluation/{humaneval,mbpp}/sample_tasks.jsonl`), original tasks authored for this repository, format-compatible but explicitly **not** the published benchmark. Used automatically, with a loud log warning, whenever `tasks_path` is missing — this is what keeps the test suite and a fresh, offline clone working exactly as they did in Week 1.

Both routes produce the same `EvalTask` shape (`evaluation/models.py`), so nothing downstream — prompt building, assembly, execution, scoring — needs to know which route supplied a task.

## 2. Fetching the real benchmarks

```bash
python scripts/fetch_benchmark_data.py            # both benchmarks
python scripts/fetch_benchmark_data.py --benchmark humaneval
python scripts/fetch_benchmark_data.py --force     # re-download and overwrite
```

| Benchmark | Source | Licence | What's kept |
|---|---|---|---|
| HumanEval | `openai/human-eval`, `data/HumanEval.jsonl.gz` | MIT | All 164 problems, unfiltered |
| MBPP | `google-research/google-research`, `mbpp/mbpp.jsonl` | CC-BY-4.0 | Test split, task_id 11–510 (Austin et al. 2021 few-shot/eval convention) minus records whose reference solution isn't a single top-level `def` matching the entry point read off `test_list[0]` — 392 of 500 kept; the rest are counted, not silently dropped |

Each fetch writes a `tasks_manifest.json` sidecar (source URL, SHA-256 of the raw download, licence, fetch timestamp, exact written/skipped counts) next to `tasks.jsonl`, so provenance is always machine-checkable.

Not run automatically and not required: this script needs network access; everything else in the repository (tests, dry runs, the sanity check against the bundled fixture) works without it.

## 3. The merged-model interface

`interfaces/edge_client.py::EdgeInferenceClient` is the protocol both benchmarks call through — one interface, `evaluation/registry.py` dispatches by benchmark to the right adapter but always through the same client.

- **`MockEdgeInferenceClient`** — deterministic, offline, always produces a syntactically valid but intentionally incorrect completion. Used by every test and every "dry run".
- **`TransformersEdgeInferenceClient`** (`interfaces/edge_transformers_adapter.py`) — the real adapter: loads a HF causal-LM (optionally 4-bit via `bitsandbytes`, matching P1's Week-1 baseline), composes any LoRA adapters via `peft`, serves `generate()`. Constructing it without `torch`/`transformers`/(`peft` if adapters given)/(`bitsandbytes` if 4-bit) installed raises `DependencyError` with the exact `pip install` command — never silently falls back to the mock.
- **Wiring it in**: `interfaces.edge_transformers_adapter.register_transformers_edge_client(model_id, adapters=[...])`, then set `backend.kind: edge` in `configs/evaluation.yaml`. `evaluation/registry.py::register_edge_client_factory` is the seam; P1 (or anyone) never needs to edit P5's code to plug in a real checkpoint.

**Status in this repository:** the adapter is implemented and unit-tested (its shaping, error path, and registration are all exercised), but not run against real DeepSeek-Coder-6.7B weights — no GPU, no downloaded checkpoint, no trained LoRA adapter exist in this environment. Every Pass@k number this repository reports comes from `MockEdgeInferenceClient` and is labelled `DEMO_TEST` in `results.json`.

## 4. Pass@k

`evaluation/scoring.py` implements the unbiased estimator (Chen et al. 2021, eq. 1):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

- `n` — samples generated for one task
- `c` — of those, how many passed every test
- `k` — the "k" in pass@k

Computed via `math.comb` directly against the written formula (exact integer arithmetic; `n` here is at most a few hundred, so there's no numerical-stability reason to use the running-product form some references use). Edge cases are explicit `ScoringError`s, not silent zeros or crashes: `n=0` ("undefined, not measured as 0%"), `k>n` ("insufficient samples"), `c>n`, `k<1`, negative counts.

`evaluation/execution.py::execute_program` determines `passed`/`failed` by actually running the assembled program (prompt + completion + tests) in a subprocess, with a timeout and best-effort POSIX resource limits. **This is process-level isolation, not container-level** — see that module's docstring for exactly what it does and does not guarantee, and why running it against genuinely untrusted input (as opposed to this repository's own mock/reference completions) still wants P3 sign-off on a stronger sandbox.

`evaluation/harness.py::EvaluationHarness` wires the two together when `scoring.execution_enabled: true`: every assembled program is executed, `TaskOutcome.passed` is populated, and `evaluation.scoring.aggregate_pass_at_k` computes the per-benchmark summary. A task/`k` combination that can't be scored (e.g. `num_samples_per_task=1` but `pass_at_k: [1, 10]` is configured) is reported by name in the log and excluded from that `k`'s mean — never silently absorbed into a lower denominator.

## 5. Baseline run and sanity checks

```bash
python scripts/run_baseline_eval.py --limit 1000 --num-samples 10 --k 1 10
python scripts/sanity_check_scoring.py
```

- **`run_baseline_eval.py`** — the harness against `MockEdgeInferenceClient`, scored for real. Pass@k is *expected* to be 0.0 for every task (the mock never produces a passing completion by design) — that is the scorer working correctly given a generator that never passes, not a bug. Writes `evaluation/results/results.json` (see §6) and `evaluation/results/<run_id>.json` (raw completions).
- **`sanity_check_scoring.py`** — the real check: every task's own canonical/reference solution, run through the identical execute→score path, must pass its own test. On the real HumanEval/MBPP data fetched via §2: **164/164 and 392/392 pass, pass@1 = 1.0000, ~10s total**. No model involved.

## 6. `results.json` (`evaluation/results_store.py`)

Two artefacts, two audiences:

- `evaluation/results/<run_id>.json` — one per run, every raw completion (can be megabytes; kept out of the dashboard's way).
- `evaluation/results/results.json` — the dashboard-facing index. One `ResultRecord` per scored benchmark run:
  - `eval_result` — schema-valid against `interfaces/schemas/eval_result.schema.json` unmodified (adapter ref, benchmark, `pass_at_k`, task/sample counts, run id, timestamp).
  - `run_metadata` — everything the frozen contract doesn't carry: `model_checkpoint`, `dataset_split` (with problem count and source URL), `seed`, `generation` config snapshot, **`provenance: "REAL" | "DEMO_TEST"`** plus a human-readable `provenance_note`, and `raw_artifact_path` pointing at the big per-run file.

`append_results` validates every record before writing; re-running for the same `(run_id, benchmark)` updates that record in place rather than duplicating it.

## Reproducing everything in this document

```bash
pytest tests/ -v                                                  # full suite
python scripts/fetch_benchmark_data.py                            # real HumanEval + MBPP
python scripts/sanity_check_scoring.py                            # real end-to-end check
python scripts/run_baseline_eval.py --limit 1000 --num-samples 10 --k 1 10
```
