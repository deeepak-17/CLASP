# CLASP-P5 · Week 3 Completion Checklist

**Scope:** P5 Week-3 daily targets (CLASP_Daily_Targets.pdf, Phase II). Week 1/2 functionality is frozen baseline and untouched except where explicitly noted.

## Monday: Wire HumanEval runner to the merged model
- [x] `evaluation/registry.py` — `register_edge_client_factory` / `build_inference_client("edge")` seam (already scaffolded Week 1/2; now provably reachable — see below)
- [x] `interfaces/edge_transformers_adapter.py` — real `EdgeInferenceClient` implementation over HF Transformers + PEFT (4-bit via bitsandbytes, matching P1's Week-1 baseline config)
- [x] `TransformersEdgeInferenceClient` fails with a clear, actionable `DependencyError` when `torch`/`transformers`/`peft`/`bitsandbytes` are absent (verified live: none are installed in this environment)
- [x] `register_transformers_edge_client()` — one-call registration; flips `backend.kind: edge` from rejected to "reachable, missing-dependency reported precisely"
- [x] Test: registering the factory makes `backend.kind: edge` stop raising "not integrated" and start raising the real dependency error instead

## Tuesday: Wire MBPP runner similarly
- [x] Same `EdgeInferenceClient` protocol serves both benchmarks — no per-benchmark wiring needed (`evaluation/registry.py`'s adapter registry already dispatches HumanEval/MBPP through one client)
- [x] `scripts/fetch_benchmark_data.py` — fetches the real HumanEval (164, openai/human-eval, MIT) and MBPP (392 of the 500-task test split, google-research/google-research, CC-BY-4.0; 108 skipped as unparseable multi-statement solutions, counted not hidden)
- [x] `configs/evaluation.yaml` — `tasks_path` points at the fetched real files; falls back to the bundled 5-task fixture (loudly, via existing Week-1 logic) when the fetch has not been run — offline-first preserved
- [x] `evaluation/mbpp/sample_tasks.jsonl` — fixed a latent bug the new real-execution path surfaced: the bundled fixture had no recorded `signature`, so the adapter's generic `*args` stub broke parameter-name binding inside canonical solutions. Added real signatures.

## Wednesday: Implement Pass@k scoring function
- [x] `evaluation/scoring.py` — unbiased estimator `pass@k = 1 - C(n-c,k)/C(n,k)` (Chen et al. 2021)
- [x] Handles: pass@1, arbitrary valid k, k=n, c=0, c=n, n=0, k>n, insufficient samples, invalid inputs — each with its own test in `TestPassAtK`
- [x] `task_pass_at_k` / `aggregate_pass_at_k` — per-task and per-benchmark aggregation; a task/​k that cannot be scored is *skipped and reported*, never silently folded into the mean
- [x] `evaluation/execution.py` — subprocess-based executor (timeout + best-effort POSIX resource limits) that determines pass/fail by actually running the assembled program
- [x] `evaluation/harness.py` — wired: `scoring.execution_enabled` drives real execution + scoring inside `EvaluationHarness`; `TaskOutcome.passed` stays the Week-1 tri-state design (`None` = not scored)

## Thursday: Run Pass@k on the Week-1 baseline model; sanity-check the numbers
- [x] `scripts/run_baseline_eval.py` — runs the harness (scoring on) against `MockEdgeInferenceClient`, labelled **DEMO/TEST** throughout (P1's real DeepSeek-Coder-6.7B is not available in this environment — no GPU, no weights, no adapter; see the script's docstring for the exact reasoning)
- [x] `scripts/sanity_check_scoring.py` — the **REAL** check: every canonical/reference solution in the real HumanEval (164) and MBPP (392) task sets, executed for real, must pass its own test. Result: **164/164 and 392/392 pass, pass@1 = 1.0000 for both**, ~10s total.
- [x] Sanity checks in `run_baseline_eval.py`: passed <= attempted, every pass@k in [0,1], task counts match the loaded dataset — all verified programmatically, non-zero exit on failure
- [x] Reproducibility: `test_run_is_reproducible_when_scored` — identical config produces identical pass@k twice

## Friday: Store results in results.json
- [x] `evaluation/results_store.py` — `ResultRecord` (schema-valid `EvalResult` + dashboard metadata: model/checkpoint, dataset split, problem/sample counts, seed, generation config, **REAL vs DEMO_TEST provenance + note**, pointer to the big raw-completions file)
- [x] `evaluation/results/results.json` — the dashboard-facing index; raw per-run completions stay in the existing `evaluation/results/<run_id>.json` artefacts (1.2 MB for the full 5,560-completion baseline run vs 4 KB for the index)
- [x] `append_results` validates every record against `interfaces/schemas/eval_result.schema.json` before writing — an invalid record cannot reach disk

## Cross-cutting Week-3 deliverables
- [x] Tests: `evaluation.scoring`, `evaluation.execution`, `evaluation.results_store`, `interfaces.edge_transformers_adapter`, harness scoring integration, real-data loading (skipped gracefully when the fetch has not run) — 86 new/updated tests
- [x] Every existing Week-1/2 test still passes; two Week-1 placeholder tests (`compute_pass_at_k` "explicitly deferred", `PerDeveloperStrategy` "not implemented") were updated to assert the now-real behaviour, exactly as their own docstrings anticipated
- [x] Full suite: 313 passing after Week 3 (up from the 256-test Week-1/2 baseline)
- [x] No credentials, API keys or secrets introduced (see Security section of the final report)

**Week 3 status: COMPLETE** — real data, real execution, real scoring; the only mock is the model backend itself, which is labelled DEMO/TEST everywhere it appears and never presented as a capability measurement.

**Explicitly deferred (out of P5's Week-3 scope, per the daily targets):**
- Real DeepSeek-Coder-6.7B / P1 merged-model inference (needs P1's checkpoint + a GPU; the adapter and registration path are ready)
- Sandboxed execution beyond process-level isolation (P3 sign-off on a container/gVisor-grade sandbox — see `evaluation/execution.py`'s module docstring)
- Dashboard (React + Recharts) — Week 5
