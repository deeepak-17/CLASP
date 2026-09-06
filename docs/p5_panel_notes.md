# CLASP-P5 · Panel Notes (Week 5)

Speaking notes for presenting P5 — Eval & Data — at the Week-6 panel review, plus a ~3 minute script. Every claim below is backed by a file in this repository; none is invented for the presentation.

## 1. P5 responsibility

P5 (Eval & Data / Data & Demo Lead) owns evaluating CLASP's merged model output and presenting the result. Concretely: the D1 corpus and its federated partitioning (Weeks 1–2, 4), the HumanEval/MBPP evaluation harness and Pass@k scoring (Week 3), and the dashboard that reads the resulting `results.json` (Week 5).

## 2. Why evaluation is required

CLASP's central claim is that a federated, multi-layer LoRA composition (`W_base + alpha·dW_cluster + beta·dW_client`) produces a usable code model without centralising training data. That claim is only checkable by running the composed model against held-out coding benchmarks and measuring whether it actually produces correct code — which is exactly what P5's harness does. Without it, "the model works" is an assertion, not a result.

## 3. HumanEval

164 hand-written Python function-completion problems (Chen et al., 2021), fetched from the benchmark's own published source (`openai/human-eval`, MIT licence) via `scripts/fetch_benchmark_data.py` — the full set, unfiltered. The model is shown a function signature and docstring and must complete the body; a bundled test harness (`check(candidate)`) determines pass/fail by actually executing the candidate.

## 4. MBPP

The "Mostly Basic Python Problems" benchmark (Austin et al., 2021). This repository uses the standard test split (task_id 11–510, the paper's own few-shot/eval convention), fetched from `google-research/google-research` (CC-BY-4.0). 392 of the 500 in-range problems are kept — 108 were excluded because their reference solution isn't a single parseable top-level function, and that count is recorded, not hidden, in `evaluation/mbpp/tasks_manifest.json`.

## 5. Pass@k

The unbiased estimator from the Codex/HumanEval paper: `pass@k = 1 − C(n−c, k) / C(n, k)`, where `n` is samples generated per task, `c` is how many passed, `k` is the number of attempts being modelled. Implemented in `evaluation/scoring.py`, with every edge case (zero samples, `k` exceeding `n`, all-pass, all-fail) handled explicitly rather than silently. Verified against real data: every one of HumanEval's 164 and MBPP's 392 canonical reference solutions passes its own test when run through the identical scoring path (`scripts/sanity_check_scoring.py` — Pass@1 = 1.0000 for both) — proof the scorer and executor are correct, independent of any model.

## 6. results.json

`evaluation/results_store.py` writes `evaluation/results/results.json`: one record per scored benchmark run, each a schema-valid `EvalResult` (adapter, benchmark, Pass@k map, task/sample counts) plus run metadata the dashboard needs — model/checkpoint, dataset split, seed, generation config, timestamp, and a mandatory `REAL` / `DEMO_TEST` provenance flag with an explanatory note. Large raw completions live in a separate per-run file; `results.json` stays small and dashboard-facing.

## 7. Dashboard

Week 5's deliverable: a React + TypeScript + Recharts static dashboard (`dashboard/`) that reads `results.json` and nothing else — no backend, no invented numbers. Four pages: Overview (what/which benchmarks/what run/REAL-or-DEMO), Benchmarks (HumanEval and MBPP side by side), Pass@k (the one chart), Evaluation Pipeline (the diagram in §8, for this presentation). See `docs/dashboard.md` for the full write-up.

## 8. Evaluation pipeline

```
Dataset -> Prompt -> Merged Model -> Generated Candidates -> Code Execution -> Pass/Fail -> Pass@k -> results.json -> Dashboard
```

Every stage is a real module: dataset fetch (`scripts/fetch_benchmark_data.py`), prompt construction (`evaluation/{humaneval,mbpp}/adapter.py`), model call (`interfaces/edge_client.py`), execution (`evaluation/execution.py`, a real subprocess run — not a static check), scoring (`evaluation/scoring.py`), storage (`evaluation/results_store.py`), display (`dashboard/`).

## 9. Relationship to the rest of CLASP

```
CLASP
  +-- P1 Edge Layer          (Integration & Inference — the merged model P5 evaluates)
  +-- P2 Cluster Layer       (Documentation & Paper — federated aggregation)
  +-- P3 Security            (Security & QA)
  +-- P4 State Registry      (DevOps & Orchestration — where a real checkpoint would come from)
  +-- P5 Eval & Data         (this module)
        +-- HumanEval / MBPP
        +-- Pass@k
        +-- results.json
        +-- Dashboard
```

P5 is a consumer, not a producer, of the model: it evaluates whatever P1's Edge Layer serves through one stable interface (`EdgeInferenceClient`), so swapping in a real checkpoint later requires no change to the evaluation code — only to which client is registered.

## 10. REAL vs PLACEHOLDER distinction

Every result currently in `results.json` is labelled `DEMO_TEST`, and the dashboard shows this plainly (a `DEMO / PLACEHOLDER DATA` badge and banner) rather than presenting it as a measured result. The reason: P1's real merged DeepSeek-Coder-6.7B checkpoint is not available in this development environment (no GPU, no downloaded weights, no trained LoRA adapter). What *is* real: the benchmark data (164 + 392 published problems), the execution, and the scoring formula — verified independently via the canonical-solution sanity check (§5). What is not real: the specific Pass@k numbers currently shown, because they come from a deterministic mock generator, not a trained model.

## 11. Current limitations

- No real model has been evaluated yet — Pass@k against `MockEdgeInferenceClient` is 0.0 for every task/k by design (the mock never produces a passing completion), which is the correct output of a working scorer given that input, not a result to report as capability.
- The real Edge-Layer adapter (`interfaces/edge_transformers_adapter.py`) is implemented but untested against real weights — no GPU/checkpoint in this environment.
- Execution sandboxing (`evaluation/execution.py`) is process-level (timeout + resource limits), not container-grade; fine for this repository's own completions, would want P3 sign-off before running untrusted input at scale.
- The dashboard is a single static snapshot with no historical trend view (see `docs/dashboard.md` §9 for the full list).

---

## Speaking script (~3 minutes)

> P5 is the evaluation and data layer for CLASP. Our job is to answer one question honestly: does the model CLASP produces actually write correct code?
>
> We evaluate on two published benchmarks — HumanEval, 164 problems, and MBPP, the standard 392-problem test split. Both are fetched from their real published sources, not hand-written stand-ins.
>
> For each problem, we prompt the model, generate candidate completions, and — this is the part that matters — we actually *execute* every candidate against the problem's own tests in a sandboxed subprocess. Pass or fail is determined by running the code, not by inspection.
>
> From those pass/fail outcomes we compute Pass@k using the standard unbiased estimator from the Codex paper. We verified our scorer independently of any model: every one of the 556 real reference solutions across both benchmarks passes its own test when run through our exact scoring path. That's not a model result — it's proof the measurement tool itself is correct.
>
> Every result is written to a `results.json` file with full provenance: which model, which checkpoint, which dataset split, what seed, what generation settings, and — critically — whether the result is REAL or DEMO/PLACEHOLDER data.
>
> This week we built the dashboard that reads that file: an Overview, per-benchmark Pass@k, a chart, and a pipeline diagram — all sourced from that one static file, no backend, no invented numbers.
>
> I want to be upfront about where we are: every number on this dashboard right now is labelled DEMO data. P1's real trained model isn't available in this environment yet — no GPU, no downloaded weights. So instead of hiding that or making the placeholder look like a real result, the dashboard says so, explicitly, on every screen. When a real checkpoint is available, the evaluation pipeline and this dashboard need zero changes to show real numbers — we've already verified the pipeline itself is correct.

(≈ 300 words / ~3 minutes at a measured pace.)
