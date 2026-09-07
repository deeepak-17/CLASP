# CLASP-P5 · In-Project Completion Metric ("in-project eval v1")

Reference for `evaluation/in_project.py` and `scripts/run_in_project_eval.py`.

## 1. Why this exists

The D5 promotion rule the State Registry runs is **two-sided**:

> promote iff the *in-project* metric improves beyond `baseline_noise_band`
> **and** the HumanEval pass@1 guard has not dropped more than 2 points; else roll back.

HumanEval/MBPP pass@k (`evaluation/scoring.py`) is only the **guard** — a regression tripwire. It does not answer "did this adapter get better at *the code this client actually writes*?" That is the **primary** signal, and until now nothing in P5 measured it: `contracts.InProjectMetrics` defines the fields, P1 measures only perplexity, so `EvalResult.in_project` could not be filled honestly and seam C2 was theatre.

This module fills two of the three `InProjectMetrics` fields.

## 2. What it measures

**Next-line completion** over each federated client's **held-out** `.py` files:

| Field | How |
|---|---|
| `exact_match` | fraction of held-out lines the model reproduces exactly (trailing whitespace ignored, leading indentation kept) |
| `edit_similarity` | mean of `1 - levenshtein(pred, target) / max(len(pred), len(target))` over the held-out lines — character-level, the CodeXGLUE code-completion convention. In `[0, 1]`; `1.0` is a perfect line |
| `perplexity` | **`null` here.** P5 has no logits. In the integrated pipeline P1 hands its held-out perplexity across and it is merged into `InProjectMetrics` at that seam |

### The held-out set

`partitions.materialize.split_held_out` carves a deterministic, seeded 10% slice off every client's shard — the same slice materialized for P1 to train *without*, at the same `seed` / `held_out_fraction`. So a client is never scored on a file it trained on, and the split is byte-reproducible from the committed `datasets/partitions/*.jsonl` without needing `datasets/materialized/`.

### Example selection

For each held-out file, every line with ≥ `min_prefix_lines` (default 3) non-blank lines of context above it is a candidate, except blank lines, comment-only lines, docstring delimiters, and lines longer than `max_target_chars` (200). Candidates across all of a client's files are pooled, hash-ordered by `sha256(seed:file_id:line_no)`, and the first `max_examples_per_client` (default 60) are kept — same hash-before-slice trick as the held-out split, so one large file can't dominate and the first lines of every file aren't over-sampled.

## 3. The noise band

`noise_band(values)` = population standard deviation of the held-out `edit_similarity` across **N repeated** baseline evaluations (`--repeats`, default 3). It is `EvalResult.baseline_noise_band`: the smallest in-project gain D5 should treat as signal rather than run-to-run jitter.

With a **deterministic** backend (the mock, or greedy `temperature=0`) every repeat is byte-identical and the band is `0.0` — correct, and exactly the placeholder the panel notes flag. It becomes a real gate once a stochastic backend (`temperature > 0`, or sampling) makes the repeats differ; the mechanism is in place and tested (`tests/test_in_project.py::TestNoiseBand`, and an evaluator test with a varying stub client).

## 4. Running it

```bash
python scripts/run_in_project_eval.py                       # all 6 clients, mock backend, 3 repeats
python scripts/run_in_project_eval.py --client client-flask --repeats 5
python scripts/run_in_project_eval.py --limit 20 --seed 7
python scripts/run_in_project_eval.py --backend edge        # P1's client, once registered
```

Writes:

- `evaluation/results/in_project/<run_id>.json` — per-client metrics, the noise band, and a **schema-valid `EvalResult` per client** (validated against `interfaces/schemas/eval_result.schema.json`, proving the shape P4's rule reads).
- `reports/in_project_eval_report.md`.

## 5. REAL vs DEMO/TEST

`MockEdgeInferenceClient` emits a fixed placeholder line, so against it `edit_similarity ≈ 0.18`, `exact_match = 0.0`, `baseline_noise_band = 0.0` for every client — the **correct** output of a working metric given a generator that never produces real code, **not** a measurement. Every artefact is labelled `DEMO_TEST`. A real number needs `--backend edge` against P1's merged model.

## 6. Contract changes (`interfaces/contracts.py`, `CONTRACT_VERSION` 0.2.0 → 0.3.0)

- New `InProjectMetrics(edit_similarity, exact_match, n_examples, perplexity=None)` — mirrors P4's frozen `contracts.InProjectMetrics` (P5's mirror keeps `perplexity` optional because P5 standalone cannot measure it).
- `EvalResult` gains `in_project: InProjectMetrics | None = None` and `baseline_noise_band: float = 0.0`. Both optional with backward-compatible defaults — a 0.2.0 guard-only document still loads and still validates.
- `interfaces/schemas/eval_result.schema.json` extended with the optional `in_project` object and `baseline_noise_band`.

## 7. What is still not done

- No **real** model has been evaluated — same environment limit as pass@k (no GPU / checkpoint / trained adapter).
- `perplexity` is not wired from P1 yet; that happens at the W7 integration seam.
- Composition-order (D3) caveat is unchanged: the clients were trained against the frozen base, not `base + α·cluster`.
