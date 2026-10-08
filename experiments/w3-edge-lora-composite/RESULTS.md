# W3 · P1 Edge — results summary

Run config: [`config.yaml`](config.yaml). Raw manifests in `results/` (gitignored;
regenerate with the commands below). Machine: RTX 2050, 4 GB VRAM, torch 2.13.0+cu126,
transformers 5.14.1, Python 3.10.11.

**Every number here is on the 1.3B `dev` profile.** The 6.7B `target` profile does not
fit in 4 GB (see `docs/hardware.md`), so none of these are comparable to a 6.7B figure
or to published HumanEval numbers.

---

## E3.1 — Client LoRA on a real partition · **MET**

Acceptance was "loss decreases over a real run under D8 caps without OOM".

| | |
|---|---|
| client | `web/client-requests` (16 files, 53,753 tokens, 52 blocks) |
| budget | 59 blocks = 1.13 epochs (sqrt policy) |
| loss (windowed) | 1.1455 → 0.6929 |
| held-out ppl | 3.46 → 3.36 (−0.102) |
| **peak VRAM** | **1.747 GB** of 4 GB |
| **throughput** | **3.05 s/optimizer step** |

Raw: `results/e31_client-requests_manifest.json`

### Real block counts (seq 1024)

| client | blocks | sqrt budget | epochs |
|---|---|---|---|
| scientific/client-numpy | 1286 | 292 | 0.23 |
| scientific/client-pandas | 1687 | 335 | 0.20 |
| scientific/client-scikit-learn | 1696 | 335 | 0.20 |
| web/client-flask | 80 | 73 | 0.91 |
| web/client-requests | 52 | 59 | 1.14 |
| web/client-werkzeug | 170 | 106 | 0.62 |
| **total** | **4971** | **1200** | |

Raw: `results/budget_plan.json`

### Two numbers for the D8/D11 conversation

- **A 1200-block round costs ~61 min** at 3.05 s/step, against D11's "one federated
  round ≤ 30 min on the reference GPU". Not a violation on this hardware (a 2050 is not
  the reference GPU) but it is the first measured figure.
- **VRAM headroom is large** — 1.75 GB of 4 GB. `micro_batch=2` or higher `grad_accum`
  is affordable, which matters because E3.2 wants a less noisy gradient signal.

### Proposed D8 amendment (needs team sign-off)

D8's flat "≤ 200 steps/client/round" produces a **33× epoch spread** across clients
(requests 3.3 epochs, scikit-learn 0.11). The sqrt policy redistributes the *same* 1200
blocks and cuts the spread to **5.7×**. Same GPU-hours; only the distribution changes.
Adopting it means three clients exceed 200 steps, hence the amendment.

Knock-on effects to raise with the owners:
- **P3 / D7** — Opacus ε depends on step count, so ε becomes per-client. The constraint
  becomes max-over-clients ≤ 8, and scikit-learn at 335 blocks is the binding case.
- **P2 / D2** — straggler skip-after-timeout assumes uniform client wall-time; the
  timeout has to become relative to each client's assigned budget.
- Aggregation weighting is **unchanged** and stays proportional to sample count.
  Scaling both local work and aggregation weight would square the size bias.

---

## E3.2 — Stable lr/steps config · **NOT CLOSED** (grid did not bracket the optimum)

Sweep 1 complete: `web/client-flask`, 10 cells, 43.5 min. Raw: `results/sweep_flask.json`.
Held-out perplexity delta from base 2.45 — more negative is better.

| lr | ga=1 | ga=4 |
|---|---|---|
| 5e-5 | −0.036 | −0.013 |
| 1e-4 | −0.066 | −0.026 |
| 2e-4 | −0.107 | −0.051 |
| 5e-4 | −0.154 | −0.098 |
| 1e-3 | **−0.184** | −0.143 |

**Every cell was stable.** No NaN, no divergence, all slopes negative, max grad norm 0.30
across the whole grid. The divergence guard never fired.

Sweep 1's improvement was monotonic right to the edge of the grid, so 1e-3 won by being
the last column rather than by being optimal. Sweep 2 extended upward until it broke.
Raw: `results/sweep_flask_high.json`.

| lr | Δppl | max grad norm | outcome |
|---|---|---|---|
| 1e-3 | −0.184 | 0.27 | stable |
| **2e-3** | **−0.194** | 0.40 | **best** |
| 5e-3 | −0.006 | 2.03 | learns nothing generalizable |
| 1e-2 | — | — | **NaN — divergence guard fired** |

The optimum is now bracketed: peak at 2e-3, collapse by 5e-3, hard failure at 1e-2. The
usable band is roughly 1e-4 … 2e-3, an order of magnitude wide, which is a comfortable
place to sit.

### The 5e-3 cell exposed a hole in the stability gate

At 5e-3 the training loss still fell (1.165 → 0.926, negative slope) so `loss_decreased`
was `True`, and held-out perplexity moved −0.006 — i.e. nothing. The run learned to fit
its own batches and generalized not at all, yet passed a gate built on the loss curve
plus a perplexity-not-worse check.

The tell was **max grad norm 2.03 against a clip threshold of 1.0**, versus ~0.3 across
the entire healthy band. Clipping saturating means nearly every update is being truncated
— the optimizer is fighting the LR on every step.

`stable` now also requires `max_grad_norm <= 2 x grad_clip`, reported as
`clip_saturated`. It is a leading indicator: it fires on the mechanism, before the loss
curve shows anything. The recorded sweep JSONs predate this and carry the old flag; the
`Δppl` and `gn` columns above are the raw evidence either way.

### `grad_accum=1` beat `grad_accum=4` at every LR, by roughly 2×

The opposite of the expectation — a larger effective batch should give a cleaner
gradient. The cause is almost certainly the budget accounting: at a fixed *block* budget,
ga=4 takes a quarter as many optimizer steps (19 vs 73). At these budgets, step count
dominates gradient quality. That makes it an artifact of the small budget rather than a
general result, and it needs re-checking on a scientific client before being believed.

### Caveat that limits what any flask sweep can conclude

`client-flask`'s held-out split is **2 files**; `client-requests`'s is 514 tokens, one
block. A perplexity delta on that is inside the noise, so these rankings are suggestive,
not decisive. The memorization check only becomes meaningful on the scientific clients
(22–26 held-out files each). **The winner must be validated on
`scientific/client-numpy` before it is recorded as the answer to E3.2**, and P5's noise
band (W5) is what would let these deltas be read as real rather than plausible.

`config.yaml`'s `lr: 2e-4` stays provisional until both are done.

---

## E3.3 — Composite merge, base ⊕ α·cluster ⊕ β·client · **DONE**

`edge.merge` builds **one** pre-merged composite (D6 — no runtime dual-adapter stacking).

Built by **rank concatenation**, not SVD truncation: stack `A` along the rank axis, fold
`coef × scaling` into `B`, pin composite scaling to 1.0. The block-matrix product then
reproduces `α·ΔW_cluster + β·ΔW_client` exactly.

The choice is driven by E3.4. Under SVD the correctness identities hold only to within
truncation error, and a gate that can only say "close enough" cannot separate a rounding
artifact from a sign error. Cost is a rank-32 served adapter; revisit once D2's SVD lands.

Composites built against the real `client-requests` adapter (96 modules each):

| α | β | rank | merge self-check (relative) |
|---|---|---|---|
| 0.5 | 1.0 | 32 | 4.078e-07 |
| 0.3 | 1.0 | 32 | 5.195e-07 |
| 0.0 | 1.0 | 16 | **0.000e+00** |

Raw: `results/e33_merge_manifest_*.json`

`validate_compatibility` runs before every merge, covering the silent-failure mode in
P1's scope: base-model, target-module and structural mismatch between adapters, plus each
adapter against the frozen contract hyperparameters. Differing ranks between adapters are
allowed (concatenation handles them); the contract is what pins r=16.

**D3 deviation:** no real cluster adapter exists until P2 ships SVD aggregation (W5+), so
the cluster term is a stand-in (`stub:zeros` or `stub:random`) and the client trained on
the frozen base alone. Recorded as `d3_deviation` in every manifest.

---

## E3.4 — Merge correctness vs unmerged baseline · **PASSED**

`services/edge/tests/test_merge.py` — 33 tests. The three gates, on both synthetic
adapters and the real 192-tensor one:

| identity | result |
|---|---|
| α=0, β=0 → recovers base | `count_nonzero(ΔW) == 0`, every module |
| α=0, β=1 → equals client adapter alone | **bitwise** (`torch.equal`), every module |
| α=1, β=0 → equals cluster alone | bitwise |
| general (α,β) vs independent reference | rel. error < 1e-6 (fp32 ε is 1.19e-7) |

Bitwise rather than approximate because the pinned config has `lora_alpha == r`, so every
scaling is exactly 1.0 and the α=0 case reduces to multiplication by 1.0, which IEEE-754
performs exactly.

`compose` is checked against `composite_delta`, a deliberately naive independent
implementation. Testing the concatenation trick against itself would prove nothing.

Also covered: linearity in (α,β), sign preservation, non-unit and rsLoRA scaling,
differing ranks, module union, no source mutation, save/load round trip, contract
violations rejected.

### Two bugs this found

- **Memory.** `delta_weights` materializes dense full-size ΔW — 96 × 2048×2048 fp32 ≈
  1.5 GB per adapter. The merge self-check held several at once and killed the
  interpreter with an access violation. Replaced by `merge_max_error`, which streams
  module-by-module. This was in the production path, not just tests.
- **Tolerance, not math.** The rsLoRA case failed at 1.53e-05 absolute — but ΔW entries
  reach |80| there, so that is 1.9e-7 *relative*, i.e. fp32 epsilon. Inexact comparisons
  now use relative tolerance; the α=0 identities stay `== 0.0`.

---

## E3.5 — First TTFT, composite vs base · **BOTH NFRs PASS**

Rank-32 composite (α=0.5, β=1.0), 20 timed repeats after 3 discarded warmups, CUDA
synchronized on both sides of every timer. Raw: `results/ttft_results.json`.

| prompt | base median | composite median | overhead |
|---|---|---|---|
| 2 tok | 166.13 ms | 232.49 ms | **+66.36 ms** |
| 9 tok | 198.18 ms | 230.12 ms | +31.94 ms |
| 30 tok | 202.46 ms | 268.30 ms | +65.84 ms |

| NFR (D11) | budget | measured | verdict |
|---|---|---|---|
| TTFT overhead vs base | ≤ 200 ms | +66.36 ms worst | **PASS** |
| adapter swap | ≤ 2 s | 0.501 s median | **PASS** |

Peak VRAM 0.886 GB base → 0.951 GB composite (+65 MB for rank 32 across 96 modules).
First cold adapter load 0.688 s — reported separately because it is *not* the swap
metric; swap keeps the base resident and moves only the adapter.

### Two things the medians hide

- **Swap p95 is 1.387 s against a 2 s budget.** The median has 4× headroom, the tail has
  1.4×. On a 1.3B adapter. This is the number to watch when the 6.7B target profile
  comes back, not the median.
- **Composite p95 reaches 368 ms** (vs base 244 ms), so p95 overhead is ~124 ms — still
  inside 200 ms, but roughly double the median overhead.

Both numbers are 1.3B-dev-profile on an RTX 2050. Overhead scales with model size and
adapter rank, so neither carries over to 6.7B unexamined. Rank is the lever if it ever
gets tight: dropping the composite from rank-32 concat to a rank-16 SVD re-factorization
(D2's machinery) would roughly halve the adapter's contribution to prefill.

---

## Reproduce

```bash
cd services/edge
python -m edge.chunking --plan --policy sqrt --total-chunks 1200
python -m edge.train_client --client web/client-requests --budget-plan budget_plan.json
python -m edge.sweep_lr --clients web/client-flask --lrs 5e-5 1e-4 2e-4 5e-4 1e-3 --grad-accums 1 4
python -m edge.merge --client artifacts/adapters/client-requests --cluster stub:random --alpha 0.5 --beta 1.0 --out artifacts/composites/a05
python -m edge.ttft --composite artifacts/composites/a05 --swap-to artifacts/composites/a03
pytest tests/ -q
```

## Status

| task | state |
|---|---|
| E3.1 real LoRA train | **met** |
| E3.2 stable lr/steps | **not closed** — optimum outside the grid; sweep 2 running |
| E3.3 composite merge (D6) | **done** |
| E3.4 merge correctness | **passed**, 33 tests |
| E3.5 TTFT + swap (D11) | **both NFRs pass** |

## Open items

- [ ] E3.2 sweep 2 (2e-3 / 5e-3 / 1e-2) to find where stability breaks; then validate the
      winner on `scientific/client-numpy`; then update `config.yaml`'s provisional `lr`
- [ ] D8 amendment (sqrt budgeting) tabled with the team
- [ ] Determinism is partial — torch warns the memory-efficient attention backward is
      non-deterministic and `warn_only=True` lets it proceed. The *dataset* is fully
      reproducible (sorted order + `order_sha256`); the loss curve is not bit-identical
      across runs. Decide before quoting E3.2 numbers as exact.
- [ ] **D1 deviation:** the web cluster on disk is {flask, requests, werkzeug}; D1
      specifies {django, flask, requests}. Werkzeug is an unrecorded substitution and
      `datasets/materialized/web/_extra/` holds three further repos (click, colorama,
      jinja). Raise with P5 — "two semantically real clusters" is a paper claim.
- [ ] Partition manifests still carry no `corpus_sha256`, so there is no traceability
      back to the source corpus.
