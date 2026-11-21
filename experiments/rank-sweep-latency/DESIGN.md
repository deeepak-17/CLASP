# Design note — Rank sweep (r = 4, 8, 16, 32) + serving latency

**Track:** P1 Edge · **Phase III** · **Date:** 2026-11-21 · **Owner:** Adithyaa (P1)
**Companion sweep:** `experiments/rank-sweep/` (quality per rank: P2 runs, P5 scores, P4 orchestrates)

## 1. Question

LoRA rank `r` controls how large each client's update is. A larger `r` can
capture more of a project's style, but it also costs more to train, store,
send and serve. The rank chosen here is **fixed for every later sweep** (α/β,
clustering, ε, μ), so the decision needs both of these:

- **quality per rank**, which the main `rank-sweep` provides, and
- **cost per rank**, which this note covers. Cost here means latency at the
  edge, the time to swap adapters, adapter size, and training cost.

## 2. Hypotheses

| # | Hypothesis | Why we expect it |
|---|---|---|
| H1 | **Merged serving (D6) is rank-free.** A pre-merged composite has the same TTFT and decode speed as the base model at every rank. | Merging folds `B·A` into `W`, so the model's shape is identical whatever `r` is. |
| H2 | **Unmerged serving overhead grows with r but stays within D11** (≤ 200 ms over base) at r = 32. | The LoRA path adds two thin matmuls (`d×r`, `r×d`) per targeted projection. That is tiny next to the 4-bit base matmuls, but it grows with `r`. |
| H3 | **Adapter size grows linearly with r**, and swap time grows with it, staying ≤ 2 s (D11). | Parameter count = `2·r·d·(4 projections)·(24 layers)`, exactly linear in `r`. |
| H4 | **Training cost is almost flat in r**: s/step and peak VRAM change by only a few percent from r = 4 to r = 32. | The frozen base's activations dominate memory and compute. LoRA weights and their Adam state are well under 1% of the base. |

**What would change the plan:** if H1 holds and H2–H4 show small costs, rank
should be chosen **on quality alone** (the P5 curve). If H2 fails at r = 32,
r = 32 is off the table for any unmerged serving path.

## 3. Levels (independent variable)

| Variable | Levels |
|---|---|
| LoRA rank `r` | **4, 8, 16, 32** |

r = 32 is above the D8 cap (16). It is a single justified override
(`caps_override` in `config.yaml`) so the curve shows where gains stop.

**Held fixed across levels:** base `deepseek-coder-1.3b-base` in NF4, targets
`q/k/v/o_proj`, `lora_alpha = r` (scaling 1.0, so the update scale does not
drift with rank), lr 2e-4, seq len 1024, micro-batch 1, same prompts, same GPU.

> Note for the merge step: rank-concat composition (E3.3) **adds** ranks, so a
> client trained at r combined with a cluster adapter at r gives a composite
> of rank 2r. Unmerged serving of an r = 32 client is therefore an r = 64
> adapter in practice.

## 4. Metrics (dependent variables)

| Metric | Definition | Budget / reading |
|---|---|---|
| **TTFT** (ms) | `generate(max_new_tokens=1)`, CUDA-synced, 3 warmups discarded, median of 20. Averaged over 3 prompt lengths (2 / 9 / ~25 tokens). | D11: overhead vs base ≤ 200 ms |
| TTFT p95 (ms) | Worst p95 across the 3 prompts | so a long tail cannot hide behind the median |
| **Decode throughput** (tok/s) | 64 forced new tokens, median of 5 | user-felt speed after the first token |
| **Adapter swap** (s) | `load_adapter` + `set_adapter` on a resident base, median and p95 of 5 | D11: ≤ 2 s |
| Adapter size (MB) | `adapter_model.safetensors` on disk | upload/download cost per round |
| Train s/step | Wall time / optimizer steps (pilot) | feeds GPU-h |
| Peak VRAM (GB) | `torch.cuda.max_memory_allocated`, train and serve | 4 GB card |
| Quality | pass@1 + in-project perplexity, from `experiments/rank-sweep` | the decision metric (not measured here) |

"Base" TTFT is measured on the **same loaded model with the adapter
disabled**. That is what a merged composite costs (H1), and it avoids
comparing across two separate model loads.

## 5. Seeds

| Part | Seeds | Why |
|---|---|---|
| Latency (this pilot) | **1** (seed 0), with 20 timed repeats per prompt | Latency depends on adapter shape, not on the random init. Run-to-run noise is handled by repeats and the median/p95, not by seeds. |
| Quality (main sweep) | **3** (0, 1, 2) | Training noise does depend on the seed. 3 seeds give a mean ± spread per rank. |
| Repro-check | 1 rank point re-run (P4, W3 Thu) | confirms the runner reproduces a result |

## 6. GPU-hours

| Item | Runs | GPU-h / run | Total | Basis |
|---|---|---|---|---|
| Latency pilot (this folder) | 4 | 0.074 (measured) | **0.29** (measured) | `gpu_hours_actual` in `results/rank_bench.json` |
| Quality sweep (`rank-sweep`) | 4 ranks × 3 seeds = 12 | 1.4 | **16.8** | one full round, `docs/compute-budget.md` |
| Repro-check | 1 | 1.4 | 1.4 | |

**Total for the rank decision: about 18.5 GPU-h** (0.29 + 16.8 + 1.4). The
latency side costs under 2% of that. The quality sweep is the expensive part.
At about 1.4 GPU-h per run it takes around 17 hours of wall time on the dev
GPU, so it runs sequentially over the W3 days as the plan schedules.

## 7. Pilot results (measured 2026-11-21, RTX 2050 4 GB, 1.3B dev profile)

Raw data: [`results/summary.csv`](results/summary.csv) (one row per rank) and
[`results/rank_bench.json`](results/rank_bench.json) (full detail).

| r | Trainable params | Adapter MB | Train peak VRAM GB | Serve peak VRAM GB | TTFT base ms | TTFT LoRA (unmerged) ms | **Overhead ms** | LoRA worst p95 ms | Swap median / p95 s |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.57 M | 6.0 | 1.70 | 1.15 | 148.0 | 202.4 | **+54** | 333 | 0.35 / 0.43 |
| 8 | 3.15 M | 12.0 | 1.73 | 1.15 | 107.3 | 160.6 | **+53** | 578 | 0.21 / 0.61 |
| 16 | 6.29 M | 24.0 | 1.77 | 1.16 | 92.3 | 235.2 | **+143** | 365 | 0.46 / 0.50 |
| 32 | 12.58 M | 48.0 | 1.87 | 1.19 | 153.4 | 213.1 | **+60** | 271 | 0.75 / 0.83 |

Training pilot: 30 steps per rank on `web/client-requests`, 2.5–6.1 s/step.
Loss stayed finite at every rank (no divergence). 30 steps is too few to say
anything about quality.

### What the pilot says, hypothesis by hypothesis

- **H3 — supported.** Parameters and file size are exactly linear in r
  (6 → 12 → 24 → 48 MB). Swap time grows with size, from about 0.2–0.35 s at
  low rank to 0.75 s at r = 32, and every p95 is under D11's 2 s.
- **H4 — supported for memory; s/step is noise.** Training peak VRAM rises
  only 1.70 → 1.87 GB (+10%) from r = 4 to r = 32, and serving VRAM rises
  1.15 → 1.19 GB. Seconds per step show no rank trend (2.5 s at r = 16 and
  r = 32). The 3.2 s at r = 4 includes the first CUDA warm-up, and the 6.1 s
  at r = 8 is a laptop slowdown, not rank.
- **H2 — within budget, but the trend is not resolved.** The median unmerged
  overhead is +53 to +143 ms at every rank, under D11's 200 ms. On this laptop,
  though, the noise between model loads (base TTFT ranges 92–153 ms) is as
  large as any rank effect. The r = 16 outlier is driven by an unusually fast
  base measurement. **Tail risk:** the LoRA p95 reaches 271–578 ms, so the
  worst single requests can exceed base + 200 ms.
- **H1 — follows from the design, not measured separately.** The "base" column
  is the same model with the adapter disabled, which is what a merged composite
  costs. Merging (D6) therefore removes the +53–143 ms overhead at every rank.

### Decision this feeds (W3 Fri gate)

On the D6 merged serving path, **rank has no meaningful latency cost**. The
cost it does carry is adapter size, which means upload/download per round
(48 MB at r = 32 vs 24 MB at r = 16) and swap time, which stays under 1 s.
**Pick the rank on the quality curve from `experiments/rank-sweep`.** Use
size/swap only as a tie-breaker, favoring the smaller rank when quality is
within seed noise.

## 8. How to reproduce

```powershell
python -m edge.rank_bench --config experiments/rank-sweep-latency/config.yaml
```

Outputs, in `results/`:

- `summary.csv` — one row per rank (open it in Excel or load it with pandas)
- `rank_bench.json` — full detail: per-prompt TTFT, swap samples, training
  convergence, config, device, and GPU-h actually used

## 9. Known limits

- **1.3B numbers only.** These figures do not carry over to 6.7B (see
  `docs/hardware.md`). Re-run this config when a ≥ 16 GB GPU is available.
- The 30-step pilot measures **cost**, not quality. The loss columns only show
  that training ran and did not diverge.
- Single GPU, Windows, batch size 1. Throughput under concurrent requests is
  not measured.
- **Laptop noise.** Base TTFT moved between 92 and 153 ms across the four
  model loads, and decode tok/s (4.8–10.6) is too noisy to read a rank trend
  from (it is still in the CSV). Next improvement: interleave base and LoRA
  timings per prompt and repeat the whole sweep 3× to put error bars on
  the overhead.
