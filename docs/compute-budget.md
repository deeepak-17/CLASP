# Compute budget — Phase III ablations (draft)

**Owner:** Deepak (P4). Every ablation is a config under `experiments/` that
`python -m registry.runs check` validates in CI. No config runs without a seed, a
per-run GPU-hours estimate, the D8 caps (or a written justification for raising
one), and a sweep total inside its own budget.

## Basis of the estimates

Measured on the development machine (RTX 2050, 4 GB; `docs/hardware.md`) with
DeepSeek-Coder-1.3B, rank 16, q/k/v/o, seq len 1024:

| Unit | Measured | Source |
|---|---|---|
| 6 clients train (≤ 200 steps each) | 55.8 min | `experiments/w4w5-g2-g3-round/SUMMARY.md` |
| aggregate + compose + evaluate, 6 clients / 2 clusters | 27.8 min | same |
| **one full federated round** | **83.6 min → 1.4 GPU-h** | sum of the above |
| compose + in-project eval of one composite | ≈ 7 min → 0.15 GPU-h with headroom | 27.8 min ÷ 4 composites |
| four-seam round, no training | 14.0–17.9 min | `experiments/w12-integration/RESULTS.md` |

Actual GPU-hours (wall × GPUs) are recorded on every run's manifest. Replace an
estimate with the measured figure after the first run of each sweep.

## The sweeps

| Experiment | Sweep | Runs | GPU-h / run | Total | Notes |
|---|---|---|---|---|---|
| `rank-sweep` | rank ∈ {4, 8, 16, 32} × 3 seeds | 12 | 1.4 | **16.8** | rank 32 exceeds D8 — justified override, one point only |
| `alpha-beta-sweep` | α ∈ {0, .25, .5, .75, 1} × β ∈ {.5, 1} | 10 | 0.15 | **1.5** | no retraining; greedy eval, so one seed |
| `clustering-ablation` | static vs dynamic (D4) × 3 seeds, 5 rounds | 6 | 7.0 | **42.0** | the most expensive; candidate to cut to 2 seeds |
| `epsilon-sweep` | ε ∈ {1, 2, 4, 8} × 3 seeds | 12 | 1.8 | **21.6** | assumes ×1.3 DP-SGD overhead — re-measure |
| `fedprox-mu-sweep` | μ ∈ {0, .001, .01, .1} | 4 | 1.4 | **5.6** | only if budget remains |
| **Total** | | **44** | | **87.5** | |

**Order** (cheapest informative first): rank → α/β → clustering → ε → μ. Rank is
fixed by the first sweep and every later sweep runs at it.

**Cuts if the GPU budget is short** (D12 order, applied to experiments):
fedprox-μ (−5.6) → clustering seeds 3→2 (−14) → ε seeds 3→2 (−7.2). That brings
the total to about 60 GPU-h without dropping any primary result.

## If the hardware changes

The figures assume the 1.3B model on the development GPU. Moving to 6.7B, or to a
different reference GPU, changes every per-run figure. Re-time one round, update
`gpu_hours_estimate` in each config (CI re-checks the budgets), and record the
new basis above. D8 allows raising the caps on better hardware, never lowering
them mid-phase.

## Running and resuming

```bash
python -m registry.runs plan experiments/rank-sweep/config.yaml      # runs + GPU-h, no GPU used
python -m registry.runs run  experiments/rank-sweep/config.yaml -- \
    <round command> --rank {rank} --seed {seed} --out {outputs_dir}
python -m registry.runs status rank-sweep                             # what is done
python -m registry.runs freeze rank-sweep --tag paper-v1              # results final
python -m registry.repro pack --registry-url http://localhost:8004    # one tarball for the artifact
```

An interrupted sweep resumes by re-running the same `run` command: completed points
are skipped.
