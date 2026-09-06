# w12-integration — results

One round, four seams, real trained adapters. Produced by

```bash
python scripts/run_services.py --background     # or: docker compose up -d registry cluster
python scripts/demo_round.py --round 1
```

Full record: `results/round1_manifest.json` (gitignored — heavy, and regenerated
by the command above). Method and caveats: `docs/integration-sprint.md`.

**Run:** 2026-09-06 16:12:53 → 16:27:13 UTC · NVIDIA GeForce RTX 2050 ·
peak VRAM **1.701 GB** of 4 GB · deepseek-coder-1.3b-base (NF4) · seed 0.

---

## 1. The round

| | |
|---|---|
| client uploads (seam A) | 6 — 192 tensors each, 24 layers, fp32 |
| cluster aggregations | 4 — naive + svd per cluster, from one upload set |
| registry snapshots (seam B) | 4 — v1 naive, v2 svd, per cluster |
| adapters pulled back (seam C1) | 4 — all sha256-verified |
| composites merged (D6) | 4 — rank 32, α=0.5, β=1.0 |
| promotion decisions (seam C2) | 2 |
| **wall clock** | **14.33 min** against the 30 min NFR (D11) — **met** |

---

## 2. Aggregation — D2's ablation, on real adapters

Relative Frobenius error against the **exact weighted-average ΔW**, per
(layer, module), 96 modules per cluster. Lower is better.

| cluster | naive (per-factor avg) | svd-exact | ratio | agg wall |
|---|---|---|---|---|
| web | 0.3287 mean / 0.3909 max | **0.1712** / 0.2270 | 1.92× | 20.8 s / 6.0 s |
| scientific | 0.5978 mean / 0.6421 max | **0.2777** / 0.4095 | 2.15× | 20.2 s / 5.8 s |

This is the SVD-vs-naive result that previously ran on 32-dimensional random
tensors (sprint Defect 7), now measured on the 2048-wide trained adapters.

**Agreement with the edge's own W4/W5 artifact** — `artifacts/round1_out/cluster-web`,
measured the same way:

| | mean | max |
|---|---|---|
| cluster service (`aggregate_svd_lowrank`) | **0.171237** | **0.227009** |
| reference `aggregate_svd`, untouched | 0.171237 | 0.227009 |
| edge `artifacts/round1_out/cluster-web` | 0.172241 | 0.229754 |
| edge's own recorded manifest | 0.172242 | 0.229755 |

The cluster's exact truncated SVD is *better* than edge's randomized
`svd_lowrank` everywhere, as it must be. The reference path took **864.6 s** for
one cluster; the low-rank path took **2.47 s** for the identical numbers.

---

## 3. In-project completion (D5's primary metric)

Next-line completion, 60 deterministic examples from the representative client's
held-out files, greedy decoding. Perplexity is the **full** held-out split.

| cluster | rep. client | version | edit_sim | exact_match | perplexity | n_tokens |
|---|---|---|---|---|---|---|
| web | client-werkzeug | v1 naive | 0.5337 | 0.2667 | 2.661 | 30 279 |
| web | client-werkzeug | **v2 svd** | **0.5470** | **0.2833** | 2.665 | 30 279 |
| scientific | client-scikit-learn | v1 naive | 0.4837 | 0.2333 | 2.378 | 248 968 |
| scientific | client-scikit-learn | **v2 svd** | **0.4861** | 0.2333 | 2.385 | 248 968 |

Both clusters scored the same problem set for both versions
(`examples_sha256` identical), so the deltas are comparable.

**The SVD aggregate beats the naive ablation on the metric D5 actually reads**,
in both clusters: +0.0133 edit similarity on web, +0.0023 on scientific.

**Perplexity disagrees with edit similarity here** — the naive composite is
marginally *lower*-perplexity in both clusters (2.661 vs 2.665; 2.378 vs 2.385)
while completing worse. Reported rather than smoothed over; it is one reason D5
gates on completion quality and not on perplexity.

### Cross-check against the earlier edge-only round

The composite is built from a cluster adapter that travelled
edge → cluster → registry → edge. Its full-split perplexity at α=0.5 matches
what `artifacts/round1_out/round_manifest.json` recorded for the same clients
from edge's purely local pipeline:

| client | W4/W5 local `alpha_ref_ppl` | this round, v2 via the registry |
|---|---|---|
| web/client-werkzeug | 2.665 | **2.665** |
| scientific/client-scikit-learn | 2.385 | **2.385** |

Four HTTP hops, a sha256-verified round trip through immutable storage, and a
different SVD implementation — and the composite behaves identically.

---

## 4. Promotion (D5, seam C2)

Both clusters: **ROLLBACK to v1**, recorded in the registry's audit trail.

```
rolled back: edit_similarity +0.0133 clears noise band 0.0000;
             HumanEval guard metrics missing on candidate or baseline
```

Read the reason string literally: the in-project half of the two-sided rule
**passed**, and the rule still refused, because the HumanEval half could not be
evaluated. That is D5 working as designed, and it is the sprint's F3 position —
`decision_is_authoritative: false` on every decision here.

`--candidate-anchor` / `--baseline-anchor` turn the guard on as soon as real
anchors exist. `tests/integration` asserts PROMOTE and all three ROLLBACK
reasons against the real registry rule.

---

## 5. What these numbers are not

- **Not a D3 result.** The six clients were trained against the frozen base, not
  against frozen `base + α·cluster`. Unchanged by this sprint.
- **Not a promotion.** Any PROMOTE is provisional while the guard is unscored.
- **Not gated on a real noise band.** 0.0 is a placeholder; "improved" currently
  means "improved by any amount".
- **Not a cluster-wide evaluation.** One representative client per cluster
  (the largest held-out split), not an average over its three members.
- **Not an indentation-aware metric.** Edit similarity and exact match compare
  stripped lines.
