# W4 / W5 — P1 Edge · slide-ready summary

**W4 = G2** 3-layer composition, 6 clients / 2 clusters · **W5 = G3** full E2E round + profiling
Date 2026-08-14 · 1.3B `dev` profile · RTX 2050 4 GB · seed 0
Config [`config.yaml`](config.yaml) · raw JSON in `results/` · code in `services/edge/src/edge/`

---

## 1 · Status board

| Gate | Item | Status |
|---|---|---|
| **G2** | 6 client adapters trained on real partitions | ✅ all 6, all stable |
| **G2** | SVD aggregation → **real** cluster adapters (D2) | ✅ 2 clusters |
| **G2** | `base + α·cluster + β·client` E2E, α/β config-driven (D6) | ✅ |
| **G2** | First personalization delta | ✅ **−0.174 ppl mean** |
| **G3** | Full E2E round, orchestrated + manifested | ✅ **6 clients, 2 clusters, 27.8 min** |
| **G3** | TTFT / swap vs NFR (D11) | ✅ both pass |
| **D3** | Client trained on frozen `base + α·cluster` | ❌ **not done — and it matters, see §5** |
| **D5** | Two-sided promotion rule | ⚠️ half — HumanEval guard unavailable |

---

## 2 · Client training — all six, all stable

lr `1e-3`, seq 1024, LoRA r=16 on q/k/v/o, sqrt step budget, 1200 blocks total.

| client | files | blocks | budget | epochs | held-out | base ppl | tuned ppl | **Δppl** | grad norm |
|---|---|---|---|---|---|---|---|---|---|
| web/flask | 21 | 80 | 73 | 0.91 | 11 | 2.452 | 2.268 | **−0.184** | 0.27 |
| web/requests | 16 | 52 | 59 | 1.14 | 1 | 3.463 | 3.269 | **−0.194** | 0.29 |
| web/werkzeug | 44 | 170 | 106 | 0.62 | 30 | 2.828 | 2.652 | **−0.176** | 0.32 |
| sci/numpy | 199 | 1286 | 292 | 0.23 | 106 | 2.736 | 2.651 | **−0.085** | 1.24 |
| sci/pandas | 239 | 1687 | 335 | 0.20 | 159 | 2.910 | 2.717 | **−0.193** | 0.42 |
| sci/scikit-learn | 222 | 1696 | 335 | 0.20 | 244 | 2.584 | 2.370 | **−0.214** | 0.71 |

**Every client improved on its own held-out files. Mean Δppl −0.174.**
Train wall 55.8 min · peak VRAM 1.75 GB of 4 GB · ~2.5 s/step.

---

## 3 · Aggregation (D2) — why not naive averaging

Relative Frobenius error against the **exact** weighted average, measured on the real adapters:

| cluster | clients | SVD rank-16 | naive per-factor avg |
|---|---|---|---|
| web | flask, requests, werkzeug | **0.172** | 1.399 |
| scientific | numpy, pandas, scikit-learn | **0.279** | 1.449 |

Naive averaging scores **> 1.0**, i.e. it lands *further from the true average than a zero
adapter would*. It is not "slightly biased" — it is worse than not aggregating. Cause: averaging
A and B separately produces cross terms Bᵢ·Aⱼ pairing one client's output projection with
another's input projection, which share no basis.

Second read: scientific loses 0.279 to rank-16 truncation vs web's 0.172. Three diverse clients
do not compress into rank 16 as cleanly as three related ones — an argument for the Phase III
rank sweep, and for letting the cluster rank exceed the client rank.

---

## 4 · Round + NFRs

**Full round — 6 clients, 2 clusters** (`results/round_manifest.json`, **27.8 min**):

| client | best α | base ppl | client-only | composite | Δppl | cluster @ α=0.5 | decision |
|---|---|---|---|---|---|---|---|
| sci/numpy | **0** | 2.736 | 2.653 | 2.653 | −0.083 | +0.011 | PROVISIONAL_PROMOTE |
| sci/pandas | **0** | 2.910 | 2.718 | 2.718 | −0.192 | +0.010 | PROVISIONAL_PROMOTE |
| sci/scikit-learn | **0** | 2.584 | 2.371 | 2.371 | −0.213 | +0.014 | PROVISIONAL_PROMOTE |
| web/flask | **0** | 2.452 | 2.269 | 2.269 | −0.183 | +0.000 | PROVISIONAL_PROMOTE |
| web/requests | **0** | 3.463 | 3.260 | 3.260 | −0.203 | +0.002 | PROVISIONAL_PROMOTE |
| web/werkzeug | **0** | 2.828 | 2.653 | 2.653 | −0.175 | +0.012 | PROVISIONAL_PROMOTE |

**6 of 6 clients selected α=0** — the sweep switches the cluster layer off in every case. The
last column is the cluster layer's effect at a forced α=0.5, measured on the full split so it
cannot be zero by construction: **positive on all six, i.e. it makes perplexity worse.** The
cluster layer is not merely inert, it is mildly harmful. See §5.

(An earlier web-only run is kept at `results/round_manifest_web.json`. It showed requests
preferring α=0.5 by 0.003 — a difference measured on its single held-out block, i.e. noise.
The full-split figures above supersede it.)

| NFR (D11) | budget | measured | verdict |
|---|---|---|---|
| TTFT overhead vs base | ≤ 200 ms | **+66.4 ms** worst | ✅ PASS |
| adapter swap | ≤ 2 s | **0.501 s** median (p95 1.387 s) | ✅ PASS |
| one federated round | ≤ 30 min | **27.8 min** excluding training; training alone is a further 55.8 min | ⚠️ see note |

**NFR caveat, and it is a close call:** D11's "one round ≤ 30 min" does not say whether client
training counts. Aggregate + compose + evaluate came in at **27.8 min — inside 30, but only by
2.2 min**. Add training and it is 83.6 min, well over. Read exclusively it passes; read
inclusively it fails. Pin the definition before anyone quotes the number. Hardware here is a
2050, not the reference GPU.

---

## 5 · The finding that matters

**The α sweep switches the cluster layer off — on all six clients.** Every client selected α=0.
Forced to α=0.5 and measured on the full held-out split, the cluster layer makes perplexity
*worse* on all six (+0.000 to +0.014). It is not inert; it is mildly harmful.

This is not a bug in the merge. E3.4 proved the composition is exact (α=0,β=1 reproduces the
client adapter bitwise). It is a **consequence of the D3 deviation**:

> D3: "ΔW_client is trained on the **frozen merged** (W_base + α·ΔW_cluster), not on W_base alone."

All six clients trained on the bare base, because no cluster adapter existed at training time.
Each client therefore already learned, on its own, the general direction the cluster average
encodes. Adding α·cluster on top re-applies what the client has: redundant at best, interference
at worst. **The cluster layer has nothing left to contribute.**

That makes D3 the load-bearing decision for the project's central claim — cross-client transfer
without sharing code — not the footnote it has been so far. **Round 2 with D3 composition order
is the experiment that tests it**, and it is now unblocked because real cluster adapters exist.

---

## 6 · Blocked / needs a decision

| # | Item | Owner | Why it blocks |
|---|---|---|---|
| 1 | **D3 round 2** — retrain clients on frozen `base + α·cluster` | P1 | Until then the cluster layer cannot be shown to help; the headline claim is untested |
| 2 | **D5 noise band** — 3× repeated baseline evals | P5 | `--noise-band` is 0.0, so "improved" = "improved at all". Not a real gate; every promotion is provisional |
| 3 | **HumanEval regression guard** | P5 / P1 | evalplus imports Unix-only `resource`; cannot score on Windows. D5's second side is unevaluable → no authoritative promote |
| 4 | **D8 amendment** — sqrt step budgeting | team | Flat 200 steps/client gives a 33× epoch spread; sqrt cuts it to 5.7× at identical GPU cost, but 3 clients then exceed 200 |
| 5 | **D1 web cluster** — werkzeug substituted for django; `_extra/` holds 3 more repos | P5 | "Two semantically real clusters" is a paper claim; the substitution is unrecorded |
| 6 | **ε becomes per-client** under variable step budgets | P3 | D7 caps ε ≤ 8 per run; with unequal steps it is max-over-clients, binding on scikit-learn |
| 7 | **Straggler timeout** assumes uniform client wall-time | P2 | Budgets now range 59–335 blocks (151 s – 1198 s) |

---

## 7 · Next

1. **Round 2 with D3 order** — the one experiment that decides whether the cluster layer earns
   its place. Everything needed now exists.
2. Feed real cluster adapters to P2 as a test oracle for the actual aggregation service.
3. Re-run the α grid after D3; if the cluster still contributes nothing, that is a genuine
   negative result about the architecture and needs to reach the paper honestly.

---

## Reproduce

```bash
cd services/edge
for c in web/client-flask web/client-requests web/client-werkzeug \
         scientific/client-numpy scientific/client-pandas scientific/client-scikit-learn; do
  python -m edge.train_client --client $c --budget-plan budget_plan.json --lr 1e-3 --out-dir artifacts/round1
done
python -m edge.aggregate --clients artifacts/round1/client-{flask,requests,werkzeug}/adapter \
  --weights 80 52 170 --cluster-id web --compare-naive --out artifacts/clusters/web
python -m edge.round --adapters artifacts/round1 --alpha-grid 0 0.25 0.5 1.0 --out artifacts/round1_out
pytest tests/ -q     # 84 passing
```
