# D3 round 2 + D7 DP-SGD — results

P1 Edge · 2026-10-06 · 1.3B `dev` profile · RTX 2050 4 GB · seed 0
Config [`config.yaml`](config.yaml) · raw manifests under `services/edge/artifacts/`
(`round2_d3/`, `round2_d3_out/`, `dp_smoke/`; adapter weights gitignored).

---

## 1 · What round 2 changes

Exactly one thing relative to round 1 (`experiments/w4w5-g2-g3-round`): each client
LoRA is trained on the **frozen base + 0.5·cluster** instead of the frozen base — the
composition order D3 prescribes. Same corpus, same sqrt budget plan, lr 1e-3, seq 1024,
seed 0, r=16 q/k/v/o.

The cluster layer is not computed by the edge. It is the `svd_exact` aggregate P2's
cluster service built from the six round-1 uploads (seam A), published to the registry
(seam B), and pulled back by `edge.registry_client` (seam C1, sha256-verified):

| cluster | registry version | sha256 | source clients |
|---|---|---|---|
| web | `cluster-web` v2 | `9758be92…5c1279` | flask, requests, werkzeug |
| scientific | `cluster-scientific` v2 | `139a801a…a5fa03f` | numpy, pandas, scikit-learn |

v2 is pulled by number. The registry's `active` tag points at v1 — the `naive_avg`
ablation baseline — because D5 rolled back when the HumanEval guard could not be
scored. D2 forbids serving the naive aggregate, so it is not a layer a client may
train on.

Mechanism (`edge.train_client.stack_frozen_cluster`): the cluster adapter is a second,
frozen PEFT adapter with its scaling multiplied by α, active together with the client
adapter, so the forward pass is `W·x + α·ΔW_cluster·x + ΔW_client·x`. Verified on the
GPU: only the 6,291,456 client parameters are trainable, and with a fresh client
adapter the stacked logits are **bitwise equal** to a single pre-merged α·cluster
composite (`tests/test_d3_dp.py`). No dequantize/requantize of the NF4 base.

## 2 · Training — all six stable

Held-out perplexity on each client's own held-out files (D5 primary metric):

| client | base | round 1 final (bare base) | base + 0.5·cluster (start) | **round 2 final (D3)** | client gain over start | steps | wall |
|---|---|---|---|---|---|---|---|
| web/flask | 2.452 | 2.268 | 2.349 | **2.262** | −0.087 | 73 | 201 s |
| web/requests | 3.463 | 3.269 | 3.335 | **3.257** | −0.078 | 59 | 178 s |
| web/werkzeug | 2.828 | 2.652 | 2.712 | **2.647** | −0.065 | 106 | 295 s |
| sci/numpy | 2.736 | 2.651 | 2.688 | **2.650** | −0.038 | 292 | 831 s |
| sci/pandas | 2.910 | 2.717 | 2.816 | **2.713** | −0.103 | 335 | 942 s |
| sci/scikit-learn | 2.584 | 2.370 | 2.488 | **2.367** | −0.121 | 335 | 1090 s |

All six `stable: true`; peak VRAM 1.77 GB; 59.0 min of training in total (round 1:
55.8 min). The frozen cluster layer alone already delivers 44–64 % of each
client's total gain; the D3 composite beats round 1's client-only result on every
client, by 0.001–0.012.

## 3 · The question round 1 left open — does the cluster layer help?

`edge.round --cluster-adapters` (the same registry adapters the clients trained on),
α grid {0, 0.25, 0.5, 1.0}, grid capped at 30 held-out blocks, then full-split
re-measure at the chosen α, at α=0 (client only) and at α_ref=0.5:

| client | best α (r1 → **r2**) | client only (α=0) | composite (α=0.5) | **cluster layer at α=0.5** — r1 → **r2** |
|---|---|---|---|---|
| web/flask | 0 → **0.5** | 2.311 | 2.264 | +0.000 → **−0.047** |
| web/requests | 0 → **0.5** | 3.323 | 3.259 | +0.002 → **−0.064** |
| web/werkzeug | 0 → **0.5** | 2.695 | 2.648 | +0.012 → **−0.047** |
| sci/numpy | 0 → **0.5** | 2.667 | 2.651 | +0.011 → **−0.016** |
| sci/pandas | 0 → **0.5** | 2.736 | 2.714 | +0.010 → **−0.022** |
| sci/scikit-learn | 0 → **0.5** | 2.388 | 2.367 | +0.014 → **−0.021** |

(negative = the cluster layer lowers perplexity, i.e. helps)

**Round 1: the sweep switched the cluster layer off on all six clients and it was
mildly harmful. Round 2: the sweep keeps it on all six and it helps on all six.** The
round-1 summary named D3 as the prime suspect; trained in the D3 order, the
three-layer composition carries information the client layer does not.

Read it with these limits:

* **Two kinds of improvement, not one.** Against round 1's client-only adapter the
  composite gains only 0.001–0.012. What changed is the *division of labour*: the
  client now learns a residual on top of the cluster, so removing the cluster costs
  0.016–0.064. Whether that also means better personalization per unit of client
  compute is a Phase III question.
* **The cluster layer contains the client's own round-1 update.** Each cluster
  adapter aggregates all three of its members, including the client being evaluated.
  That is inherent to federated averaging; the held-out files were never trained on.
* **α peaks at the training α.** 0.5 is where the clients learned their residual, so
  the sweep is expected to land there. On the capped 30-block grid, α=0.25 and α=1.0 sit between client-only and
  α=0.5 on every client (`round_manifest.json`).
* **requests' held-out split is one block (513 tokens).** Its numbers are the
  noisiest in the table.
* **Composite vs train-time number.** The round scores the composite as one rank-32
  concatenated adapter; the trainer scores the same model as two stacked adapters.
  They agree to ≤ 0.002 ppl. The two are mathematically identical (bitwise at
  initialisation, §1); the residual is presumably floating-point accumulation order
  after training. Not investigated further.

**NFR (D11).** The evaluation round took **31.8 min** against the 30-minute budget —
round 1 was 27.8 min with a 2-point α grid; this one swept 4. With round 1's grid it
fits; as configured here it misses by 1.8 min. Training is excluded, as in round 1.

**D5.** Decisions remain `PROVISIONAL_PROMOTE`: the noise band is still the 0.0
placeholder and the HumanEval guard is still unscored (Docker unavailable on this
machine). Nothing here is authority to promote.

## 4 · D7 — DP-SGD through P3's security library

`edge.train_client --dp` on web/client-requests, D3 on cluster-web v2 at α=0.5, using
P3's `security.DPConfig` + `security.make_private` from the unmerged `security`
branch (put on `PYTHONPATH`; nothing of P3's was copied into the edge).

| | |
|---|---|
| target | ε ≤ 8, δ = 1e-5 (D7) |
| **ε spent (Opacus RDP accountant, steps actually taken)** | **7.992** ✅ within budget |
| ε, P3's `security.EpsilonTracker`, same σ/q/steps | 11.086 |
| noise multiplier σ (auto-calibrated by `make_private`) | 0.7016 |
| clip norm / logical batch / sample rate | 1.0 / 8 / 0.0769 |
| logical steps (planned = taken) / empty Poisson batches | 26 / 0 |
| seq len / gradient checkpointing | 512 / off |
| peak VRAM / wall | 2.84 GB / 275 s |
| held-out ppl: base → base+0.5·cluster → final | 3.462 → 3.366 → 3.362 |

ε reaches the seam-A envelope: `edge.wire.upload_payload(privacy=…)` carries
`{"epsilon": 7.991991, "delta": 1e-05, "noise_multiplier": 0.701599,
"max_grad_norm": 1.0}` as a `contracts.PrivacySpec`.

Utility, stated plainly: under DP the client adds **−0.004** ppl over its frozen
starting point, against −0.078 without DP — at ε=8 and 26 steps the noise absorbs
most of the client's learning. The run is flagged `stable: false` because the noisy
loss slope (+0.003/step) trips the divergence tolerance; held-out perplexity did not
get worse. This is one client on a one-block held-out split: a smoke test of the
integration, not a privacy-utility curve (that is Phase III W7).

### Re-run on P3's rewritten API (#19, `security` @ `5924cfc`)

The `security` branch was rewritten after the run above: `DPConfig` lost the
run-shape fields, `EpsilonTracker` became `PrivacyAccountant(config, sample_rate)`
and `create_client_ssl_context` became `client_ssl_context`. The edge now calls the
new API. `security.make_private(..., epochs=)`, the path meant to calibrate σ to the
budget, crashes on Opacus 1.6 (`engine.py` reads `accountant.noise_multiplier`, which
the default PRV accountant does not have), so the edge calibrates σ with Opacus's
`get_noise_multiplier` over the engine's own accountant, sample rate and epochs (what
`make_private_with_epsilon` computes) and hands it to `make_private` via `DPConfig`.
Same command, same client, same seed (manifest not committed; weights not saved):

| | |
|---|---|
| **ε spent (engine's PRV accountant, the one σ is calibrated for)** | **7.992** ✅ within budget |
| ε, P3's `security.PrivacyAccountant` (RDP), same σ/q/steps | 9.399 |
| σ (calibrated) / sample rate / logical steps | 0.6519 / 0.0769 / 26 |
| peak VRAM / wall | 2.84 GB / 234 s |
| held-out ppl: base → base+0.5·cluster → final | 3.46 → 3.37 → 3.36 (client −0.004) |

The uploaded ε is now the engine's PRV figure. RDP is a looser bound and reads over 8
for the same run, so it is recorded beside it, not used for the budget. Findings 1–3
below describe the old API and no longer apply; finding 4 still does.

### Findings for P3 (not fixed here — P3's module)

1. **`security.make_private` crashes in its default mode.** `DPConfig` defaults to
   `grad_sample_mode="ghost"`; Opacus 1.6's `PrivacyEngine.make_private` then returns
   four objects (module, optimizer, criterion, loader) and `make_private` unpacks
   three. The edge passes `grad_sample_mode="hooks"`, which returns three.
2. **`EpsilonTracker` over-states ε by 1.3–3×** against Opacus's RDP accountant (the
   one `make_private` actually builds): 11.09 vs 7.99 here; 10.98 vs 8.79, 13.90 vs
   6.32, 2.99 vs 1.01 on the settings spot-checked. Conservative, but it would report
   D7's budget as blown when it is not.
3. **`make_private_lora` would unfreeze the D3 cluster layer.** It sets
   `requires_grad=True` on every parameter named `lora_*`, which includes a frozen
   second adapter. The edge therefore freezes explicitly and calls `make_private`.
4. **Opacus per-sample hooks do not survive gradient checkpointing** on this model
   (`grad_sample` never populated). DP runs disable checkpointing; on 4 GB that forces
   seq 512 (seq 1024 without checkpointing spills to shared memory and stalls).
5. **Privacy unit.** Opacus gives record-level DP — here one packed block. D7 asks for
   client-level DP, which is a property of the federated protocol, not of one
   client's training loop. Recorded, not claimed.

## 5 · Reproduce

```bash
# cluster adapters, from the running registry (python scripts/run_services.py --background)
python -c "from edge.registry_client import RegistryClient as R; r=R(); \
  r.materialize('cluster-web','artifacts/round2_cluster/cluster-web',version=2); \
  r.materialize('cluster-scientific','artifacts/round2_cluster/cluster-scientific',version=2)"

# D3 training, per client
python -m edge.train_client --client web/client-flask --corpus-root ../../datasets/materialized \
  --budget-plan budget_plan.json --lr 1e-3 --seed 0 \
  --cluster-adapter artifacts/round2_cluster/cluster-web --alpha 0.5 --out-dir artifacts/round2_d3

# round-2 evaluation
python -m edge.round --adapters artifacts/round2_d3 --corpus-root ../../datasets/materialized --round 2 \
  --alpha-grid 0 0.25 0.5 1.0 --alpha-ref 0.5 --grid-max-blocks 30 --seed 0 \
  --cluster-adapters web=artifacts/round2_cluster/cluster-web scientific=artifacts/round2_cluster/cluster-scientific \
  --out artifacts/round2_d3_out

# DP smoke (needs P3's security branch importable)
python -m edge.train_client --client web/client-requests --corpus-root ../../datasets/materialized \
  --seq-len 512 --lr 1e-3 --seed 0 --cluster-adapter artifacts/round2_cluster/cluster-web --alpha 0.5 \
  --dp --dp-epsilon 8 --dp-delta 1e-5 --dp-max-grad-norm 1.0 --dp-batch-size 8 --out-dir artifacts/dp_smoke
```

Note on round 1's reproduce line: `experiments/w4w5-g2-g3-round/SUMMARY.md` gives
`--alpha-grid 0 0.25 0.5 1.0`, but the manifest that produced its table
(`artifacts/round1_out/round_manifest.json`, 27.8 min) records `alpha_grid [0, 0.5]`
and `grid_max_blocks 30`. The full-split α=0 and α=0.5 numbers compared above are
measured the same way in both rounds.
