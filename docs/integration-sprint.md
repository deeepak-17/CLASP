# Integration Sprint — the four seams, wired

**Branch:** `integration/panel` (cut from `origin/main`; do **not** merge to `main` this sprint)
**Scope:** wiring only. No retraining, no new ML science, no DP/mTLS deployment, no
dynamic re-clustering, no dashboard.

Three services that had never been run against each other now run as one loop:

```
Edge ──A──▶ Cluster ──B──▶ Registry ──C1──▶ Edge ──C2──▶ Registry
```

| Seam | Call | Who serves it |
|---|---|---|
| **A** | `POST /uploads` | `services/cluster` — one client's trained LoRA |
| **B** | `POST /adapters/{cluster_id}/publish` → registry `POST /adapters/{name}/versions` | cluster → registry |
| **C1** | `GET /adapters/{name}/active` + `GET .../versions/{v}/file` | registry → edge |
| **C2** | `POST /adapters/{name}/promote` | edge → registry, D5 decides |

---

## 1. Run it

```bash
docker compose up -d registry cluster
```

If the docker daemon is not available, the same two services run on the host:

```bash
python scripts/run_services.py --background
```

> **Verified how.** Everything below was measured with `scripts/run_services.py`
> — the same two ASGI apps, same ports — because the docker daemon would not
> start on the demo machine (see §7). `docker-compose.yml` and the Dockerfiles
> were updated for this sprint (the cluster image's `CMD` finally resolves: it
> points at `cluster.server:app`, which did not exist before), and a cluster
> healthcheck was added, but **the images have not been built or run here**.
> Do a `docker compose build registry cluster` before relying on the compose
> path in front of the panel.

Then, one command for the whole round:

```bash
python scripts/demo_round.py --round 1
```

Panel commands from the run-of-show, unchanged:

```bash
curl -s localhost:8004/adapters/cluster-web/versions | python -m json.tool
```

```bash
curl -s localhost:8004/adapters/cluster-web/promotions
```

CPU-only proof, no services and no GPU needed:

```bash
python -m pytest tests/integration -q
```

Everything the round produces lands in `experiments/w12-integration/results/`:
`round1_manifest.json`, the materialized adapters under `pulled/`, and each
version's `materialize_manifest.json`.

### Environment

The trained weights and the materialized corpora are gitignored, so a fresh
clone has the code but not the data. Point the round at wherever they live:

```bash
python scripts/demo_round.py --round 1 --adapters /path/to/round1 --corpus-root /path/to/materialized
```

Without a GPU, `--skip-eval` still exercises A, B, C1 and C2; the in-project
metric is then recorded as unmeasured rather than guessed (see §6).

---

## 2. What changed

### `services/cluster` (P2's module — extended, not rewritten)

| File | Change |
|---|---|
| `aggregation.py` | `_svd` retry on LAPACK non-convergence; `aggregate_layerwise`; `layerwise_exact_average_error`; `aggregate_svd_lowrank` |
| `schemas/messages.py` | `AdapterUpload.cluster_id`; fp32 wire contract enforced; `ClusterAdapterBroadcast.source_clients` |
| `server.py` | HTTP surface reworked for **two** clusters; `/download`; `/publish` (seam B); `/adapters/{id}/active`, `/adapters/{id}/manifest`; `retain_uploads` |
| `pyproject.toml` | `safetensors`, `httpx` promoted to runtime deps |

`aggregate_svd`, `aggregate_naive` and `exact_average_delta` are **unmodified**.

### `services/edge` (new modules)

| File | Purpose |
|---|---|
| `edge/wire.py` | B3 — bidirectional PEFT ⇄ cluster key map, fp32 contract, upload envelope, safetensors (de)serialization |
| `edge/registry_client.py` | B4 — C1/C2 client, sha256 verification, PEFT-directory materialization |
| `edge/completion_eval.py` | B5 — greedy next-line generation against a real model |
| `edge/promote.py` | B6 — EvalResult assembly, HumanEval guard resolution, seam C2 |

### `services/evaluation` (P5's module — the missing metric)

`evaluation/completion.py`: example selection, Levenshtein edit similarity,
exact match, `InProjectMetrics` construction. Pure Python, no torch, so it runs
in CPU-only CI.

### Repo level

`scripts/demo_round.py`, `scripts/run_services.py`,
`tests/integration/test_four_seams.py`, a CI job that runs it,
`experiments/w12-integration/`.

---

## 3. Seam A and the adapter format (B3)

### The two key conventions

```
cluster : layers.3.q_proj.lora_A.weight
PEFT    : base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
```

`edge/wire.py` maps between them in both directions with one regex over the
layer index — never a hard-coded layer or module list:

```python
wire.to_peft_keys(state_dict)      # anything accepted -> PEFT
wire.to_cluster_keys(state_dict)   # anything accepted -> cluster
wire.parse_key(key)                # -> (layer, module, part), raises on anything else
```

Cluster currently emits PEFT keys directly (`LoRAAdapter.to_peft_state_dict`),
so the map is not strictly on the critical path today. It is still applied
unconditionally on every inbound payload, which means the sprint's **F1
fallback is already the running configuration**: if Cluster ever reverts to its
short internal convention, the Edge path keeps working with no change.

### Verified against the real adapters

The six trained client adapters are **192 tensors, 24 layers × q/k/v/o, r=16,
`lora_alpha=16` (so PEFT scaling is exactly 1.0), fp32**. Read off disk, not
assumed.

### The dtype contract

FP32, both directions, **enforced**: `wire.as_fp32` on the way out and
`AdapterUpload`'s validator on the way in (a non-fp32 tensor is a 422, not a
silent cast). The aggregation promotes to float64 internally and casts back, so
fp32 is the only dtype that ever crosses a seam.

### Two real bugs the real adapters exposed

Neither could be reached by the 32-dimensional synthetic demo.

**1. LAPACK `gesdd` non-convergence.** `np.linalg.svd` raised
`LinAlgError: SVD did not converge` on the averaged ΔW of
`layers.12.self_attn.o_proj` for the three web clients — a real 2048×2048
matrix, finite, norm 0.69. `aggregation._svd` now retries on the transpose:
for `A = U S Vᵀ`, `Aᵀ = V S Uᵀ`, so swapping the factors back recovers exactly
the same decomposition through a different LAPACK entry path. Verified to give
identical singular values on the failing matrix.

**2. The reference path does not fit the machine.** `aggregate_svd` allocates
one dense float64 accumulator per (layer, module) up front: at real width that
is 24 × 4 × 2048² × 8 B = **3.2 GB** before the first SVD, on a machine with
8 GB total and ~1 GB free. `aggregate_layerwise` runs the *same* aggregator one
layer at a time (134 MB peak), stitching the results back together — bit-for-bit
identical, since both aggregators already treat every (layer, module)
independently.

### The performance problem, and the exact fast path

Even layer-by-layer, an exact 2048² float64 SVD measured **~12 s**, so 96
modules is ~15 min per cluster against a 30-minute round NFR. `aggregate_svd`
was not modified. Instead `aggregate_svd_lowrank` evaluates *the same*
pipeline without ever forming the dense average:

```
dW = Σ cᵢ Bᵢ Aᵢ = Bcat @ Acat          Bcat: out×kr,  Acat: kr×in
Bcat = Q_B R_B,  Acatᵀ = Q_A R_A       (thin QR)
dW = Q_B (R_B R_Aᵀ) Q_Aᵀ               SVD the kr×kr core, lift by Q_B / Q_A
```

`Q_B` and `Q_A` have orthonormal columns, so this is the exact SVD of `dW`.
Truncating to r and splitting `√S` symmetrically is the same construction
`truncated_svd_refactor` performs.

Measured on the real web cluster, whole 24-layer adapters:

| path | wall clock | mean rel. error vs exact weighted average | max |
|---|---|---|---|
| reference `aggregate_svd`, layerwise | **864.6 s** | 0.171237 | 0.227009 |
| `aggregate_svd_lowrank` | **2.47 s** | **0.171237** | **0.227009** |

Identical to six decimals at full width, on real weights. The reference path
stays available via `{"exact_lowrank": false}` and remains the oracle;
`tests/integration` asserts the two agree.

### Compared against `artifacts/round1_out/cluster-web` — the Thursday gate

| measured against the exact weighted-average ΔW | mean | max |
|---|---|---|
| cluster (`aggregate_svd_lowrank`) | 0.171237 | 0.227009 |
| edge (`artifacts/round1_out/cluster-web`) | 0.172241 | 0.229754 |
| edge's own recorded manifest | 0.172242 | 0.229755 |

Two things follow. First, reproducing edge's numbers from its own artifact to
six decimals confirms the client set, the FedAvg weights `[80, 52, 170]` and
the pipeline were reconstructed correctly. Second, **the cluster's error is
smaller than edge's, module for module** — which is what must happen: an exact
truncated SVD is the optimal rank-16 approximation, and edge uses randomized
`torch.svd_lowrank(niter=8)`.

The two adapters' ΔW differ by 5.2 % mean / 17.4 % max. **That is not a bug.**
At the r=16 cut the singular spectrum is nearly flat — `s₁₇/s₁₆` runs 0.85–0.99
across modules — so the top-16 subspace is numerically near-degenerate and two
near-optimal rank-16 approximations can differ while both sit ~0.17 from the
exact average. The measurement that matters is distance to the exact average,
and on that the cluster path wins everywhere.

### Also settled: Defect 7

The D2 ablation now runs on the real 2048-wide adapters instead of 32-dim
noise. Naive per-factor averaging vs the exact weighted average, web cluster:
**0.3287 mean** against the SVD path's **0.1712** — the cross-term bias, measured.

---

## 4. Seam B — cluster → registry

`POST /adapters/{cluster_id}/publish` is the only place Cluster talks to another
service. It sends exactly the bytes `GET /adapters/{cluster_id}/download`
serves, plus a metadata envelope read off the aggregate itself: aggregation
method (`svd_exact` / `naive_avg`), source clients, round, and the LoRA
hyperparameters. Nothing is typed by hand. The registry computes and stores its
own sha256.

`lora_alpha` is published pinned to `r`, so PEFT's scaling is exactly 1.0 and
the stored tensors mean ΔW itself — the convention `edge.aggregate` already
uses.

### A third bug the real path exposed: the payload was not byte-reproducible

`safetensors` serializes its header — `__metadata__` included — from a Rust
hash map, so the same tensors with the same metadata come out in a different
key order call to call. Measured directly: **three `save` calls in one process
produced two distinct sha256 digests.** The aggregate itself is bit-identical
across processes (verified on the real web cluster: same digest over the
tensors from four separate runs), so this was purely a serialization artefact —
but it meant the digest the registry records against an adapter changed run to
run for an unchanged adapter.

That matters twice over. D9's whole theme is reproducibility, and a digest that
moves on its own cannot support "same inputs, same version". And it makes
version deduplication by digest impossible.

`edge.wire.canonicalize_safetensors` (mirrored as `cluster.server._canonical_safetensors`)
rewrites the header as sorted, compact JSON. It touches the header only —
`data_offsets` are relative to the buffer that *follows* it, so every offset
stays valid. Asserted by `test_serialize_is_byte_reproducible` and
`test_published_payload_is_byte_reproducible`, and the two copies are asserted
identical by `tests/integration`.

The registry's sha256 was already doing its real job — transport integrity on
the C1 download — and still is. It is now also a content fingerprint.

---

## 5. Seam C1 — registry → edge materialization

The registry serves tensor bytes with no config beside them (Defect 4).
`RegistryClient.materialize` rebuilds a loadable PEFT directory and records
where every field came from.

**Sources, in order of authority:**

1. **Registry metadata** — `hparams.rank → r`, `hparams.lora_alpha → lora_alpha`,
   `hparams.dropout → lora_dropout`, `hparams.target_modules → target_modules`.
2. **The payload's own `__metadata__`** — the producer embeds the full
   `adapter_config.json` inside the safetensors header when it serializes
   (`cluster.server._peft_bytes` / `wire.serialize`). This is what carries
   `base_model_name_or_path`, which contracts v1.0 `LoRAHyperParams` has no
   field for.
3. **`PEFT_STRUCTURAL_DEFAULTS`** for the structural flags
   `edge.merge.STRUCTURAL_FIELDS` compares between adapters.

Fields present in both (1) and (2) are **cross-checked**; a disagreement raises
`MetadataConflict` rather than picking a winner silently. If a payload has no
embedded config and no `base_model` is passed, materialization **refuses**
rather than guessing. Every field's provenance is written to
`materialize_manifest.json`.

**Checksum.** Every download is hashed and compared against the registry's
recorded sha256 *before anything touches the filesystem*; a mismatch raises
`ChecksumMismatch` and leaves no directory behind. Directories are staged and
renamed, so an interrupted pull never leaves a half-written adapter that the
merge path would happily load. Both behaviours are asserted in
`tests/integration`.

Verified: the materialized adapter loads through the **existing**
`edge.merge.load_adapter`, passes `validate_compatibility` against
`CONTRACT_HYPERPARAMS` with `scaling = 1.0`, and composes with a real client
adapter at a merge self-check of ~2.8e-07 relative.

---

## 6. Evaluation behaviour (B5)

D5 promotes on `edit_similarity`, which nothing measured before this sprint.

**What is measured.** Next-line completion on each client's held-out files
(`held_out/`, a sibling of `repo/`, never trained on):

- `edit_similarity` — `1 − levenshtein(pred, gold) / max(|pred|, |gold|)`,
  averaged. The RepoBench / CodeXGLUE convention.
- `exact_match` — fraction where the stripped prediction equals the stripped
  gold line.
- `perplexity` — from the **existing** `edge.train_client.evaluate`, not
  recomputed, so the repo keeps one perplexity implementation.

**How.** Greedy decoding (`do_sample=False`), completion cut at the first
newline, prompt truncated from the left to 512 tokens. Example selection is
deterministic — sorted files, fixed stride, no RNG — and a SHA-256 fingerprint
over `(file, line_no)` goes into the results, so two runs are only ever compared
if they scored the same problems.

**What is excluded and why.** Blank lines, comment-only lines and sub-4-character
lines: a model scores near 1.0 on those regardless of whether it learned the
project, which inflates the metric without measuring personalization.

**Not measured.** Both fields compare `line.strip()`, so indentation
correctness is not scored. Stated rather than hidden.

**When it cannot run.** With `--skip-eval` or no GPU, the metric is *not*
faked. contracts v1.0 has no "unmeasured" encoding, so `in_project` is sent
with `n_examples: 0` and structural zeros, and the record carries
`in_project_measured: false`. D5 then rolls back — correctly, for the stated
reason.

---

## 7. The HumanEval guard — decision and status

**Status: UNAVAILABLE. Every promotion decision in the live round is therefore
PROVISIONAL, and the D5 rule returns ROLLBACK.** This is sprint fallback **F3**,
taken deliberately.

The sprint's three options, and what happened to each:

1. *Score a frozen subset inside a Linux container.* Attempted, and it is the
   right answer — but not reachable here. `python -c "import resource"` on this
   machine gives `ModuleNotFoundError: No module named 'resource'` (Windows), so
   evalplus cannot score on the host; and the Docker daemon would not start
   (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file
   specified` — `com.docker.service` is Stopped and starting it needs
   elevation). The only WSL distribution present is Docker's own minimal
   `docker-desktop`. Not available in the sprint window.
2. *Generation on the host, scoring in the container.* Same blocker. The
   generation half is already done and committed —
   `services/edge/eval_out/samples.jsonl` holds 20 real greedy completions from
   the base model over the frozen first-20 HumanEval subset — so this option is
   one working container away, not one rewrite away.
3. *Declare the guard unavailable and let the rule roll back.* **Taken.**

`edge.promote.resolve_guard` reads real anchor files and returns
`available=False` with a reason when there are none. Nothing fabricates a
pass@1. When anchors do exist, pass them and the guard becomes live:

```bash
python scripts/demo_round.py --round 1 \
  --candidate-anchor eval_out/candidate/anchor.json \
  --baseline-anchor  eval_out/baseline/anchor.json
```

`tests/integration` asserts all four D5 outcomes with real anchor files:
PROMOTE when both halves hold, ROLLBACK on a metric regression, ROLLBACK on a
pass@1 regression, and ROLLBACK on a missing guard.

### To close it

Score `services/edge/eval_out/samples.jsonl` inside any Linux container:

```bash
docker run --rm -v "$PWD/services/edge/eval_out:/w" python:3.11-slim \
  bash -c "pip install -q evalplus==0.3.1 && cd /w && python -m evalplus.evaluate --dataset humaneval --samples samples.jsonl"
```

Then feed the result through `edge.humaneval_baseline.compute_pass_at_1` to
mint an `anchor.json`, and repeat with the composite's completions for the
candidate side.

---

## 8. What the round actually does

Per cluster (`web`, `scientific`):

1. Three real client adapters upload over HTTP. The FedAvg sample weight is the
   client's block count, **recomputed by packing its corpus**, not copied from
   an old manifest.
2. The same uploads are aggregated twice — `naive` (D2's ablation baseline) and
   `svd` (exact) — using `retain_uploads` so the two are comparable without
   making every client re-send 25 MB.
3. Both are published to the registry as v1 and v2. **This is why there are two
   versions:** D5's first promote call cannot succeed otherwise — rollback with
   no previous version raises, and the API returns 422.
4. Both are pulled back, sha256-verified, and composed with the representative
   client's adapter into `base + α·cluster + β·client`.
5. Each composite is scored on the client's held-out files; v2 goes to
   `POST /promote` with v1's metrics as the baseline.

Counters for a full round: **6 uploads, 4 aggregations, 4 registry snapshots,
4 composites, 2 promotion decisions**.

### Measured, 2026-09-06 (RTX 2050, 1.701 GB peak VRAM, seed 0)

**14.33 min** wall clock against the 30-minute NFR. Full numbers in
`experiments/w12-integration/RESULTS.md`; the highlights:

| cluster | rep. client | v1 naive edit_sim | v2 svd edit_sim | decision |
|---|---|---|---|---|
| web | client-werkzeug | 0.5337 | **0.5470** | ROLLBACK (guard) |
| scientific | client-scikit-learn | 0.4837 | **0.4861** | ROLLBACK (guard) |

The reason string is worth reading aloud:

```
rolled back: edit_similarity +0.0133 clears noise band 0.0000;
             HumanEval guard metrics missing on candidate or baseline
```

The in-project half of the two-sided rule passed and the rule refused anyway,
because the guard half could not be evaluated. That is D5 doing its job.

**The chain reproduces the edge-only pipeline exactly.** The v2 composite is
built from a cluster adapter that went edge → cluster → registry → edge, and
its full-split perplexity at α=0.5 matches what `artifacts/round1_out/round_manifest.json`
recorded from edge's purely local run: werkzeug **2.665 vs 2.665**,
scikit-learn **2.385 vs 2.385**.

---

## 9. Debt and gaps — read this before the panel asks

- **D3 composition order is not closed.** The six clients were trained against
  the frozen base, not against frozen `base + α·cluster`. Unchanged by this
  sprint; it needs a second training pass.
- **Any PROMOTE is provisional** while the HumanEval guard is unscored (§7).
- **The noise band is a placeholder at 0.0.** "Improved" therefore means
  "improved by any amount". `evaluation.completion.noise_band` can compute a
  band from repeats, but with greedy decoding repeats are identical, so that
  measures decode noise and *not* the training-seed band D5 wants — the function
  says so in its own return value.
- **DP and mTLS are a library, not a deployment.** Nothing on this channel is
  differentially private or mutually authenticated.
- **No dashboard.** Results are JSON under `experiments/`.
- **Clustering is static** — two hand-assigned clusters. Dynamic re-clustering is W10.
- **The web cluster on disk is `{flask, requests, werkzeug}`**; D1 says
  `{django, flask, requests}`. Werkzeug is an unrecorded substitution.
- **Cluster state is in-memory and single-process.** Restarting the cluster
  service drops buffered uploads and resets round counters. Durable state is
  the registry's job; the cluster is deliberately not a database.
- **`/publish` has no retry.** If the registry is down mid-round, the call
  returns 502 and the round stops with a clear error rather than half-committing.
- **The representative-client choice matters.** Each cluster's D5 decision is
  measured on one client's held-out files (the largest split), not averaged
  across the cluster. Recorded in the manifest as `representative_clients`.
- **Seam B publishes from the cluster process.** The orchestrator never relays
  the bytes — but note `RegistryClient.save_version` exists as the documented
  fallback if that hop has to move.

## 10. Fallback status

| | | |
|---|---|---|
| **F1** key round-trip fails | not needed | the explicit key map is implemented and applied on every inbound payload anyway |
| **F2** cluster HTTP service slips | not needed | seam A and seam B are real HTTP; nothing is in-process |
| **F3** completion metric slips | **partially taken** | the metric is implemented and measured; the *guard* is what is unavailable, so decisions are ROLLBACK and marked provisional |
| **F4** integration fails outright | not needed | all four seams run |

---

## 11. Acceptance

| | Item | | Evidence |
|---|---|---|---|
| **B1** | Integration branch | **PASS** | `integration/panel`, cut from `origin/main` — which already carried P2's cluster work merged with edge and registry, so the trial merge the plan expected was unnecessary. Not merged to `main`. |
| **B2** | Cluster HTTP service | **PASS** | `/healthz`, `/uploads`, `/aggregate`, `/adapters/{id}/download`, `/adapters/{id}/active`, `/adapters/{id}/manifest`, `/adapters/{id}/publish` on :8002. Six real uploads land; aggregation runs on real adapters; the manifest carries reconstruction error. `aggregate_svd`, `aggregate_naive`, `exact_average_delta` unchanged. |
| **B3** | Real adapter format path | **PASS** | 192 tensors × 24 layers accepted; bidirectional key map, layer-generic; fp32 enforced; PEFT → cluster → PEFT round trip on a real adapter; result loads through `edge.merge.load_adapter` and passes `validate_compatibility` against the frozen contract. Compared against `artifacts/round1_out/cluster-web` — see §3. |
| **B4** | Edge registry client | **PASS** | `/active` + `/file`, sha256 verified before anything is written, PEFT directory materialized with recorded provenance, corruption path tested. |
| **B5** | Completion metric | **PASS** | Edit similarity, exact match and perplexity measured on real held-out files; `InProjectMetrics` filled from real numbers. |
| **B6** | Promotion path | **PASS**, guard unavailable | Two versions published before the first promote, D5 not bypassed, decision recorded in the audit trail with a readable reason. PROMOTE and all three ROLLBACK reasons are asserted in `tests/integration` against the real rule. In the live round every decision is ROLLBACK because the HumanEval guard cannot be scored here — fallback F3, `decision_is_authoritative: false`. See §7. |
| **B7** | One-command round | **PASS** | `python scripts/demo_round.py --round 1` — 14.3 min, six uploads, four aggregations, four snapshots, four composites, two decisions, all values from the run. |
| **B8** | CPU-only CI integration | **PASS** | `tests/integration/test_four_seams.py`, 22 tests, ~4 s, no GPU / model / network / services, deterministic. Wired into CI as its own job. |

### Not verified here

**`docker compose up -d registry cluster`.** The daemon would not start on this
machine, so every measurement above came from `scripts/run_services.py` — the
same two ASGI apps on the same ports. The compose file and Dockerfiles were
updated (the cluster image's `CMD` now resolves to an app that exists, which it
did not before) but the images have not been built. Build them before the panel.
