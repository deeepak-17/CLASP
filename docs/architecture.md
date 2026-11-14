# CLASP — Architecture v2 (modules, data flow, owners)

> Phase II W1 deliverable (P4, DevOps & Orchestration Lead). Plan of record:
> `MASTER_PLAN.md` (D1–D12). This document is the module/data-flow reference the
> panel and the team work against; update it via PR when contracts change.

## 1. What CLASP is

A federated, privacy-preserving code-generation system. Per-developer
personalization is composed from three weight layers on top of a frozen base
model:

```
Wnew = Wbase + alpha * dWcluster + beta * dWclient
```

- **Wbase** — frozen base model, 4-bit (NF4). The design names DeepSeek-Coder-6.7B;
  every Phase II run uses the 1.3B profile because the team's GPUs have 4 GB.
- **dWcluster** — LoRA adapter aggregated across a cluster of clients (federated).
- **dWclient** — per-client LoRA, trained on the frozen merged `Wbase + alpha·dWcluster` (D3).
- **alpha, beta** — composition coefficients; the composite is merged **on the
  edge** at pull time (D6, amended by U1): the edge pulls the merged cluster
  adapter and merges it with its own client layer, and inference loads a single
  adapter. No runtime dual-adapter stacking. The registry's `/compose` path and
  the `composite` block of `/promote` exist and are tested but are not called by
  the live loop. See `docs/decisions/2026-11-14-amended-decisions.md`.

## 2. Modules and owners

| Module | Path | Owner | Deploys as | Responsibility |
|---|---|---|---|---|
| Edge | `services/edge/` | Adithyaa (P1) | container | PEFT training, 4-bit inference, composite merge (on the edge) |
| Cluster | `services/cluster/` | Prasanth (P2) | container | FastAPI service; FedProx; SVD aggregation (D2); Flower in simulation only; re-clustering (D4) |
| Security | `security/` | Kapilan (P3) | **library only** | Opacus DP-SGD (D7), mTLS — imported by edge & cluster, never a service |
| **Registry** | `services/registry/` | **Deepak (P4)** | **stateful container** | **versioned safetensors, two-sided promotion/rollback (D5), composite storage, run manifests** |
| Evaluation | `services/evaluation/` | Aditya (P5) | library + scripts | library + scripts + static dashboard: in-project eval (D5 primary), HumanEval/MBPP guard |
| Contracts | `contracts/` | shared (seam) | installed pkg | the integration types; **v1.1.0** |

Architectural invariants: monorepo (not multi-repo); Security is a library with
**no Dockerfile, ever**; `contracts/` is the only cross-module import path;
registry data persists via a named docker volume and safetensors are never
overwritten in place.

## 3. Data flow (one federated round)

```
                 (per client, local)                    (per cluster)                 (offline, at promotion)
  ┌────────┐  AdapterUpload   ┌─────────┐  ClusterSnapshot  ┌──────────┐  EvalResult   ┌────────────┐
  │  Edge  │ ───────────────► │ Cluster │ ────────────────► │ Registry │ ◄──────────── │ Evaluation │
  │ (P1)   │  dWclient +      │  (P2)   │  dWcluster (SVD-  │  (P4)    │  in-project + │   (P5)     │
  │        │  hparams,ε,seed  │         │  aggregated)      │          │  guard metrics│            │
  └────────┘                  └─────────┘                   └──────────┘               └────────────┘
       ▲                             │
       └─────────────────────────────┘
         promoted cluster adapter (Cluster → Edge); the edge merges it
         with its own client layer. Evaluation → Registry: results go to
         `/promote`, which makes the candidate active or rejects it (D5).
```

1. **Edge → Cluster** (`AdapterUpload`): each client trains a LoRA on its repo
   partition and uploads factors + `num_train_samples` (for FedProx weighting),
   LoRA hyperparams, ε spent, and seed. mTLS on this channel from W7 (D7).
2. **Cluster aggregation** (D2): server reconstructs each `dWi = Bi·Ai` per target
   module, exact-averages (FedProx-weighted), re-factorizes to rank r via
   truncated SVD, streamed module-by-module. Naive A/B averaging is kept only as
   an ablation baseline.
3. **Cluster → Registry** (`ClusterSnapshot`): the aggregated cluster adapter is
   versioned and stored with its aggregation method, participating clients, ε,
   and seed.
4. **Evaluation → Registry** (`EvalResult`): the evaluator returns primary
   in-project metrics (edit similarity, exact match, perplexity on the client's
   held-out files) plus the line-completion guard numbers (HumanEval/MBPP and
   PY150 are reported only).
5. **Promotion (D5, two-sided)**: a published version is a candidate;
   promotion makes it live; a rejected candidate is never served. The cluster's
   aggregate is its candidate and is published to the registry inactive; the
   edges keep pulling the active adapter. The registry promotes the candidate
   iff in-project improves beyond the measured noise band over the active
   version (the bare base model when nothing is active yet) **and** the
   line-completion guard's edit similarity drops by at most 0.02 (D5).
   PROMOTE moves `active` to the candidate; otherwise the decision is REJECT
   and `active` stays where it was. The cluster is then told the same
   (`candidate/promote` or `candidate/reject`, by the registry's sha256), and
   only a promoted aggregate is served to the edges. Never gated on Pass@k
   alone. Moving `active` back to an earlier live version is a separate,
   recorded operator restore (`/restore`), never automatic.

### 3a. Status of the four seams (Integration Sprint, Panel Review 2)

All four hops above are now implemented and run as one loop on branch
`integration/panel`. **`docs/integration-sprint.md` is the record**: what
changed, the PEFT ⇄ cluster key translation, registry materialization, the
evaluation metric, the HumanEval guard decision, fallback status, the exact
commands, and the debt log.

| Seam | Call | Status |
|---|---|---|
| Edge → Cluster | `POST /uploads` (:8002) | live over HTTP with the real 192-tensor adapters |
| Cluster → Registry | `POST /adapters/{id}/publish` → `POST /adapters/{name}/versions` (:8004) | live |
| Cluster → Edge | `GET /clusters/{id}/adapters/active` (:8002) | live; the edge pulls the merged cluster adapter and merges its own client layer |
| Evaluation → Registry | `POST /adapters/{name}/promote` | live; the demo panel posts the result of `scripts/evaluate_candidate.py`; the promote decision uses the line-completion guard (D5); its noise measurement is pending, see `services/evaluation/results/line_guard/noise.json` |

## 4. Registry internals (P4 — this repo's primary module)

Storage layout on the persistent volume (`CLASP_REGISTRY_DATA`, default
`/data/registry`):

```
adapters/<name>/
  v1/  adapter.safetensors   metadata.json
  v2/  adapter.safetensors   metadata.json
  active                     # text file: the promoted version int
```

- **Immutable versions** — a new adapter version is a new directory; `save`
  refuses to clobber an existing version (D9). Every payload gets a sha256.
- **Opaque payloads** — the registry treats adapter tensors as bytes, so the
  service needs no torch/numpy/safetensors at runtime (stdlib-only storage layer,
  with a structural safetensors-header validity check on write).
- **`metadata.json`** — an `AdapterMetadata` record: ref, LoRA hyperparams,
  privacy spec (ε, δ), aggregation method (cluster adapters), round, seed, sha256,
  size, source clients, timestamp.
- **Run manifests** — `RunManifest` (run_id, seed, config hash, adapter versions,
  GPU-hrs estimate/actual) written per experiment for reproducibility (D9).

The endpoint contract is the route list in the module docstring of
`services/registry/src/registry/app.py` (14 routes).

## 5. Process & reproducibility (D9)

- Monorepo, per-member feature branches + PRs, no direct pushes to `main`.
- CI (`.github/workflows/ci.yml`): ruff over the whole repo + pytest across the
  6-package matrix (contracts, security, edge, cluster, registry, evaluation) —
  a required check.
- Contracts v1.0 frozen (W2); changes need a semver bump + all-hands sign-off.
- Registry has a named docker volume; safetensors never overwritten in place.
- Every experiment: config in `experiments/`, seed + manifest recorded, GPU-hrs
  estimated before running (D8).

## 6. Gates (D12)

- **G1 (W7)** — Edge↔Cluster live over mTLS.
- **G2 (W8)** — 3-layer composition, 6 clients / 2 clusters.
- **G3 (W9)** — full end-to-end federated round (all-hands; ugly is fine).

Slip policy: if a gate slips ≥ 1 week, cut in order — dynamic clustering →
MBPP → dashboard polish. Never cut: E2E round, DP, in-project eval, report.
