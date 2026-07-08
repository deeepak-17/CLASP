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

- **Wbase** — frozen DeepSeek-Coder-6.7B (NF4 4-bit).
- **dWcluster** — LoRA adapter aggregated across a cluster of clients (federated).
- **dWclient** — per-client LoRA, trained on the frozen merged `Wbase + alpha·dWcluster` (D3).
- **alpha, beta** — composition coefficients; composition happens **offline at
  promotion time**. The registry stores one pre-merged composite per developer;
  inference loads a single adapter (D6). No runtime dual-adapter stacking.

## 2. Modules and owners

| Module | Path | Owner | Deploys as | Responsibility |
|---|---|---|---|---|
| Edge | `services/edge/` | Adithyaa (P1) | container | PEFT training, 4-bit inference, composite merge |
| Cluster | `services/cluster/` | Prasanth (P2) | container | Flower + FedProx, SVD aggregation (D2), re-clustering (D4) |
| Security | `security/` | Kapilan (P3) | **library only** | Opacus DP-SGD (D7), mTLS — imported by edge & cluster, never a service |
| **Registry** | `services/registry/` | **Deepak (P4)** | **stateful container** | **versioned safetensors, two-sided promotion/rollback (D5), composite storage, run manifests** |
| Evaluation | `services/evaluation/` | Aditya (P5) | container | in-project eval (D5 primary), HumanEval/MBPP guard, dashboard |
| Contracts | `contracts/` | shared (seam) | installed pkg | the integration types; **frozen v1.0** |

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
       ▲  composite adapter (base⊕α·cluster⊕β·client)            │ promote / rollback
       └───────────────────────────────────────────────────────┘  repoints `active` tag (D5)
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
4. **Registry → Evaluation** (`EvalResult`): the evaluator returns primary
   in-project metrics (edit similarity, exact match, perplexity on the client's
   held-out files) plus HumanEval/MBPP guard numbers.
5. **Promotion (D5, two-sided)**: the registry promotes a version iff in-project
   improves beyond the measured noise band **and** HumanEval pass@1 drop ≤ 2 pts;
   otherwise it repoints `active` to the previous version (rollback). Never gated
   on Pass@k alone.

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
- **Auth** — endpoints go behind mTLS in W10 (D7); currently open on the internal
  compose network.

See `docs/registry_api.md` for the endpoint contract.

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
