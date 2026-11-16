# CLASP: Clustered LoRA Adapter Stacking for Secure and Personalized Code Generation

CLASP is an on-premise, hierarchical federated code-generation system for enterprise software
teams. Cloud-based assistants require proprietary source code to leave the organization; CLASP
delivers personalized, project-aware code generation with **no code leaving local
infrastructure**.

Personalization is composed from three weight layers on top of a frozen base model:

```
W_new = W_base + α·ΔW_cluster + β·ΔW_client
```

- **W_base** — frozen base model. The design names DeepSeek-Coder-6.7B; every Phase II run
  uses the 1.3B profile because the team's GPUs have 4 GB.
- **ΔW_cluster** — a project-specific LoRA adapter trained via federated aggregation.
- **ΔW_client** — a per-developer LoRA adapter trained locally.
- **α, β** — coefficients controlling project vs. personal contribution. The composite is
  merged **on the edge** at pull time (D6, amended): the edge pulls the merged cluster adapter
  and merges it with its own client layer. The registry's `/compose` path exists and is
  tested but is not called by the live loop. See
  `docs/decisions/2026-11-14-amended-decisions.md`.

## Repository layout

```
CLASP/
├── contracts/          # shared interface package (the integration seam)
├── services/
│   ├── edge/           # Edge Layer — PEFT training, 4-bit inference, dynamic merger
│   ├── cluster/        # Cluster Layer — FedProx + SVD aggregation (FastAPI; Flower for simulation)
│   ├── registry/       # State Registry — versioning, two-sided promotion, composites (FastAPI)
│   └── evaluation/     # Evaluation & Dashboard — HumanEval/MBPP, Pass@k, React/Recharts
├── security/           # Security library — Opacus DP-SGD + mTLS (imported, not deployed)
├── experiments/        # reproducible ablation configs and results
├── scripts/            # demo_round.py (four-seam round), run_services.py, fresh_machine_check.sh
├── tests/integration/  # CPU-only end-to-end proof of all four seams
└── docs/               # reports, paper draft, figures
```

`security/` is a **library**, imported by the edge and cluster layers; it is not a deployed
service and has no Dockerfile.

## Getting started

Each module is an installable package that depends on the shared `contracts` package.

```bash
# install the shared contracts, then a module + its tests
pip install -e contracts -e "services/registry[test]"
pytest services/registry/tests

# the full demo, one command: registry + cluster + demo UI (:8010) + a registry walkthrough
docker compose --profile demo up --build

# or just the services: registry (:8004) + cluster (:8002)
docker compose up -d --build
```

`docs/devops.md` covers the compose profiles, CI and the
transport security that is on by default (`docs/mtls_setup.md`: mTLS between
services, HTTPS and logins for the pages, roles, signatures, audit), and the
measured non-functional requirements; `scripts/fresh_machine_check.sh` proves a
clean clone installs, tests and runs.

### Run one federated round across all four seams

```bash
docker compose up -d registry cluster
```

```bash
python scripts/demo_round.py --round 1
```

Edge → Cluster → Registry → Edge → Registry, on the real trained adapters,
writing to `experiments/w12-integration/results/`. `docs/integration-sprint.md`
has the full record, including the known gaps. The same loop runs CPU-only over
tiny synthetic adapters in CI:

```bash
python -m pytest tests/integration -q
```

## Modules

| Module | Path | Description |
|--------|------|-------------|
| Edge Layer | `services/edge` | Local LoRA training (PEFT), 4-bit inference, dynamic adapter merger |
| Cluster Layer | `services/cluster` | FedProx + SVD aggregation (FastAPI; Flower for simulation), cluster LoRA redistribution |
| Security | `security` | Library: Opacus DP-SGD and the per-run privacy ledger; mTLS and certificate roles; HMAC signatures; JWT logins; headers, rate limit, audit |
| State Registry | `services/registry` | safetensors versioning, two-sided promotion/rollback, pre-merged composites, retention, run manifests, FastAPI |
| Evaluation & Dashboard | `services/evaluation` | HumanEval/MBPP pipeline, Pass@k scoring, React/Recharts dashboard |

## Tech stack

DeepSeek-Coder (6.7B design, 1.3B profile in Phase II) · HuggingFace PEFT · FedProx + SVD aggregation · Flower (simulation) · Opacus (DP-SGD) · mTLS ·
FastAPI · safetensors · HumanEval/MBPP · React + Recharts · Docker Compose.

## Team

Team 102 — Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham.
Guide: Dr. Swapna T R, Associate Professor, Dept. of CSE.

| Member | Module |
|--------|--------|
| Adithyaa Seyyone D R | Edge Layer |
| Prasanth P M | Cluster Layer |
| Kapilan V | Security |
| Deepak S | State Registry |
| Aditya S | Evaluation & Dashboard |
