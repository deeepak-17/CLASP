# CLASP: Clustered LoRA Adapter Stacking for Secure and Personalized Code Generation

CLASP is an on-premise, hierarchical federated code-generation system for enterprise software
teams. Cloud-based assistants require proprietary source code to leave the organization; CLASP
delivers personalized, project-aware code generation with **no code leaving local
infrastructure**.

Personalization is composed from three weight layers on top of a frozen base model:

```
W_new = W_base + α·ΔW_cluster + β·ΔW_client
```

- **W_base** — frozen DeepSeek-Coder-6.7B.
- **ΔW_cluster** — a project-specific LoRA adapter trained via federated aggregation.
- **ΔW_client** — a per-developer LoRA adapter trained locally.
- **α, β** — coefficients controlling project vs. personal contribution.

## Repository layout

```
CLASP/
├── contracts/          # shared interface package (the integration seam)
├── services/
│   ├── edge/           # Edge Layer — PEFT training, 4-bit inference, dynamic merger
│   ├── cluster/        # Cluster Layer — Flower + FedProx aggregation (FastAPI)
│   ├── registry/       # State Registry — safetensors versioning, Pass@k rollback (FastAPI)
│   └── evaluation/     # Evaluation & Dashboard — HumanEval/MBPP, Pass@k, React/Recharts
├── security/           # Security library — Opacus DP-SGD + mTLS (imported, not deployed)
├── experiments/        # reproducible ablation configs and results
└── docs/               # reports, paper draft, figures
```

`security/` is a **library**, imported by the edge and cluster layers; it is not a deployed
service and has no Dockerfile.

## Getting started

Each module is an installable package that depends on the shared `contracts` package.

```bash
# install the shared contracts, then a module + its tests
pip install -e contracts -e services/registry
pytest services/registry/tests

# or bring up the full stack
docker compose up --build
```

## Modules

| Module | Path | Description |
|--------|------|-------------|
| Edge Layer | `services/edge` | Local LoRA training (PEFT), 4-bit inference, dynamic adapter merger |
| Cluster Layer | `services/cluster` | Flower orchestration, FedProx aggregation, cluster LoRA redistribution |
| Security | `security` | Opacus DP-SGD on gradients, mTLS between edge and cluster (library) |
| State Registry | `services/registry` | safetensors versioning, metadata, Pass@k-based rollback, FastAPI |
| Evaluation & Dashboard | `services/evaluation` | HumanEval/MBPP pipeline, Pass@k scoring, React/Recharts dashboard |

## Tech stack

DeepSeek-Coder-6.7B · HuggingFace PEFT · Flower · FedProx · Opacus (DP-SGD) · mTLS ·
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
