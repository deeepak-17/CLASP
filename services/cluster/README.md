# Cluster Layer — Flower + FedProx aggregation, FastAPI

**Owner:** Prasanth (P2)

Depends on the shared `contracts` package; keep cross-module interaction
interface-driven.

```bash
pip install -e contracts -e services/cluster
pytest services/cluster/tests
```

## HTTP surface (:8002)

```bash
uvicorn cluster.server:app --host 0.0.0.0 --port 8002
```

| Endpoint | Purpose |
|---|---|
| `GET  /healthz` | liveness + per-cluster round bookkeeping |
| `POST /uploads` | **seam A** — one client's trained LoRA (`AdapterUpload`) |
| `POST /aggregate` | run the existing D2 aggregation over one cluster's buffer |
| `GET  /adapters/{cluster_id}/active` | the aggregate, PEFT keys + `adapter_config` |
| `GET  /adapters/{cluster_id}/download` | the same thing as safetensors bytes |
| `GET  /adapters/{cluster_id}/manifest` | last aggregation's reconstruction error |
| `POST /adapters/{cluster_id}/publish` | **seam B** — push it to the State Registry |

State is in-memory and keyed by `cluster_id` (D1: `web`, `scientific`), so the
two clusters buffer and aggregate independently. Durable versioning is the
registry's job.

`aggregate_svd`, `aggregate_naive` and `exact_average_delta` are unchanged by
the integration sprint. `aggregate_svd_lowrank` is the same mathematics
evaluated through the low-rank factors — identical results, seconds instead of
minutes at real model width. See `docs/integration-sprint.md` §3.
