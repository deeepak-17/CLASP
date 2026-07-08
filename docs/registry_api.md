# State Registry — API (v0.1, W3)

FastAPI service. Owner: Deepak (P4). Adapters are stored as versioned,
immutable safetensors with a JSON metadata sidecar. W3 scope is **save / load /
list**; two-sided promotion & rollback (D5) land in W5.

Run locally:

```bash
pip install -e contracts -e "services/registry[test]"
CLASP_REGISTRY_DATA=./_data uvicorn registry.app:app --port 8004
# interactive docs at http://localhost:8004/docs
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/healthz` | Liveness; echoes contracts version. |
| GET  | `/adapters` | List adapter names. |
| POST | `/adapters/{name}/versions` | Save a new immutable version (returns metadata). |
| GET  | `/adapters/{name}/versions` | List versions + metadata + active pointer. |
| GET  | `/adapters/{name}/versions/{v}` | Metadata for one version. |
| GET  | `/adapters/{name}/versions/{v}/file` | Download the safetensors blob. |
| GET  | `/adapters/{name}/active` | Metadata for the promoted (active) version. |

### Save

`POST /adapters/{name}/versions` — `multipart/form-data`:

- `file` — the safetensors payload (validated by header structure on write).
- `meta` — a JSON envelope (all fields optional except `kind`):

```json
{
  "kind": "client",                       // "client" | "cluster"
  "cluster_id": "web",
  "round": 3,                             // cluster snapshots
  "seed": 0,
  "aggregation": "svd_exact",             // "svd_exact" | "naive_avg" | null
  "privacy": {"epsilon": 7.5, "delta": 1e-5},
  "hparams": {"rank": 16, "alpha": 0.8, "beta": 0.5,
              "target_modules": ["q_proj","k_proj","v_proj","o_proj"]},
  "source_clients": ["django", "flask", "requests"],
  "set_active": true
}
```

Returns `201` with the written `AdapterMetadata` (includes assigned `version`,
`sha256`, `num_bytes`, `created_at`). Versions increment automatically and are
never overwritten (D9). Invalid names or non-safetensors payloads → `422`.

```bash
curl -F "file=@adapter.safetensors" \
     -F 'meta={"kind":"client","cluster_id":"web","seed":0}' \
     http://localhost:8004/adapters/django/versions
```

### Load

```bash
curl http://localhost:8004/adapters/django/versions/2                # metadata
curl -OJ http://localhost:8004/adapters/django/versions/2/file       # safetensors bytes
```

## metadata.json schema (`AdapterMetadata`)

Written beside each version; immutable. Canonical field definitions live in
`contracts/types.py`.

| Field | Type | Notes |
|---|---|---|
| `ref` | `{name, version, kind, cluster_id}` | adapter identity |
| `hparams` | LoRAHyperParams | rank, lora_alpha, dropout, target_modules, alpha, beta |
| `privacy` | PrivacySpec | epsilon, delta, noise_multiplier, max_grad_norm (D7) |
| `aggregation` | enum \| null | `svd_exact` for cluster adapters; null for client |
| `round` | int \| null | federated round index |
| `seed` | int | reproducibility |
| `sha256` | str | digest of the payload |
| `num_bytes` | int | payload size |
| `source_clients` | list[str] | clients that contributed (cluster adapters) |
| `created_at` | ISO-8601 | write time |
| `contracts_version` | str | `"1.0.0"` |

## Run-manifest schema (`RunManifest`, D9)

One per experiment run, written **before** the run (GPU-hrs estimated first, D8).
Helpers in `registry.manifest`:

```python
from registry.manifest import build_manifest, write_manifest
m = build_manifest("run-001", seed=0, config=resolved_cfg,
                   adapter_versions={"django": 2}, gpu_hours_estimate=3.5)
write_manifest(m, "experiments/<run>/results/manifest.json")
```

| Field | Type | Notes |
|---|---|---|
| `run_id` | str | unique run identifier |
| `seed` | int | run seed |
| `config_hash` | str | sha256 of the resolved config (order-insensitive) |
| `adapter_versions` | dict[str,int] | adapters produced by the run |
| `gpu_hours_estimate` | float | estimated before running (D8 gate) |
| `gpu_hours_actual` | float \| null | filled in on completion |
| `notes` | str | free text |
| `created_at` | ISO-8601 | write time |

## Not yet (later weeks)

- `promote` / `rollback` endpoints driven by `EvalResult` (D5) — W5.
- Composite-adapter storage + promotion pipeline — W8.
- Retention/GC policy + compose profiles — W10.
- mTLS in front of all endpoints (D7) — W10.
