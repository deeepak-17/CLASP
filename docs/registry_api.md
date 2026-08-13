# State Registry — API (v0.2, W4)

FastAPI service. Owner: Deepak (P4). Adapters are stored as versioned,
immutable safetensors with a JSON metadata sidecar. Save/load/list landed W3;
two-sided promotion & rollback (D5) landed W4 (ahead of the original W5 plan).
Runnable end-to-end walkthrough: [`services/registry/demo/save_promote_rollback_demo.sh`](../services/registry/demo/save_promote_rollback_demo.sh).

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
| POST | `/adapters/{name}/promote` | Apply the D5 two-sided rule to the active version — confirms (`promote`) or reverts (`rollback`). |
| GET  | `/adapters/{name}/promotions` | Promotion/rollback audit trail (append-only, never rewritten). |

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

### Promote / rollback (D5)

`POST /adapters/{name}/promote` — JSON body, applies the two-sided rule to
whatever version is **currently active** (saves auto-activate on write, D9;
this is the checkpoint that confirms or reverts that choice once P5's
evaluation is in):

```json
{
  "eval": {
    "adapter": {"name": "django", "version": 2, "kind": "client"},
    "in_project": {"edit_similarity": 0.80, "exact_match": 0.5, "perplexity": 3.0, "n_examples": 20},
    "guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.30}}],
    "baseline_in_project": {"edit_similarity": 0.70, "exact_match": 0.5, "perplexity": 3.2, "n_examples": 20},
    "baseline_noise_band": 0.02
  },
  "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.30}}]
}
```

Rule (`registry.promotion.decide`, pure function — no filesystem access):
promote iff in-project edit similarity clears `baseline_noise_band` **and**
HumanEval pass@1 hasn't dropped more than `GUARD_PASS_AT_1_TOLERANCE` (0.02)
vs `baseline_guard`. Otherwise the registry rolls the `active` pointer back to
`previous_version`. Never gates on pass@k alone (D5).

Responses:

| Status | Meaning |
|---|---|
| `200` | Decision applied (`action: "promote"` or `"rollback"`); body is the `PromotionDecision`, also appended to the audit log. |
| `409` | `eval.adapter.version` isn't the currently active version — promotion only evaluates the current active. |
| `422` | Malformed payload, `eval.adapter.name` doesn't match the path, or a rollback was indicated but there's no `previous_version` to fall back to. |

```bash
curl -X POST http://localhost:8004/adapters/django/promote \
     -H 'Content-Type: application/json' \
     -d '{"eval": {...}, "baseline_guard": [...]}'

curl http://localhost:8004/adapters/django/promotions   # audit trail
```

See [`services/registry/demo/save_promote_rollback_demo.sh`](../services/registry/demo/save_promote_rollback_demo.sh)
for a full runnable save → save → promote → save → rollback → lineage cycle.

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

- Composite-adapter storage + promotion pipeline (D6) — W8.
- Retention/GC policy + compose profiles — W10.
- mTLS in front of all endpoints (D7) — W10.
