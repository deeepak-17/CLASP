# State Registry — versioned adapters, two-sided promotion, composites

**Owner:** Deepak (P4)

The registry is CLASP's only stateful service. It stores every LoRA adapter as an
immutable, sha256-addressed safetensors version, decides promotion with the D5
two-sided rule, stores the pre-merged composite each developer is served (D6),
and keeps an append-only audit trail of every decision. It never needs torch:
tensors are opaque bytes, except when building a composite (numpy).

```bash
pip install -e contracts -e "services/registry[test]"
pytest services/registry/tests -q --cov=registry          # 163 tests, ~94 % coverage
CLASP_REGISTRY_DATA=./_data python -m registry.serve      # :8004, OpenAPI at /docs
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/healthz` | liveness + contracts version |
| GET | `/adapters` | adapter names |
| POST | `/adapters/{name}/versions` | save a new immutable version (multipart: `file` + `meta` JSON) |
| GET | `/adapters/{name}/versions` | every version's metadata + the active one |
| GET | `/adapters/{name}/versions/{v}` | one version's metadata |
| GET | `/adapters/{name}/versions/{v}/file` | the safetensors payload |
| GET | `/adapters/{name}/active` | metadata of the active version |
| GET | `/adapters/{name}/lineage` | versions with parents, sha256, ε and the decisions on each |
| POST | `/adapters/{name}/compose` | store `α·cluster + β·client` as one COMPOSITE version (D6) |
| POST | `/adapters/{name}/promote` | D5 rule on the active version; optional build-on-promote composite |
| POST | `/adapters/{name}/restore` | operator restore / rollback drill (audited) |
| GET | `/adapters/{name}/promotions` | the audit trail |
| POST | `/adapters/{name}/gc` | retention for one adapter (dry run by default) |
| POST | `/gc` | retention across every adapter |

Errors map to status codes in one place: unknown adapter/version → 404, invalid
name or payload → 422, version race or kind conflict → 409, corrupt on-disk record
→ a clean 500 with the reason.

### Save

```bash
curl -F file=@adapter_model.safetensors \
     -F 'meta={"kind":"cluster","aggregation":"svd_exact","round":1,"cluster_id":"web",
               "source_clients":["flask","requests","werkzeug"],
               "hparams":{"rank":16,"lora_alpha":16},"privacy":{"epsilon":3.0}}' \
     localhost:8004/adapters/cluster-web/versions
```

`meta` follows `contracts.AdapterMetadata`: `kind` (`client` | `cluster` |
`composite`), `hparams`, `privacy` (ε logged per round, D7), `aggregation`,
`round`, `seed`, `cluster_id`, `source_clients`, `set_active` (default true), and
for composites `composed_from`. A save auto-activates; `promote` then confirms or
reverts it. All versions under one name share one kind.

### Promote (D5) and build-on-promote (D6)

```json
{"eval": <contracts.EvalResult>,
 "baseline_guard": [{"benchmark": "HumanEval", "pass_at_k": {"1": 0.30}}],
 "composite": {"name": "composite-flask", "cluster": "cluster-web",
               "client": "flask", "alpha": 0.5, "beta": 1.0}}
```

PROMOTE iff in-project edit similarity improves beyond the measured noise band
**and** HumanEval pass@1 drops by at most 2 points; otherwise ROLLBACK repoints
`active` to the previous version. Missing guard metrics mean ROLLBACK — the rule
refuses to promote what it cannot check. With `composite`, a PROMOTE also stores
the composite built from the parts' active versions. The composite is built in
memory before anything is written, so a bad composite request rejects the whole
call and records no decision.

A ROLLBACK changes what the edge serves, not just the part's pointer: every
composite whose active version was built from the rolled-back version moves to
the composite of the restored version. An existing one is reused; otherwise it is
rebuilt. Each move is recorded in that composite's audit trail and listed under
`composites` in the response. If a replacement cannot be built, the call is
refused with 409 and nothing changes. Composites pinned to other versions are
left alone.

### Composites

`POST /adapters/composite-flask/compose` with
`{"cluster": "cluster-web", "client": {"name": "flask", "version": 2}, "alpha": 0.5, "beta": 1.0}`.
The merge is exact, not an approximation: factors are concatenated along the rank
axis with each part's coefficient and scaling folded into B, so the stored adapter's
`B·A` equals `α·ΔW_cluster + β·ΔW_client` (rank r_c + r_l, scaling 1.0). It matches
`edge.merge.compose` tensor for tensor (`tests/integration/test_registry_composite.py`).
Both parts must adapt the same (layer, module) set; otherwise the composite would
mix ranks under one declared `r` and PEFT could not load it, so it is refused
(the edge applies the same rule). A part with coefficient 0 is dropped and exempt.
Provenance (exact part versions, α, β, base model) is stored on the version. The
base model is read from the parts' embedded `base_model_name_or_path`; parts on
different bases, or a `base_model` that contradicts them, are refused. A
composite's ε is the basic-composition bound ε_cluster + ε_client, and `null` if
either part lacks DP. Re-composing identical inputs reuses the existing version.

### Restore and retention

`POST /adapters/{name}/restore` `{"reason": "...", "to_version"?: n}` repoints
`active` (default: the previous version) and records it in the audit trail. Like
a D5 ROLLBACK, it moves the composites built from the current version too. This
is the rollback drill and the D11 "restore ≤ 10 s" path.

`POST /adapters/{name}/gc` `{"keep_last"?: n, "dry_run"?: bool}` keeps the newest
N versions plus everything that was ever live or is still needed — the active
version, anything promoted, anything a rollback made active, and any version a
composite names — and deletes the rest. Dry run by default; `keep_last` defaults to
`CLASP_REGISTRY_KEEP_LAST` (else 5). Version numbers are never reused.

## Operations

| Variable | Default | Meaning |
|---|---|---|
| `CLASP_REGISTRY_DATA` | `/data/registry` | storage root (compose volume `registry-data`) |
| `CLASP_REGISTRY_PORT` / `_HOST` | `8004` / `0.0.0.0` | bind |
| `CLASP_REGISTRY_KEEP_LAST` | `5` | retention default |
| `CLASP_TLS_CERT`, `CLASP_TLS_KEY`, `CLASP_TLS_CA` | unset | all three → mTLS, TLS 1.3, client cert required |

On-disk layout: `adapters/<name>/v<n>/{adapter.safetensors,metadata.json}`, an
`active` pointer (written atomically), `promotions.jsonl` and `gc.jsonl`
(append-only). A version directory is published by a single rename, so a crash
never leaves a half-written version. One process-wide write lock serialises every
mutation; run one worker.

- `python -m registry.demo --url URL` — drives every capability against a live
  registry and asserts each answer (the compose `demo-seed` job).
- `python -m registry.healthcheck` — container healthcheck, TLS-aware.
- `python -m registry.runs` — experiment configs, resumable sweeps, result store.
- `python -m registry.repro` — build / verify the reproducibility pack.
- `demo/save_promote_rollback_demo.sh` — the same lifecycle with curl only.

See `docs/devops.md` for compose profiles, CI, mTLS and the NFR evidence.
