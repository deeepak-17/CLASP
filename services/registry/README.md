# State Registry — versioned adapters, two-sided promotion, composites

**Owner:** Deepak (P4)

The registry is CLASP's only stateful service. It stores every LoRA adapter as an
immutable, sha256-addressed safetensors version, holds each upload as an inactive
candidate until the D5 two-sided rule decides it, and keeps an audit trail of
every decision. It can also store a pre-merged D6 composite, but in the live
loop the composite is built on the edge (U1): the edge pulls the promoted
cluster adapter and merges its own client layer. Metadata lives in SQLite,
weights stay on disk. It never needs torch: tensors are opaque bytes, except
when building a composite (numpy).

```bash
pip install -e contracts -e "services/registry[test]"
pytest services/registry/tests -q --cov=registry --cov-fail-under=90
CLASP_REGISTRY_DATA=./_data python -m registry.serve      # :8004, OpenAPI at /docs
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/healthz` | liveness + contracts version |
| GET | `/adapters` | adapter names |
| POST | `/adapters/{name}/versions` | save a new immutable version (multipart: `file` + `meta` JSON) |
| GET | `/adapters/{name}/versions` | every version's metadata, `active`, `candidate` and `rejected` |
| GET | `/adapters/{name}/versions/{v}` | one version's metadata |
| GET | `/adapters/{name}/versions/{v}/file` | the safetensors payload |
| GET | `/adapters/{name}/active` | metadata of the active version |
| GET | `/adapters/{name}/lineage` | versions with parents, sha256, ε and the decisions on each |
| POST | `/adapters/{name}/compose` | store `α·cluster + β·client` as one COMPOSITE version (D6) |
| POST | `/adapters/{name}/promote` | D5 rule on an inactive candidate; optional build-on-promote composite |
| POST | `/adapters/{name}/restore` | operator restore / rollback drill (audited) |
| GET | `/adapters/{name}/promotions` | the audit trail |
| POST | `/adapters/{name}/gc` | retention for one adapter (dry run by default) |
| POST | `/gc` | retention across every adapter |
| GET | `/sync/pending` | cluster steps of decisions not yet applied |
| POST | `/sync/{id}/done` | report one cluster step: `{"ok": bool, "error"?: str}` |

The reference for every route and response field is the module docstring of
`src/registry/app.py`.

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
`round`, `seed`, `cluster_id`, `source_clients`, `set_active` (default false), and
for composites `composed_from`. A save is an inactive candidate: the active
pointer does not move until `promote` decides it (`set_active: true` is for
operator seeding only). All versions under one name share one kind.

`GET /adapters/{name}/versions` answers `name`, `active`, `versions` and two
derived fields: `candidate`, the newest version above `active` with no decision
yet (null when there is none), and `rejected`, the sorted versions that were
refused: a REJECT, or a D5 `rollback` recorded before the candidate flow (it
named the refused candidate). A refused version is never a candidate and never
a restore target.

### Promote (D5) and build-on-promote (D6)

```json
{"eval": <contracts.EvalResult>,
 "baseline_guard": [{"benchmark": "LineCompletion", "edit_similarity": 0.60,
                     "exact_match": 0.30, "n_examples": 300}],
 "composite": {"name": "composite-flask", "cluster": "cluster-web",
               "client": "flask", "alpha": 0.5, "beta": 1.0}}
```

`composite` is optional. The version in `eval.adapter` must exist (404), be newer
than the active version (409) and not be decided already (409). PROMOTE iff
in-project edit similarity improves beyond the measured noise band over the
baseline (the active version, or the bare base model when nothing is active)
**and** the `LineCompletion` guard's edit similarity drops by at most 0.02
(2 points); PROMOTE makes the candidate active. Otherwise the decision is REJECT
and `active` stays where it was, so a failed candidate is never served. A guard
without `LineCompletion` edit similarity on both sides is a 422 and nothing is
recorded (contracts >= 1.2.0). With `composite`, a PROMOTE also stores the
composite built from the parts, pinned to the version being promoted. The
composite is built in memory before anything is written, so a bad composite
request rejects the whole call and records no decision.

The response is the decision plus `cluster_sync_id`. For a `cluster` adapter the
promotion transaction also writes a `cluster_sync` row (the cluster service must
switch to the candidate, or drop it); for any other kind it is null. Client
adapters carry their cluster's `cluster_id` too, but their decisions are not
cluster steps. `GET /sync/pending` lists the rows still to apply (`id`,
`adapter`, `version`, `cluster_id`, `sha256`, `action`, `attempts`,
`last_error`, `superseded_by`); `POST /sync/{id}/done` marks one applied
(`ok: true`) or counts a failed attempt (`ok: false`). The demo panel retries
until it lands.

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
`active` by hand and records it in the audit trail. `to_version` defaults to the
highest version below the active one that was live (promoted, or the target of a
recorded decision); with none, 409. An explicit `to_version` that was refused
(see `rejected` above) is a 409; an undecided one is the operator's choice. It
is the only way `active` moves backwards. Every composite whose active version
was built from the current version moves with it: an existing composite is
reused, otherwise it is rebuilt; each move is recorded and listed under
`composites`; if a replacement cannot be built, 409 and nothing changes. This is
the rollback drill and the D11 "restore ≤ 10 s" path. The cluster service's own
`/clusters/{id}/adapters/active/restore` likewise never serves a refused version.

`POST /adapters/{name}/gc` `{"keep_last"?: n, "dry_run"?: bool}` keeps the newest
N versions plus everything that was ever live or is still needed — the active
version, anything promoted, anything a restore made active, and any version a
composite names — and deletes the rest. Dry run by default; `keep_last` defaults to
`CLASP_REGISTRY_KEEP_LAST` (else 5). Version numbers are never reused.

## Operations

| Variable | Default | Meaning |
|---|---|---|
| `CLASP_REGISTRY_DATA` | `/data/registry` | storage root (compose volume `registry-data`) |
| `CLASP_REGISTRY_DB` | `sqlite:///<CLASP_REGISTRY_DATA>/registry.db` | metadata database URL |
| `CLASP_REGISTRY_PORT` / `_HOST` | `8004` / `0.0.0.0` | bind |
| `CLASP_REGISTRY_KEEP_LAST` | `5` | retention default |
| `CLASP_TLS_CERT`, `CLASP_TLS_KEY`, `CLASP_TLS_CA` | unset | all three → mTLS, TLS 1.3, client cert required |

On-disk layout: `registry.db` (SQLite, WAL: adapters, versions, the active
pointer, composite parts, promotion decisions, cluster sync) and
`adapters/<name>/v<n>/{adapter.safetensors,metadata.json}` for the weights, plus
`gc.jsonl` per adapter (the retention log, still a file). A version directory is
published by a single rename and served only once its database row commits; a
directory without a row (an interrupted write) is removed by the orphan sweep,
which runs once when the store is constructed and at every non-dry-run GC. A
volume written by the older file-based registry (an `active` file and
`promotions.jsonl` per adapter) is imported into the database on the first
start. One process-wide write lock serialises every mutation; run one worker.

- `python -m registry.demo --url URL` — drives every capability against a live
  registry and asserts each answer (the compose `demo-seed` job).
- `python -m registry.healthcheck` — container healthcheck, TLS-aware.
- `python -m registry.runs` — experiment configs, resumable sweeps, result store.
- `python -m registry.repro` — build / verify the reproducibility pack.
- `demo/save_promote_rollback_demo.sh` — the same lifecycle with curl only.

See `docs/devops.md` for compose profiles, CI, mTLS and the NFR evidence.
