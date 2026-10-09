# Cluster Layer — Flower + FedProx aggregation, FastAPI

**Owner:** Prasanth (P2)

Skeleton module. See the team's internal conventions notes (local). Depends on
the shared `contracts` package; keep cross-module interaction interface-driven.

```bash
pip install -e contracts -e "services/cluster[test]"
pytest services/cluster/tests
```

## 3-client aggregation demo

One command runs a full simulated federated round end to end: 3 clients do
local training, updates are collected, and the cluster server aggregates them
via delta-W exact averaging + truncated-SVD re-factorization (`aggregation.py`),
exactly as `simulation.run_round` already does for tests.

- Python >= 3.10; `pip install -e contracts -e services/cluster`.
- `torch` is needed for the real FedProx client engine (`pip install torch`).
  Without it the demo falls back to the `DummyClient` round-trip engine and still
  completes (no real training losses). Force one with `--engine auto|fedprox|dummy`.

```bash
python -m cluster.demo --seed 42        # flags: --clients 3 --aggregation svd|naive --rank 16 --output <path>
```

It prints a stage-by-stage trace, a round-metrics block and a
`FINAL ROUND STATUS: SUCCESS` banner (exit code `0`), and writes a JSON artifact to
`services/cluster/demo_runs/`. The demo is seeded end to end, so two runs with
the same seed produce a byte-identical aggregated adapter
(`tests/test_demo.py::test_round_result_reproducible_end_to_end`).
A narrated straggler / re-clustering / fallback / HTTP run: `python -m cluster.demo_phase2` (~10 s).

## HTTP surface (:8002)

```bash
uvicorn cluster.server:app --host 0.0.0.0 --port 8002
```

Two route sets over **one** per-cluster state (same locking, same isolation rules):

**Integration surface** — the cluster is named in the body or path; used by `demo_ui`, Edge and
`tests/integration`:

| Endpoint | Purpose |
|---|---|
| `GET  /healthz` | liveness + per-cluster round bookkeeping |
| `POST /uploads` | **seam A** — one client's trained LoRA (`AdapterUpload`, `cluster_id` in the body; the cluster is created by its first upload, capped at `MAX_CLUSTERS`) |
| `POST /aggregate` | run the D2 aggregation over one cluster's buffer (`cluster_id` in the body) |
| `GET  /adapters/{cluster_id}/active` | the aggregate, PEFT keys + `adapter_config` |
| `GET  /adapters/{cluster_id}/download` | the same thing as safetensors bytes |
| `GET  /adapters/{cluster_id}/manifest` | last aggregation's manifest (reconstruction error, ε, skipped clients …) |
| `POST /adapters/{cluster_id}/publish` | **seam B** — push the active aggregate to the State Registry (`CLASP_REGISTRY_URL`), including the cluster's ε |
| `GET  /adapters/cluster/active`, `GET /aggregate/manifest` | aliases for the default cluster (`cluster-default`) |

**Cluster-addressed surface** — membership, isolation, quorum, re-clustering:

| Endpoint | Purpose |
|---|---|
| `PUT  /clusters/{id}/members` `{"client_ids": [...]}` | assign clients (moving them out of any other cluster); **the only call that registers a cluster** here (`[]` = an empty one) |
| `POST /clusters/{id}/uploads`, `POST /clusters/{id}/aggregate` | as above, but the id must be registered: **404** for an unknown id, **403** for a client assigned to another cluster |
| `GET  /clusters/{id}/adapters/active`, `GET /clusters/{id}/aggregate/manifest` | retrieve the aggregate / the last manifest |
| `GET  /clusters` | per cluster: round, pending uploads, has-active-adapter, members |
| `POST /recluster` | dynamic re-clustering over the last completed round — dry run unless `{"apply": true}` |

Behaviour worth knowing:

* **Quorum and stragglers.** `POST …/aggregate` accepts `straggler_timeout_s`, `min_clients`, `min_fraction`,
  `expected_clients`, `weighting`. `min_fraction` is a fraction of the clients the round *expected*: the request's
  `expected_clients`, else the number of clients assigned to the cluster, else (nobody assigned) the uploads that
  arrived — in which case it cannot bite, so assign members or pass `expected_clients`. Below quorum the answer is
  `409` and the buffer is kept.
* **Concurrency.** Handlers run on FastAPI's thread pool. Each cluster has a lock held across a whole upload and a
  whole aggregate (and `publish` takes it to read the active adapter), so an upload that arrives during an aggregate
  waits and is then answered `409` (its round was consumed) instead of a `201` that is later dropped. `/healthz`,
  `/clusters` and the GET endpoints take no lock. State is per process: run **one worker**.
* **Aggregation paths.** `exact_lowrank` (default) is the same mathematics as the reference path evaluated through the
  low-rank factors — seconds instead of minutes at real width; `exact_lowrank: false` forces the reference path.
  `retain_uploads: true` keeps the buffer for the D2 ablation; a retained run is not a completed round (not
  published to a sink, not evidence for `/recluster`). `aggregate_svd`, `aggregate_naive` and `exact_average_delta` are
  unchanged; a LAPACK non-convergence on a real 2048×2048 module is retried on the transpose (`_svd`).
* **D7 (ε).** `AdapterUpload.epsilon` is optional. The cluster's ε is the **max over the aggregated clients**
  (stragglers that were skipped do not count) and is recorded in the manifest, on the broadcast and as
  `privacy.epsilon` in what `publish` sends the registry. If any aggregated client did not report an ε the cluster's ε
  is `null` (a non-DP contribution means no finite ε describes the aggregate) and the manifest lists
  `epsilon_not_reported_by`.

State is in-memory and keyed by `cluster_id` (D1: `web`, `scientific`); durable versioning is the registry's job.
`aggregate_svd_lowrank` is described in `docs/integration-sprint.md` §3.

## Straggler policy, multi-cluster, clustering

| Piece | Module | What it does |
|---|---|---|
| Straggler policy | `cluster/straggler.py` | timeout-skip, sample (or uniform) weighting, quorum (`min_clients`, `min_fraction`) |
| Fault-tolerant `aggregate_fit` | `cluster/server.py::SVDLoRAStrategy` | skips non-OK / malformed / non-finite / over-timeout clients, counts them in `num_failures`, skips the round below quorum |
| Redistribution + retry | `cluster/redistribution.py` | `build_broadcast` / `adapter_from_broadcast` round-trip; `redistribute` retries per client with exponential backoff and returns a `DeliveryReport` |
| Multi-cluster rounds + isolation | `cluster/federation.py` | `MultiClusterFederation`: per-cluster FedProx round, SVD aggregation, redistribution; `isolation_violations()` |
| Clustering | `cluster/clustering.py` | flatten delta_W update -> cosine matrix (also computed from the LoRA factors without forming delta_W) -> k-means (k=2) -> `ClusterAssigner`: warm start, dynamic re-clustering after round 2, static fallback, dynamic-vs-warm-start comparison |
| Seams for other modules | `cluster/tls.py`, `cluster/integration.py`, `server.configure_identity` / `configure_snapshot_sink` | **Hooks only, not wired at startup.** See the limits below |

Re-clustering uses each client's *update* (`delta_W(returned) - delta_W(adapter it started from)`).
Order inside a round: every cluster aggregates the clients that trained from its adapter,
*then* assignments are re-evaluated; a moved client gets its new cluster's adapter for the next round.
Static fallback triggers (assignment left unchanged, reason recorded): fewer clients than `k`,
updates from unassigned clients, a degenerate (tiny) cluster, weak separation
(within - between cosine gap < `min_separation`), or any clustering error. A moved client's *buffered* upload in
its old cluster is dropped, and `base_cluster_id` on an upload names the adapter the client trained from — a
mismatch is refused (`409`, `stale_adapter`).

FedProx stability: the proximal step is explicit Euler, stable only while `lr * mu < 2`; a diverged
client returns non-finite tensors, which are rejected at the adapter-format boundary (HTTP 422 / skipped in a round).

## Known limits

* **mTLS is not finished and is not wired into the running service.** `client_id` on the HTTP path is
  self-asserted. `server.configure_identity` (P3's identity) and `server.configure_snapshot_sink` (automatic
  publish of every aggregate) are hooks that only tests call, so by default uploads are not tied to a client
  certificate. `tls.py` builds TLS settings and is tested against an ephemeral test CA; uvicorn does not expose the
  peer certificate to the app, and `start_grpc_server` is insecure. G1 has not passed. Publishing to the registry
  works today through `POST /adapters/{id}/publish`, on request.
* Membership and re-cluster endpoints are unauthenticated admin operations; HTTP state is in memory and per process.
* The Flower strategy aggregates a single cluster per server; multi-cluster = the HTTP service + the in-process
  federation.
* The demos use a toy workload, not the real model, and not DP: no FedProx µ sweep under DP has been run.

Per module (Edge / Security / Registry / Evaluation): what Cluster expects, what exists, what is mocked and what is
blocked — [`docs/INTEGRATION_BOUNDARIES.md`](docs/INTEGRATION_BOUNDARIES.md). End-semester demo script:
[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).
