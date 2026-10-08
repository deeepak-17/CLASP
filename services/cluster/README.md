# Cluster Layer — Flower + FedProx aggregation, FastAPI

**Owner:** Prasanth (P2)

Skeleton module. See the team's internal conventions notes (local). Depends on
the shared `contracts` package; keep cross-module interaction interface-driven.

```bash
pip install -e contracts -e "services/cluster[test]"
pytest services/cluster/tests
```

## Week 5 demo — 3-client aggregation round

One command runs a full simulated federated round end to end: 3 clients do
local training, updates are collected, and the cluster server aggregates them
via delta-W exact averaging + truncated-SVD re-factorization (`aggregation.py`),
exactly as `simulation.run_round` already does for tests.

### Requirements

- Python >= 3.10
- `pip install -e contracts -e services/cluster` (installs numpy, pydantic, flwr)
- `torch` for the real FedProx client engine (`pip install torch`). If torch
  is not installed, the demo automatically falls back to the Week 1/2
  `DummyClient` round-trip engine so it still completes — it just won't have
  real training losses. Force one or the other with `--engine`.

### Command

```bash
python -m cluster.demo --seed 42
```

Useful flags: `--clients 3` (default), `--aggregation svd|naive`,
`--engine auto|fedprox|dummy`, `--rank 16`, `--output <path>`.

### Expected output

A stage-by-stage trace (client init → local training → collection →
aggregation) followed by a round-metrics block (round id, client count,
training time, per-client losses, aggregation method, SVD rank, status) and a
`FINAL ROUND STATUS: SUCCESS` banner. Exit code `0` on success.

### Where results go

Every run writes a JSON artifact to `services/cluster/demo_runs/` (default
`round_<UTC timestamp>.json`, or the path given via `--output`) containing
the config, resolved engine, round metrics, per-client losses, and the final
aggregated adapter's shapes — usable as the backup/dry-run evidence record.
`demo_runs/week5_dry_run.*` and `demo_runs/week5_backup_run.*` are the
recorded Week 5 evidence runs.

### Reproducibility

The demo is seeded end to end (`--seed`, default 42): client data partitions,
model init, and the initial global adapter are all derived from it, so two
runs with the same seed produce a byte-identical aggregated adapter (see
`services/cluster/tests/test_demo.py::test_round_result_reproducible_end_to_end`).


## Weeks 7-13 — straggler policy, multi-cluster, clustering novelty

| Piece | Module | What it does |
|---|---|---|
| Straggler policy (W7) | `cluster/straggler.py` | timeout-skip, sample (or uniform) weighting, quorum (`min_clients`, `min_fraction`) |
| Fault-tolerant `aggregate_fit` (W4/W7) | `cluster/server.py::SVDLoRAStrategy` | skips non-OK / malformed / non-finite / over-timeout clients, counts them in `num_failures`, skips the round below quorum |
| Redistribution + retry (W4) | `cluster/redistribution.py` | `build_broadcast` / `adapter_from_broadcast` round-trip; `redistribute` retries per client with exponential backoff and returns a `DeliveryReport` |
| Multi-cluster rounds + isolation (W8) | `cluster/federation.py` | `MultiClusterFederation`: per-cluster FedProx round, SVD aggregation, redistribution; `isolation_violations()` |
| Multi-cluster HTTP (W8) | `cluster/server.py` | `/clusters/{id}/uploads\|aggregate\|adapters/active\|aggregate/manifest`, `PUT /clusters/{id}/members` (403 for a client uploading to a cluster it is not assigned to) |
| Clustering novelty (W10) | `cluster/clustering.py` | flatten delta_W update -> cosine matrix (also computed from the LoRA factors without forming delta_W) -> k-means (k=2) -> `ClusterAssigner`: warm start, dynamic re-clustering after round 2, static fallback, dynamic-vs-warm-start comparison |
| Re-clustering over HTTP (W10) | `cluster/server.py` | `POST /recluster`: dry run by default, `apply` moves clients; evidence = last completed round |
| P2-side seams (W7, W9) | `cluster/tls.py`, `cluster/integration.py`, `server.configure_identity` / `configure_snapshot_sink` | **Hooks only, not wired at startup.** `tls.py` builds TLS settings and is tested against an ephemeral test CA; `configure_identity` / `configure_snapshot_sink` are called only from tests, so the running service has **no** authenticated client identity and publishes **nothing** to the registry. **mTLS is not finished** (G1 blocked by P3); the registry sink is a stub (**blocked by P4**) |

Re-clustering uses each client's *update* (`delta_W(returned) - delta_W(adapter it started from)`).
Order inside a round: every cluster aggregates the clients that trained from its adapter,
*then* assignments are re-evaluated; a moved client gets its new cluster's adapter for the next round.
Static fallback triggers (assignment left unchanged, reason recorded): fewer clients than `k`,
updates from unassigned clients, a degenerate (tiny) cluster, weak separation
(within - between cosine gap < `min_separation`), or any clustering error.

Re-clustering is also available over HTTP: `POST /recluster` (dry run by default; `{"apply": true}` moves
clients) uses the last completed round of each cluster. Details, request/response and the isolation argument:
[`docs/P2_CLUSTER_REPORT.md`](docs/P2_CLUSTER_REPORT.md) §11-12.

HTTP service behaviour worth knowing:

* **Only `PUT /clusters/{id}/members` creates a cluster.** An upload or aggregate for an id nobody registered is a
  `404` (a typo cannot open a new cluster). `PUT` with an empty `client_ids` registers an empty cluster.
* **Quorum denominator.** `min_fraction` is a fraction of the clients the round *expected*: the request's
  `expected_clients`, else the number of clients assigned to the cluster, else (nobody assigned) the uploads that
  arrived — in which case `min_fraction` cannot bite, so assign members or pass `expected_clients`. The manifest
  records `expected_clients`.
* **Concurrency.** Handlers run on FastAPI's thread pool; each cluster has a lock held across a whole upload and a
  whole aggregate, so an upload that arrives during an aggregate waits and is then answered `409` (its round was
  consumed) instead of a `201` that is later dropped. `/healthz`, `/clusters` and the GET endpoints take no lock.
  State is per process: run **one worker**.

Known limits (full list in the report, §15): **mTLS is not finished and not wired into the running service.**
`client_id` on the HTTP path is self-asserted: `server.configure_identity` / `configure_snapshot_sink` are hooks for
P3's identity and P4's registry that only tests call, so by default uploads are not tied to a client certificate and
nothing is published (real identity = P3, **G1 blocked by P3**); membership and re-cluster endpoints are
unauthenticated admin operations; HTTP state is in memory; the Flower strategy aggregates a single cluster per
server; all evidence is a toy workload, not the real model, and not DP (**µ tuning under DP is blocked by P3**).

## Weeks 14-16 documentation

| Document | Content |
|---|---|
| [`docs/P2_CLUSTER_REPORT.md`](docs/P2_CLUSTER_REPORT.md) | P2's Phase-II report sections: architecture, Flower, FedProx, ΔW = B·A, aggregation, redistribution, straggler/quorum, isolation, clustering, HTTP API, reproducibility, evidence, limitations |
| [`docs/INTEGRATION_BOUNDARIES.md`](docs/INTEGRATION_BOUNDARIES.md) | per module (Edge / Security / Registry / Evaluation): what Cluster expects, what exists, match, mockable, blocked; G1 and DP µ tuning status |
| [`docs/WEEK6_REVIEW.md`](docs/WEEK6_REVIEW.md) | Week 6 code-review pass (original panel feedback was unavailable) |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | end-semester demo script |
| [`demo_runs/phase2_cluster_evidence.md`](demo_runs/phase2_cluster_evidence.md) | generated evidence (`python -m cluster.evidence`) |

### Evidence run

```bash
python -m cluster.demo_phase2         # narrated Weeks 7-13 run: straggler, re-clustering, fallback, HTTP (~10 s)
python -m cluster.evidence            # writes demo_runs/phase2_cluster_evidence.{json,md} (~80 s, CPU, needs torch)
pytest services/cluster/tests         # real-FedProx multi-cluster tests are skipped without torch
```

FedProx stability: the proximal step is explicit Euler, stable only while `lr * mu < 2`; a diverged
client returns non-finite tensors, which are now rejected at the adapter-format boundary (HTTP 422 / skipped in a round).
