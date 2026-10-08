# P2 Cluster — Phase II report sections (Weeks 14–16)

Author: Prasanth (P2 Cluster). Package: `services/cluster` (`clasp-cluster`).
Source of truth for the plan: *CLASP Swimlane v2* and *CLASP Daily Targets v2*.

This document is P2's contribution to the Phase-II report (Daily Targets W14: "Report §1–3 /
§4–6 draft"). It describes **what the code does today**, cites the test or evidence file behind
every claim, and lists what is **not** done. The whole-team report (other modules' sections,
rubric mapping, guide feedback, final PDF, submission) is outside what P2 can complete alone —
see §16.

## 1. Scope and status in one table

| Area | Status | Where |
|---|---|---|
| Flower server strategy + client, FedProx | done, tested | `server.py`, `client.py` |
| LoRA adapter format, ΔW = B·A, PEFT keys | done, tested | `adapter_format.py` |
| Streamed exact average, SVD aggregation, naive baseline | done, tested | `aggregation.py` |
| Round metrics, redistribution, retry, fault tolerance | done, tested | `server.py`, `redistribution.py` |
| Straggler policy + quorum | done, tested | `straggler.py` |
| Multi-cluster federation + isolation (in-process and HTTP) | done, tested | `federation.py`, `server.py` |
| Clustering novelty: flatten, cosine, k-means, warm start, dynamic, static fallback | done, tested | `clustering.py` |
| Re-clustering over HTTP | done, tested (new) | `server.py` `POST /recluster` |
| Reproducibility, evidence, demos | done, tested | `evidence.py`, `demo*.py`, `demo_runs/` |
| Live Edge ⇄ Cluster ⇄ Registry ⇄ Evaluation, mTLS (G1), DP µ tuning | **blocked by other modules** | `INTEGRATION_BOUNDARIES.md` |
| Real model / real data | not done | §15 |

Verification environment: Linux sandbox, Python 3.11 and 3.10, torch 2.14 (executed on CPU), flwr 1.32 (3.11) / 1.30 (3.10). **Not** run on Prasanth's
Windows machine from here (§14).

## 2. Architecture

```
 Edge clients (P1)                       Cluster service (P2)                        Registry (P4)
 ─────────────────                       ────────────────────                        ─────────────
 train LoRA (r=16, α=16)
     │  AdapterUpload (HTTP)  ──►  /clusters/{id}/uploads ─► per-cluster buffer
     │                              │   straggler policy (timeout, weights, quorum)
     │                              ▼
     │                         aggregation: ΔW_k = B_k·A_k → streamed weighted mean → truncated SVD → (A', B')
     │                              │                                  ▲ naive A/B mean kept as ablation
     │                              ▼
     │  ClusterAdapterBroadcast  ◄── /clusters/{id}/adapters/active  ──► SnapshotSink.publish (stub today)
     │                              │
     │                              ▼ retained updates of the last round
     │                         POST /recluster: flatten ΔW-updates → cosine → k-means(k=2) → membership
     └── (Flower gRPC path: SVDLoRAStrategy.aggregate_fit, one cluster per server)
```

Two network entry points exist and share the aggregation core without duplicating its math:

* **Flower path** — `SVDLoRAStrategy` (a `FedAvg` subclass) for one cluster per Flower server.
* **HTTP path** — FastAPI app `cluster.server:app` (port 8002 in `docker-compose.yml`/`Dockerfile`): the
  multi-cluster surface Edge talks to. State is in memory.

The in-process `MultiClusterFederation` (`federation.py`) drives real FedProx clients through the same
straggler, aggregation, re-clustering and redistribution code for tests, demos and evidence.

| Module | Responsibility |
|---|---|
| `adapter_format.py` | `LoRAAdapter`, PEFT/short key conversion, validation (shape, rank, NaN/inf) |
| `aggregation.py` | `StreamingWeightedMean`, `aggregate_svd`, `aggregate_naive`, `exact_average_delta`, `ensure_compatible` |
| `client.py` | `DummyClient`, `ToyLoRAModel`, `LoRAClient` (FedProx) |
| `server.py` | `SVDLoRAStrategy`, `build_server_app`, `start_grpc_server`, HTTP service |
| `straggler.py` | `StragglerPolicy`, `apply_policy` |
| `redistribution.py` | `build_broadcast`, `adapter_from_broadcast`, `redistribute`, `DeliveryReport` |
| `clustering.py` | flattening, cosine, k-means, `ClusterAssigner` |
| `federation.py` | `MultiClusterFederation`, `isolation_violations()` |
| `integration.py`, `tls.py` | P2-side seams to P4 / P3 (§16) |
| `evidence.py`, `demo.py`, `demo_all_weeks.py`, `demo_phase2.py` | evidence + demos |

## 3. Flower server and client

* **Client** — `LoRAClient(NumPyClient)`: `fit(parameters, config)` rebuilds the adapter from the flat
  array list, runs `local_steps` of SGD on a toy-but-real torch LoRA regression with the FedProx term, returns the
  updated arrays, `len(data)` as `num_examples`, and metrics `loss`, `client_id`, `duration_s`.
  `DummyClient` is the Week-1/2 skeleton (adds a `bump`; no torch). `simulated_delay_s` lets tests report a
  slow client **without sleeping**.
* **Server** — `SVDLoRAStrategy.aggregate_fit` replaces FedAvg's parameter averaging by §6, after the
  fault-tolerance filter of §9. `build_server_app` (new ServerApp API) and `start_grpc_server`
  (legacy `start_server`) expose it. The wire order of the flat arrays is layer-major, `target_modules`
  order, `[A, B]` per module (`LoRAAdapter.to_ndarrays`).
* The Flower path was exercised through the strategy's `aggregate_fit` with real `FitRes` objects
  (`tests/test_fault_tolerance.py`) and through the in-process simulation. A live Flower gRPC server with
  remote clients was **not** run in this environment, and `start_grpc_server` is insecure (no TLS, §16).

## 4. LoRA adapter representation and ΔW = B × A

Per transformer layer and target module (`q_proj`, `k_proj`, `v_proj`, `o_proj`): `lora_A` is `(r, in)`,
`lora_B` is `(out, r)`, ΔW = `lora_B @ lora_A`, applied as `W0 + (alpha / r) · ΔW`. The adapter holds
`modules[layer][module] = {"lora_A", "lora_B"}`; contract values are r = 16, **alpha = 16** (scaling 1.0),
four attention modules. `LoRAAdapter.validate()` checks completeness, 2-D shapes, rank agreement, and rejects
NaN/inf. Two key conventions are read and written: a short `layers.<i>.<module>.lora_{A,B}.weight` form
and the PEFT form `base_model.model.model.layers.<i>.self_attn.<module>.lora_{A,B}.weight`
(`to_peft_state_dict`, `to_peft_config`); a round trip is lossless (`tests/test_peft_interop.py`,
`tests/test_adapter_format.py`). `delta_w(module, layer)` is the single place ΔW is formed.

## 5. FedProx

Local objective: `task_loss + (µ/2) · Σ‖w − w_global‖²` over the trainable A and B tensors, `w_global` being
the adapter received at the start of the round (`LoRAClient.train_local`, plain SGD). Verified:
the proximal term keeps parameters closer to the global adapter than µ = 0; µ = 0 equals plain SGD and is
deterministic; the step is explicit Euler on the proximal term and **diverges when `lr·µ ≥ 2`**, which the
tests document (`tests/test_fedprox.py`). A diverged client returns NaN/inf, which `validate()` rejects.
What is **not** known: the best µ under differential privacy (blocked, §16).

## 6. Aggregation

For each (layer, module) the cluster combines client adapters *k* with weights *w_k*:

1. **Reconstruct** ΔW_k = B_k · A_k.
2. **Streamed exact average**: `m ← m + (w_i / W_i)(x_i − m)` (`StreamingWeightedMean`) — one accumulator per
   (layer, module), exact up to floating point, independent of the number of clients. The adapter iterator
   is consumed once, so a generator can feed it. (The HTTP service and Flower both still hold all uploads in
   memory until aggregation; the streaming is inside the aggregator.)
3. **Truncated SVD re-factorization**: ΔW̄ = U S Vᵀ; `B' = U_r √S_r`, `A' = √S_r V_rᵀ`, so `B'A'` is the best
   rank-r approximation of the exact mean (Eckart–Young), rank r = the clients' rank by default. The
   result is an ordinary adapter of the same shape, i.e. what a client can train from. If another output rank
   is requested, `alpha` is scaled with it so `(alpha/r)·B'A'` is unchanged.
4. **Naive baseline (D2 ablation)** — `aggregate_naive` averages A and B separately. `mean(B)·mean(A) ≠ mean(B·A)`
   in general; it is kept only for comparison. (It is exact in the degenerate case where every client shares the same A — e.g.
nobody has trained A yet — which is why a difference only appears once A drifts per client; `test_naive_equals_exact_only_in_the_degenerate_shared_A_case`.)

`ensure_compatible` requires identical `rank`, `alpha`, `target_modules`, `num_layers` across contributions (alpha/rank
is the scaling, so mixed values are not comparable) and raises otherwise. Tests: SVD ≈ exact average,
exactness cases, multi-layer, streamed single-pass, weights/adapters length mismatch, naive ≠ exact
(`tests/test_aggregation.py`, `tests/test_week6_review.py`). Measured on the toy (seed 1, one run, not a
statistical claim): after 5 rounds the SVD path reached mean loss 1.473 vs 1.543 for the naive path
(`demo_runs/phase2_cluster_evidence.md`). The cost of full SVDs at real model width (e.g. 4096 × 4096 × modules × layers per round)
was **not benchmarked**.

## 7. Round metrics

`SVDLoRAStrategy.round_log` (and the metrics Flower receives): `round`, `num_clients`, `num_failures`,
`aggregation`, `mean_loss` (over accepted clients that reported one), `duration_s` (`time.monotonic()`). The
federation records `RoundMetrics(round_id, num_clients, mean_loss, duration_s, aggregation)` per cluster per
round, and the HTTP manifest (`GET /clusters/{id}/aggregate/manifest`) records `round_id`, `num_clients`,
`aggregation`, `weighting`, `source_clients`, `skipped_clients`, optionally the SVD reconstruction error
(`include_manifest`), and `published_as` / `publish_error` when a registry sink is configured.

## 8. Redistribution and retry

After aggregation `build_broadcast` turns the adapter into a `ClusterAdapterBroadcast` (PEFT-keyed tensors +
an `adapter_config`-style `peft_config`); `adapter_from_broadcast` is the lossless inverse a client runs.
`redistribute(payload, client_ids, send, max_retries=3, backoff_s=0.1, backoff_factor=2.0, require_all=False)`
retries each failed delivery with exponential backoff and returns a `DeliveryReport(delivered, failed,
attempts)`; a client is **never skipped silently**, and one client's failure never blocks the others;
`require_all=True` raises `RedistributionError`. The transport is injected (`send`), so the same code serves
the simulation, HTTP and a future push channel. On the HTTP path distribution is **pull-based**
(`GET /clusters/{id}/adapters/active`); there is no push. Tests: `tests/test_redistribution.py`.

## 9. Retry, error handling and fault tolerance

| Situation | Behaviour |
|---|---|
| Flower transport failure | counted in `num_failures`, never aggregated |
| `FitRes` with non-OK status | skipped, reason `status:<CODE>` |
| malformed / wrong-shape / NaN-inf payload | skipped (`malformed: ...`), round continues |
| over-timeout or zero-example client | skipped by the straggler policy |
| quorum not met | Flower: `(None, {})`, round skipped; federation: previous cluster adapter kept; HTTP: 409 and the buffer is kept |
| undeliverable broadcast | retry with backoff; failure reported; a moved client that never received its new adapter is skipped as `stale_adapter` |
| HTTP | 422 malformed/incompatible/impossible rank, 409 stale `round_id` / `stale_adapter` / quorum, 403 isolation or identity mismatch, 401 identity required, 404 unknown cluster / nothing aggregated yet |
| crashing client in the federation | recorded as `failed: ...`, treated as a dropout |

## 10. Straggler policy and quorum

`StragglerPolicy(timeout_s=None, min_clients=1, min_fraction=0.0, weighting="samples")`. A client is skipped when
it has no examples or its **reported** `duration_s` exceeds `timeout_s`; accepted clients are weighted by
`num_examples` (or uniformly). **Quorum** = `max(min_clients, ceil(min_fraction · expected))` accepted updates,
where `expected` counts clients that never answered. Limits: the timeout acts on the *reported* duration (HTTP:
arrival time relative to the round's first upload) — it is not a wall-clock deadline that interrupts a hung
client, and Flower's own `round_timeout` is not configured. Evidence: with a 5 s timeout and a client reporting
30 s the client is skipped every round and the other clusters are unaffected; a quorum failure leaves the
cluster adapter bit-identical and the next round recovers (`demo_runs/phase2_cluster_evidence.md`;
`tests/test_straggler.py`, `tests/test_federation.py`).

## 11. Multi-cluster federation and cluster isolation

`MultiClusterFederation` runs a round for every cluster from a shared client pool; the HTTP service keeps
one `_ClusterState` per cluster id (buffer, round counter, active adapter, retained last-round updates).
**Isolation rules** (structural, then re-checked by `isolation_violations()`):

1. a client only trains from, contributes to, and receives the adapter of the cluster it is *currently assigned* to;
2. a round's aggregate contains only that cluster's members;
3. a client holding another cluster's adapter (failed redistribution after a move) is skipped as `stale_adapter`;
4. HTTP: a client assigned to cluster A may not upload to B (403) and, when an identity provider is installed,
   may not read B's adapter (403); a moved client's buffered upload in its old cluster is dropped;
   `base_cluster_id` on an upload names the adapter the client trained from, a mismatch is refused (409).

Round order inside the federation: **aggregate with the old assignment, then re-cluster, then redistribute to the
new members.** A moved client therefore never contributes to a cluster it did not belong to at the start of the round.
`isolation_violations()` is proven to bite (`test_isolation_check_detects_an_adversarial_delivery`).
Evidence: 0 violations across all evidence runs, including an adversarial placement of two clients in the wrong
cluster (`isolation_stress`).

### HTTP API (port 8002)

| Method / path | Purpose |
|---|---|
| `GET /healthz` | liveness + default-cluster state |
| `POST /uploads`, `POST /aggregate`, `GET /adapters/cluster/active`, `GET /aggregate/manifest` | legacy single-cluster surface (cluster id `cluster-default`) |
| `GET /clusters` | per cluster: round, pending uploads, has-active-adapter, members |
| `PUT /clusters/{id}/members` `{"client_ids": [...]}` | assign clients (moving them out of any other cluster); **the only call that creates a cluster** (`[]` registers an empty one) |
| `POST /clusters/{id}/uploads` | accept one `AdapterUpload` (201); **404** for a cluster nobody registered |
| `POST /clusters/{id}/aggregate` | aggregate the buffer: `{aggregation, rank, include_manifest, straggler_timeout_s, min_clients, min_fraction, expected_clients, weighting}` → `ClusterAdapterBroadcast` |
| `GET /clusters/{id}/adapters/active`, `GET /clusters/{id}/aggregate/manifest` | retrieve the aggregated adapter / the last manifest |
| `POST /recluster` | dynamic re-clustering, §12 |

State is in memory (lost on restart) and per process: run a single worker.

**Concurrency.** The handlers are plain `def` functions, so FastAPI runs them on a thread pool and several can touch
one cluster at once. Each `_ClusterState` has a lock that is held across the whole of an upload and the whole of an
aggregate: an upload cannot be iterated while it is added, and an upload that arrives while `aggregate_svd` is running
waits and is then answered 409 (its round was consumed) instead of 201-then-dropped. `PUT .../members` and `POST
/recluster` take the registry lock (and, for the latter, every cluster lock in sorted order); `/healthz`, `/clusters`
and the GET endpoints take no cluster lock and read snapshots, so a liveness probe never queues behind an aggregate.
Evidence: `tests/test_server_concurrency.py` drives the real app from real threads (a gated slow aggregate; a
12-uploader/continuous-aggregator stress run asserting every accepted upload is aggregated exactly once); with the
per-cluster lock disabled two of those tests fail.

**Quorum denominator.** `min_fraction` is a fraction of the clients the round *expected*, not of the uploads that
arrived (that would always be 100 %): `expected_clients` from the request, else the number of clients assigned to the
cluster, else — when nobody is assigned — the uploads that arrived, where `min_fraction` therefore cannot bite. The
value used is recorded in the manifest (`expected_clients`) and never falls below the uploads actually received.

## 12. Clustering: flattening, cosine similarity, k-means, warm start, dynamic re-clustering, static fallback

* **Flattening** — a client's *update* is `flatten(ΔW_returned) − flatten(ΔW_start)`, with `flatten` the concatenation of
  ΔW over (layer, module) (`flatten_update`); with no start (round 0, B = 0 init) the update is ΔW itself.
* **Cosine similarity** — computed from the LoRA factors without forming ΔW: ⟨ΔW_i, ΔW_j⟩_F =
  `Σ (B_iᵀ B_j) ∘ (A_i A_jᵀ)` per module, expanded bilinearly for `(returned − start)` (`delta_gram`, `update_gram`);
  tests check it equals the flattened inner products and handles identical / opposite / zero updates.
* **k-means** — k-means (k = number of clusters, 2 here) on the cosine matrix as a kernel (equivalent to k-means on
  L2-normalized vectors), `n_init=10`, seeded, canonical label order; it matches scikit-learn on the flattened vectors
  (`test_kmeans_matches_scikit_learn_on_flattened_vectors`, skipped only where scikit-learn is absent).
* **Warm start** — `ClusterAssigner(warm_start=...)`: the initial assignment (e.g. project type), kept until re-clustering is due
  and used as the fallback; new k-means labels are permuted onto the current assignment so cluster ids stay stable.
* **Dynamic re-clustering (D4)** — due after round 2 (`recluster_after_round=2`, optionally `recluster_every`), on the
  clients that returned updates; dropouts keep their assignment. The result records method, reason, moved clients,
  cohesion/separation and a **dynamic-vs-warm-start comparison** (agreement, ARI, cohesion/separation of each partition).
* **Static fallback (D12)** — assignment left unchanged, reason recorded, when: fewer updates than k, updates from
  unassigned clients, a degenerate cluster (`< min_cluster_size`), weak separation (within − between cosine gap
  `< min_separation`, default 0.1), or any clustering error. `mode="static"` never re-clusters (the baseline).

### Why an HTTP endpoint (`POST /recluster`) — decision

Appropriate, so implemented. The deployed service (Dockerfile / compose) is the HTTP app, and it owns exactly
the state re-clustering needs (membership, per-cluster active adapter, each contributor's upload); without an
endpoint a deployed Cluster could never re-cluster, since `MultiClusterFederation` is in-process only. It does not
bypass isolation: the only cross-cluster read is the similarity computation clustering is by definition, the response
contains membership and scalar statistics (never adapter tensors), and applying a decision uses the same move
rules as `PUT .../members`. Constraints that follow from that: it is **administrative** and, like
`PUT .../members`, has no authentication of its own (belongs behind P3 authorization / network policy).

**Request** `POST /recluster` (body optional):
`{"apply": false, "min_separation": 0.1, "min_cluster_size": 1, "seed": 0, "include_similarity": false}`

* Evidence = for each cluster, the adapters its *accepted contributors* uploaded in the **last completed round**
  and the cluster adapter they started from (`None` ⇒ round 0, update = ΔW). Only clients assigned to a cluster take part
  (others are listed in `ignored`).
* `apply=false` (default) is a **dry run**; `apply=true` moves clients when the method is `dynamic`, drops a moved
  client's buffered upload in its old cluster, and **consumes the evidence** (a second apply → 409).
* Errors: 409 fewer than two clusters with members, or no completed aggregation round; 422 invalid parameters.

**Response** (200): `{"applied": bool, "method": "dynamic" | "static_fallback", "reason": str, "completed_rounds": int,
"clusters": [...], "clients_clustered": [...], "moved": {client: [old, new]}, "assignment": {client: cluster},
"cohesion": float|null, "separation": float|null, "comparison": {agreement, ari, dynamic_cohesion, ...}|null,
"ignored": {client: reason}, "similarity": [[...]] (only with include_similarity)}`.

HTTP is pull-based, so a moved client must fetch its new cluster's adapter and should send `base_cluster_id`
on its next upload. Tests: `tests/test_server_recluster.py` (dry run vs apply, evidence consumption, isolation after a
move, stale adapter, weak-separation fallback, parameter validation, round-2 start adapter).

## 13. Reproducibility

Seeds are explicit end to end: data partitions and base maps (`torch.Generator`), toy model init, the initial adapter,
and the k-means seed; clients are processed in sorted order. Two runs with the same seed produce **byte-identical**
cluster adapters including the re-assignments (`test_two_seeded_runs_are_bit_identical_including_reassignments`;
`reproducibility` in the evidence run, SHA-256 of the adapters, every seed). This is claimed for the same machine and torch build only — not across platforms, torch builds or GPUs.
Observed, not guaranteed: the digests were also identical between a Python 3.11 / numpy 2.4.6 / flwr 1.39 run and a
Python 3.10 / numpy 2.2.6 / flwr 1.30 run on that machine (`demo_runs/validation/VALIDATION_SUMMARY.md`).

## 14. Evidence and results

All figures: `demo_runs/phase2_cluster_evidence.md` / `.json`, generated by `python -m cluster.evidence`
(seeds 1, 2, 3; 5 rounds; synthetic toy-LoRA regression, dim 32, rank 16, CPU; not DP; not the real model).

Run header: Seeds [1, 2, 3], 5 rounds, python 3.11.17, torch 2.14.1+cu130, flwr 1.39.0, numpy 2.4.6, Linux-6.18.44-fc-v70-x86_64-with-glibc2.39; run took 82 s.

### Dynamic re-clustering vs static (mean over seeds)

| scenario | mode | mean ARI vs true groups | seeds recovering truth | mean final loss |
|---|---|---|---|---|
| correct_warm_start | dynamic | 1.000 | 3/3 | 1.4853 |
| correct_warm_start | static | 1.000 | 3/3 | 1.4853 |
| one_wrong_label | dynamic | 1.000 | 3/3 | 1.5139 |
| one_wrong_label | static | 0.324 | 0/3 | 1.5878 |

Reproducible in all seeds (adapters byte-identical across two runs): **True**. Isolation violations found across all runs: **0**.

### FedProx mu sweep (toy, non-DP)

| mu | lr*mu | stable in all seeds | mean final loss (stable seeds) |
|---|---|---|---|
| 0.0 | 0 | True | 1.4655 |
| 0.01 | 0.001 | True | 1.4853 |
| 0.1 | 0.01 | True | 1.6680 |
| 1.0 | 0.1 | True | 3.2764 |
| 10.0 | 1 | True | 4.2781 |
| 50.0 | 5 | False | diverged |

### 3-client training (seed 1, 5 rounds)

| aggregation | round mean losses | loss decreased |
|---|---|---|
| svd | 3.999, 3.032, 2.343, 1.834, 1.473 | True |
| naive | 3.999, 3.081, 2.418, 1.917, 1.543 | True |

### Straggler / dropout / quorum (seed 1)

- timeout straggler (client-2 of cluster-web, 30 s vs 5 s timeout): contributors per round [2, 2, 2]
- dropout and rejoin (client-1 of cluster-sci offline in rounds 1-2): sci contributors per round [2, 2, 3]
- quorum failure (min_clients=2, 1 answer): web aggregated=False, web adapter unchanged=True, sci adapter updated=True, next round web aggregated=True

### Static fallback (seed 1)

- too few updates: static_fallback - only 1 client update(s) present for k=2
- no group structure: static_fallback - weak separation (within-between gap 0.037 < 0.1); wrong label kept: True
- static mode history: ['warm_start']

What the numbers do and do not show: the group structure is **planted by construction**, so recovery shows the mechanism
works when structure exists — not that real project clusters exist. The µ sweep says nothing about DP. Loss differences
between SVD and naive come from a single seed.

Test and lint status at the time of writing (this environment): **216 passed, 0 failed, 1 skipped** on Python 3.11.17 and again on Python 3.10.20 (the one skip, `tests/test_demo.py:42`, is environment-dependent: it asserts the torch-missing path and skips when torch is installed); `ruff check .` clean on both ruff 0.16.8 and the team-pinned ruff 0.15.22 (built-in defaults, run from `services/cluster`; the repository commits no ruff configuration); `python -m cluster.demo --seed 42`, `python -m cluster.demo_all_weeks`, `python -m cluster.demo_phase2` and `python -m cluster.evidence` all exit 0. Raw logs: `demo_runs/validation/`.

Verification matrix:

| Item | Verified in the sandbox | Needs verification on Prasanth's Windows machine | Blocked by another module |
|---|---|---|---|
| unit/integration tests, lint, demos, evidence | yes | yes (paths, console encoding, `.venv` there) | — |
| real TLS handshake of the HTTP app against a test CA | yes | recommended | P3 certificates / identity |
| Edge ⇄ Cluster live, G1 mTLS | no | — | P1 + P3 |
| DP µ tuning | no | — | P3 |
| Registry publish, Evaluation | stubs only | — | P4, P5 |

## 15. Known limitations

1. **Toy workload.** All training/evidence uses a synthetic regression with a chained 4-module toy "layer"; no real
   DeepSeek-Coder adapter, dataset, GPU or real-width run. Real-width shapes appear only in shape/format tests with synthetic adapters.
2. **Timeouts act on reported/arrival time**, not a hard deadline; Flower's `round_timeout` is not configured.
3. **In-memory HTTP state**: resets on restart, no persistence, one worker (locks are per process).
4. **Identity** is self-asserted. `server.configure_identity` and `server.configure_snapshot_sink` are **hooks that only tests call**: nothing in the running service installs them, so by default uploads are not bound to a client certificate and nothing is published to a registry. Membership and re-cluster endpoints are unauthenticated admin operations.
5. **mTLS is not finished.** `cluster/tls.py` builds TLS settings and is tested against an ephemeral test CA, but it is not wired into a deployed service, uvicorn does not expose the peer certificate to the app, and `start_grpc_server` is insecure. G1 is blocked by P3.
6. **Flower strategy = one cluster per server**; multi-cluster over Flower gRPC is not implemented (multi-cluster = HTTP service + in-process federation).
7. **Re-clustering** uses the last round only, needs ≥ k updates, keeps k = number of existing clusters (never creates/merges clusters), and
   its `min_separation = 0.1` is a heuristic tuned on nothing but the toy. Clustering on real update directions is untested.
8. **Memory/compute at scale**: uploads are buffered whole; full SVDs per (layer, module) were not benchmarked at 6.7B width.
9. **Same-alpha/rank requirement**: heterogeneous ranks/alphas are rejected, not merged.
10. **Epsilon**: the broadcast has an `epsilon` field but Cluster never fills it (P3's accountant).
11. `base_model_name_or_path` defaults to Edge's 1.3B dev model.
12. Edge's `merge.py` is not in the working tree; the contract constants pinned in `test_peft_interop.py` were copied from the Edge branch earlier and are not re-verified live.

## 16. Cross-module dependencies (summary — details in `INTEGRATION_BOUNDARIES.md`)

* **P1 Edge**: upload/download client missing; alpha/rank/modules aligned (Cluster default alpha 32 → 16 in Week 6).
* **P3 Security**: nothing implemented; Cluster provides `tls.py`, the identity hook and an ephemeral-CA test. **G1 BLOCKED BY P3. DP µ tuning BLOCKED BY P3.**
* **P4 Registry**: nothing implemented; Cluster provides `SnapshotSink` + `AdapterRef` mapping + an in-memory stub.
* **P5 Evaluation**: nothing implemented; Cluster makes no calls into it.
* **Report (W14–W15)**: P2's sections are this document; the rubric file is an unfilled template, there is no guide feedback
  to incorporate, and assembling the whole-team draft / building the final PDF / submitting to the guide depends on the
  other members' sections.

## 17. How to run

```bash
pip install -e contracts -e "services/cluster[test]"
pytest services/cluster/tests
(cd services/cluster && ruff check .)
python -m cluster.demo --seed 42          # Week 5 3-client round
python -m cluster.demo_all_weeks          # whole-pipeline walkthrough (runs the pipeline; checks the report/evidence files exist)
python -m cluster.demo_phase2             # Weeks 7-13 walkthrough (~10 s)
python -m cluster.evidence                # Phase-II evidence (~80 s CPU) -> demo_runs/
uvicorn cluster.server:app --port 8002    # HTTP service
```
