# Amended decisions (2026-11-14)

`MASTER_PLAN.md` D1-D12 are the plan of record. Seven of them, or details of
them, no longer match what is built. This file records each one: what the plan
said (*as written*), what the code does (*as built*), and why. The plan itself
is left as the historical record; where the two disagree, this file and the
code win.

## D1 — web cluster composition

- **As written:** the web cluster is {django, flask, requests}.
- **As built:** the web cluster is {flask, requests, werkzeug}. Every federated
  round so far, and the integration round, ran on this composition.
- **Reason:** Django is not in the collected corpus and Werkzeug took its
  place. The reason was not recorded at the time. D1 changes need all-hands
  sign-off; this one has none on record yet, and every result so far (both
  federated rounds, the integration round) is on the werkzeug composition.
  Restoring Django would mean collecting it, re-partitioning and retraining
  every web-cluster adapter. See `docs/dataset.md`.

## D2 / D3 — cluster aggregation and the client layer

- **As written:** each round the cluster aggregates the client uploads afresh
  (FedProx-weighted exact average, truncated-SVD re-factorization), and the
  client layer trains on the frozen merged `Wbase + alpha·dWcluster`.
- **As built:** since 2026-10-14 the cluster adds each round onto the last good
  merge: `C_{N+1} = C_N + (1/α)·Σ ŵ R`, where `R` is each edge's residual and
  `ŵ` its normalised FedProx weight. Edges train with the FedProx proximal
  term, μ = 0.01 (`μ = 0` disables it and the manifest says so).
- **Reason:** user decision of 2026-10-14, "cluster adds round N's learning on
  top (FedProx-exact)", which departs from D2/D3 as literally written.

## D6 — where the composite is merged (user decision U1)

- **As written:** composition happens offline at promotion time; the registry
  stores one pre-merged composite per developer and inference loads that single
  adapter.
- **As built:** the composite is merged **on the edge**, at pull time. The edge
  pulls the merged cluster adapter from the cluster
  (`GET /clusters/{id}/adapters/active`) and merges it with its own client
  layer locally. The registry's `/compose` path and the `composite` block of
  `/promote` exist, are tested, and are not called by the live loop.
- **Reason:** U1 in `2026-11-14-panel-completion-decisions.md`:
  keep merging on the edge, so the developer's client layer never leaves the
  edge, and describe the registry's composite code as present, tested and
  unused.

## D4 — re-clustering (user decision U7)

- **As written:** clients are re-clustered dynamically during the federation, by
  the similarity of their updates.
- **As built:** rounds 1–2 use the project grouping. After the final round the
  cluster computes k-means (k = 2) on the cosine similarity of the clients'
  updates and reports the grouping it would choose. Applying it is the start of a
  new run.
- **How it is shown:** the demo panel has a "Re-cluster (dry run)" button,
  enabled once every cluster with members has merged round 2. It calls the
  panel's `POST /api/live/recluster`, which calls the cluster's `POST /recluster`
  with `apply: false` whatever the browser sends, and shows the current and the
  proposed membership side by side, the clients that would move, the similarity
  matrix, and the cluster's own sentence when it falls back to the project
  grouping (too few updates, or groups not separated enough). Nothing moves.
- **Why not apply it inside a run:** a client that changes cluster starts
  training on a different cluster adapter; that is a new run, and a new run has a
  new privacy budget (D7). The cluster's `apply: true` path exists and is tested,
  and nothing in the demo calls it.
- **Who may call it:** the panel route needs the `operator` login; the cluster's
  `/recluster` route is for the `svc:evaluation` and `admin` roles.
- **Reason:** U7 in the panel-completion decisions.

## D7 — DP unit (user decision U2)

- **As written:** client-level DP (the privacy unit is a client's whole
  participation in a round).
- **As built:** DP is **example-level, applied on each client**: Opacus DP-SGD
  over 1024- or 512-token blocks, on the LoRA parameters only. Nothing is
  clipped or noised at the client-participation level.
- **Consequence:** this is weaker than client-level DP, because one developer
  contributes many blocks, so the guarantee covers a single block, not a
  developer's whole contribution.
- **Reason:** U2 in the panel-completion decisions: reword to "example-level DP applied on
  each client"; no mechanism change. Client-level DP was rejected for this
  project.

## D7 — the privacy budget (user decision U3)

- **As written:** epsilon at most 8 (delta = 1e-5) per run.
- **As built until 2026-11-16:** every round was calibrated to epsilon 8 on its
  own, and the cluster added a round's epsilon to the adapter it built on, so a
  two-round run was recorded at about 16.
- **As built now:** epsilon at most 8 at delta = 1e-5 over the **whole run of 2
  rounds, per client, example-level**. The rounds are composed with the Opacus
  (PRV) accountant, never added up. The budget is charged when an adapter is
  **uploaded**, and it is enforced in two places and recorded in a third:
  - the **edge** keeps a ledger per client and run
    (`security.dp.ledger.BudgetLedger`, under
    `services/edge/artifacts/privacy_ledger/<client>/<run>.json`) and refuses,
    before any network call, an upload that would be a third round or would take
    the run past 8;
  - the **cluster** refuses (422) a differentially private upload that does not
    report the run's cumulative epsilon, reports more than 8, or reports more
    than 2 rounds; its own epsilon for a merge is the largest cumulative epsilon
    among the clients in it;
  - the **registry** records each client's cumulative epsilon with the version
    that contains its round (`GET /privacy/ledger`), and the demo panel shows it.
- **How the noise is set:** both rounds train with the same noise multiplier,
  the smallest for which two rounds of that size compose to at most 8
  (`security.dp.ledger.calibrate_run_sigma`). One round alone therefore spends
  less than 8.
- **Consequence, stated plainly:** an uploaded round whose merge is later
  **rejected** has still spent its share of the budget, because the update left
  the laptop. Training that is never uploaded costs nothing. A new run
  (`--run-id`) starts a new budget.
- **Open, needs a decision:** the budget is enforced against the run the laptop
  declares. A new `--run-id` on the same promoted adapter is a new budget for the
  client while the adapter keeps the earlier rounds; the cluster cannot tell
  (`docs/dp_config.md`, "What is not enforced").
- **Not yet measured:** whether the adapter still learns at the noise two rounds
  within epsilon 8 require. That needs a GPU round and is recorded in
  `experiments/d3-round2-dp/RESULTS.md` when it has been run.
- **Reason:** U3 and calls C8, C9 in the panel-completion decisions.

## D7 — transport, as built (user decisions U4, U5, U6)

- **As written:** mTLS on all service channels.
- **As built until 2026-11-16:** mTLS was an option (an overlay compose file and a
  `--mtls-dir` flag); the default stack ran plain HTTP. The security library's
  roles, HMAC, JWT, middleware and audit code existed and was not wired in.
- **As built now:**
  - **mTLS is the default.** `docker compose up` and `scripts/run_services.py`
    start the registry and the cluster behind mutual TLS 1.3 with certificates
    from `.runtime/certs`, generated when missing. Plain HTTP is `--insecure` or
    the `docker-compose.insecure.yml` overlay. A service started by hand without
    certificates still runs plain HTTP with a warning.
  - **Services are identified by certificate, people by login (U5).** Certificate
    names map to roles and every registry and cluster route names the roles
    allowed to call it. The live panel, the dashboard and the edge page are
    served over HTTPS with a server certificate; the panel's operator and an edge
    page's developer log in with a JWT that lasts 8 hours.
  - **HMAC is used twice (U4):** on evaluation results sent to `/promote`, signed
    by the evaluator and tied to the candidate's sha256; and on edge uploads,
    signed by the edge and verified by the cluster.
  - **Wired in from the library (U4):** security headers, a request-size limit, a
    rate limit and the audit logger on the registry, the cluster and the panel;
    the registry also keeps an access log. API keys stay in the library, unused.
  - **Fall-back (U6, C11):** a check that is not configured is skipped with a
    warning; one that is configured and fails is refused.
- **Not done here:** the Docker bring-up of the default-mTLS stack and a
  three-laptop round over it need Docker Desktop and the laptops; see
  `docs/mtls_setup.md` for what to check.
- **Reason:** U4, U5, U6 and calls C1–C3, C10, C11 in the panel-completion decisions.

## D5 — promotion guard

- **As written:** the guard is HumanEval pass@1: promote only if the drop is
  at most 2 points.
- **As built:** the guard is next-line completion on a frozen 300-example
  general set (click, jinja, colorama; none of them trains any cluster),
  scored as edit similarity with a tolerance of 0.02. Measured noise of the
  guard: pending measurement, see
  `services/evaluation/results/line_guard/noise.json`. HumanEval and PY150
  (`scripts/eval_py150.py`) are reported only and never gate a promotion.
- **Reason:** HumanEval tests docstring-to-function synthesis, not CLASP's
  in-project completion, and on the 20-problem subset the project uses one
  problem is 5 points, so a 2-point tolerance is below its resolution.
- **Promotion flow, as written:** a new version became active on upload and a
  failed D5 check rolled `active` back to the previous version.
- **Promotion flow, as built:** a published version is a candidate; promotion
  makes it live; a rejected candidate is never served. Uploads are inactive;
  the registry judges the candidate (newer than the active version) against
  the active version, or against the bare base model when nothing is active.
  PROMOTE makes it active; REJECT records the decision and leaves `active`
  where it was. The cluster holds its aggregate as a candidate and serves it
  to the edges only after `candidate/promote`; `candidate/reject` drops it.
  An operator restore never returns to a refused version (a REJECT, or a
  D5 `rollback` recorded before the candidate flow): the registry's
  `/restore` answers 409, and so does the cluster's
  `/clusters/{id}/adapters/active/restore` (`rejected_version`), which checks
  the registry's `rejected` list before fetching the version.
- **Reason:** with auto-activation every edge could pull a merge in the
  window between upload and the D5 decision, so a candidate that later failed
  was briefly served and trained on.

## Base model

- **As written:** DeepSeek-Coder-6.7B.
- **As built:** every Phase II run uses the 1.3B profile.
- **Reason:** the team's GPUs have 4 GB of VRAM, so the 6.7B model does not fit.

## Cluster transport

- **As written:** Flower orchestrates the cluster.
- **As built:** the live cluster is a FastAPI service. Flower is used in
  `services/cluster/src/cluster/simulation.py` and the experiment code only.
- **Reason:** the live loop is edge-initiated HTTP upload and pull against one
  cluster service; Flower's round protocol was kept for simulation and
  experiments.

## Evaluation

- **As written:** an evaluation service.
- **As built:** a library plus scripts (`scripts/evaluate_candidate.py`), not a
  long-running service. The demo panel (`demo_ui/server.py`) runs the script and
  posts the result to the registry's `/promote`.
- **Reason:** evaluation is run on demand per candidate; nothing needs to stay
  resident.
