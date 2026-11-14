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
  top (FedProx-exact)", which departs from D2/D3 as literally written. Recorded
  in `docs/superpowers/plans/2026-10-14-federated-loop-audit-fedprox.md`
  (Spec line and Global Constraints).

## D6 — where the composite is merged (user decision U1)

- **As written:** composition happens offline at promotion time; the registry
  stores one pre-merged composite per developer and inference loads that single
  adapter.
- **As built:** the composite is merged **on the edge**, at pull time. The edge
  pulls the merged cluster adapter from the cluster
  (`GET /clusters/{id}/adapters/active`) and merges it with its own client
  layer locally. The registry's `/compose` path and the `composite` block of
  `/promote` exist, are tested, and are not called by the live loop.
- **Reason:** U1 in `docs/superpowers/specs/2026-11-14-panel-completion-design.md`:
  keep merging on the edge, so the developer's client layer never leaves the
  edge, and describe the registry's composite code as present, tested and
  unused.

## D7 — DP unit (user decision U2)

- **As written:** client-level DP (the privacy unit is a client's whole
  participation in a round).
- **As built:** DP is **example-level, applied on each client**: Opacus DP-SGD
  over 1024- or 512-token blocks, on the LoRA parameters only. Nothing is
  clipped or noised at the client-participation level.
- **Consequence:** this is weaker than client-level DP, because one developer
  contributes many blocks, so the guarantee covers a single block, not a
  developer's whole contribution.
- **Reason:** U2 in the design spec: reword to "example-level DP applied on
  each client"; no mechanism change. Client-level DP was rejected for this
  project.

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
  An operator restore (`/restore`) can return to an earlier version that was
  live, never to a rejected one.
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
