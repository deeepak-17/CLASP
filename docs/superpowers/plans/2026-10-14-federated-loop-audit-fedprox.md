# Federated Loop Audit + FedProx Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (chosen: Native, one whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the real 3-laptop federated round (B: web, C: scientific, A: cluster + registry + evaluation) works end to end with no simulated hops or placeholders, and make it FedProx-correct: edges train with the proximal term, the cluster's round N+1 adapter builds on round N's.

**Architecture:** Audit first (tests, static trace of every hop, placeholder sweep). Then three known fixes: an edge-side FedProx term, an upload that declares which merged adapter it trained on, and a cluster-side cumulative FedProx server step. After that, fixes for whatever the audit finds, appended to this plan as tasks 7+. Last, a live loop on laptop A over its LAN IP with real GPU training (two rounds, both clusters), then a whole-branch review, then push.

**Tech Stack:** Python 3.10 `.venv` (torch 2.14+cu126, opacus 1.6, peft 0.20, flwr 1.30), FastAPI services, pytest, ruff 0.15.20.

**Spec:** the user's five goals (2026-10-14 chat) + `MASTER_PLAN.md` D2/D3/D5/D7 + user decision 2026-10-14: *cluster adds round N's learning on top (FedProx-exact)* — departs from D2/D3 as literally written.

## Global Constraints

- Branch `integration/demo`, repo `CLASP_Team_Bundle/CLASP`. Commits authored as the user; no Claude attribution (memory: commit-attribution-user-only). Push only after review + fixes (user, 2026-10-14).
- All four services and the 4 uncommitted `services/evaluation` files are in scope.
- Edge imports other services, never copies their logic. FedProx's proximal term is client-side by D2 ("FedProx proximal term stays client-side"), so it is edge code.
- CI is CPU-only: GPU tests `skipif(not torch.cuda.is_available())`; `ruff check .` must be clean for the whole repo.
- LoRA r=16, lora_alpha=16, q/k/v/o; D3 composition α default 0.5; D7 ε ≤ 8, δ = 1e-5.
- FedProx μ default **0.01** (the value `cluster.client.LoRAClient` uses); `μ = 0` disables it and the manifest says so.
- Test command per package: `.venv/Scripts/python.exe -m pytest <pkg>/tests -q`.

## Review Focus

1. Upload trained on a merged adapter from a **previous cluster process** (in-memory state reset, same round number) → refused 409 by digest, never folded in as if current.
2. A round mixing uploads that declare a base with ones that don't, or different α → 422 at upload time, not a wrong aggregate later.
3. FedProx under DP: the proximal gradient is added **after** Opacus noise (step hook) so ε is unchanged; under non-DP it is added before clipping.
4. Pull-back then train then upload on a laptop whose page restarted → the base is read from `pull_manifest.json`, not page memory.
5. `/recluster` after a cumulative round uses each edge's residual as its update (start = None), not residual − cluster.

---

### Task 1: Baseline evidence (no code changes)

**Files:** read-only; output goes into this plan's "Audit findings" section.

- [ ] Run every package's tests + `tests/integration` + `tests/e2e`, and `ruff check .`. Record pass/fail counts.
- [ ] `git diff services/evaluation` (the 4 uncommitted files): what wrote them, are they real outputs of a run, and should they be committed, regenerated or reverted?

### Task 2: Static trace + placeholder sweep (goals 1, 2, 4, 5)

- [ ] Trace each hop, file:line: edge page → `train_client` → `upload` → cluster `/clusters/{id}/uploads` → aggregate → publish → registry → evaluation evidence → promote → `pull_cluster_adapter` → next train.
- [ ] Sweep for synthetic/simulated/placeholder/hardcoded paths (`demo_ui` `/api/demo/*`, `random`, `synthetic`, `placeholder`, `TODO`, `dummy`, constants posing as metrics) and mark each **reachable from the demo** or **dead/legacy**.
- [ ] Security: mTLS wiring (cluster, registry, edge session, demo_ui client), identity binding, DP enforcement on the page, registry validation of uploads.
- [ ] Registry: versioning, sha256, promote/rollback rule, lineage, metadata from the cluster publish.
- [ ] Evaluation: what `demo_ui/evidence.py` feeds Promote, and whether each number traces to a real run.
- [ ] Write findings into "Audit findings" below, each tagged defect / limitation / OK.

### Task 3: Edge trains with FedProx

**Files:**
- Create: `services/edge/src/edge/fedprox.py`
- Modify: `services/edge/src/edge/train_client.py` (`train`, `train_dp`, `main`, manifest)
- Test: `services/edge/tests/test_fedprox.py`

**Interfaces:**
- Produces: `snapshot_anchor(params: list[torch.nn.Parameter]) -> list[torch.Tensor]`; `add_proximal_grad(params, anchors, mu: float) -> float` (adds `mu*(p - p0)` to `p.grad`, returns `0.5*mu*Σ‖p-p0‖²`); `train(..., fedprox_mu: float = 0.0)`, `train_dp(..., fedprox_mu: float = 0.0)`; CLI `--fedprox-mu` (default 0.01); manifest `"fedprox": {"mu", "anchor": "client adapter at round start", "applied": "pre-clip" | "post-noise (Opacus step hook)", "final_prox_term"}`.

- [ ] Failing tests: (a) `add_proximal_grad` adds exactly `mu*(p-p0)` to existing grads and returns the half-squared term; mu=0 is a no-op; (b) on a tiny CPU torch model, `train`-style steps with mu=1.0 end closer to the anchor than mu=0 (same seed); (c) Opacus on a tiny CPU model: with the hook installed, `p.grad` seen by the inner optimizer equals noised grad + `mu*(p-p0)` (prove by capturing in a wrapped `original_optimizer.step`); (d) argparse default is 0.01.
- [ ] Implement: non-DP path calls `add_proximal_grad` after the accumulation window's last backward, before `clip_grad_norm_`. DP path calls it inside the existing `_count_step` hook (runs after `add_noise`/`scale_grad`, before the inner step: opacus 1.6 `pre_step` lines 554–561). Anchor = trainable params snapshotted right before the optimizer is built.
- [ ] Run edge tests; ruff. Commit `edge: train with the FedProx proximal term (mu 0.01 default)`.

### Task 4: Edge declares the merged adapter it trained on

**Files:**
- Modify: `services/edge/src/edge/upload.py` (`pull_cluster_adapter` records `sha256`; `upload_adapter` sends the base)
- Modify: `services/edge/src/edge/wire.py` (`upload_payload(..., base: Optional[Dict] = None)`)
- Modify: `services/edge/src/edge/train_client.py` (manifest `d3.cluster_pull` = the pull manifest when the cluster adapter came from the cluster)
- Test: `services/edge/tests/test_upload.py`, `test_wire.py`

**Interfaces:**
- Consumes: Task 5's broadcast field `sha256` and upload fields below.
- Produces upload body fields: `base_cluster_id: str`, `base_round_id: int`, `base_sha256: str`, `composition_alpha: float`. Sent iff the training manifest's `d3.cluster_pull` exists. A D3 adapter whose frozen layer did **not** come from a cluster pull (registry-materialized) is refused by `upload_adapter` with a hint to train on the pulled adapter or on the bare base; `--allow-foreign-base` overrides for experiments, sending no base.

- [ ] Failing tests: payload carries the four fields from a manifest with `d3.cluster_pull`; none without `d3`; a registry-based D3 manifest raises `UploadError`; `pull_cluster_adapter` writes `sha256` from the broadcast into `pull_manifest.json`.
- [ ] Implement, run edge tests, ruff, commit `edge: uploads name the merged adapter they trained on`.

### Task 5: Cluster FedProx server step (cumulative) + stale check + recluster

**Files:**
- Modify: `services/cluster/src/cluster/schemas/messages.py` (`AdapterUpload`: optional `base_round_id: int|None`, `base_sha256: str|None`, `composition_alpha: float|None (gt 0)`; `ClusterAdapterBroadcast`: optional `sha256: str|None`)
- Modify: `services/cluster/src/cluster/aggregation.py` (`accumulate_svd_lowrank(start, residuals, weights, alpha, rank) -> tuple[LoRAAdapter, dict]`)
- Modify: `services/cluster/src/cluster/server.py` (`_upload_locked`, `_aggregate_locked`, `_StoredUpload`, `_peft_bytes` use, publish meta, stale docstring at :390)
- Test: `services/cluster/tests/test_fedprox_server.py`

**Rules:**
- `accumulate_svd_lowrank` = `C + (1/α)·Σ ŵ_k R_k` (ŵ normalised), computed as `aggregate_svd_lowrank([C, R_1..R_K], [1, ŵ_1/α, ...])` with `lora_B` rescaled by `1 + 1/α`. Truncation error is relative, so it is unaffected.
- Upload with a base: the cluster must have an active adapter whose sha256 == `base_sha256`, and `base_cluster_id` == this cluster; otherwise 409 `stale_adapter` (hint: pull and retrain). An upload without a base while the cluster has an active adapter: 409 (it did not start from the global model). An upload with a base while there is no active adapter: 409.
- One round = all uploads with a base or none, one α: else 422.
- Aggregate: base round → `accumulate_svd_lowrank(active, ...)`, manifest `update_rule: "fedprox_cumulative"`, `base_sha256`, `composition_alpha`; no-base round → unchanged (`update_rule: "average"`). `naive` with a base round → 422. `last_round` start = `None` for base rounds (the residual is the update).
- `sha256` = sha256 of `_peft_bytes(broadcast)` (what publish sends and the registry hashes), set on the broadcast at aggregate time; publish meta gets `update_rule`, `composition_alpha`, `base_sha256`.

- [ ] Failing tests: math (`accumulate_svd_lowrank` ΔW equals `C + Σŵ R/α` exactly at full rank; equals plain average when C = 0); HTTP round 0 → round 1 cumulative ΔW; stale digest 409; missing base 409; mixed 422; different α 422; recluster updates equal residuals; broadcast `sha256` equals sha256 of `/adapters/{id}/download` bytes.
- [ ] Implement, run cluster tests + `tests/integration` + `tests/e2e`, ruff, commit `cluster: FedProx server step builds on the previous merge`.

### Task 6: Edge page + bundle carry FedProx and the base

**Files:**
- Modify: `services/edge/src/edge/webui.py` (`train_command`: `--fedprox-mu`, `--seq-len 512` under DP, measured 4.447 GB peak at 1024 on a 4 GB card), `webui_page.html` (show μ and the base round on adapter cards)
- Modify: `scripts/make_edge_bundle.py` (live fine-tune steps: install `security`, `--dp`, `--seq-len 512`, page-style `live-<ts>` sets)
- Test: `services/edge/tests/test_webui.py`

- [ ] Failing tests for the command line and the card fields; implement; run; commit.

### Task 5b: Rollback reaches the cluster (and so the edges)

**Files:**
- Modify: `services/cluster/src/cluster/server.py` (`POST /clusters/{id}/adapters/active/restore`; `GET /clusters` lists `active_sha256`)
- Modify: `demo_ui/server.py` (`live_promote`: on ROLLBACK, call the restore with the registry's restored version)
- Modify: `services/edge/src/edge/webui_page.html` (auto-pull when the cluster's `active_sha256` differs from the pulled one, not only when the round number grows)
- Test: `services/cluster/tests/test_fedprox_server.py`, `tests/test_demo_ui_live.py` (new, under repo `tests/integration`)

**Rules:** restore body `{registry_url, adapter_name, version}`; the cluster downloads `/adapters/{name}/versions/{v}/file`, checks sha256 against `/adapters/{name}/versions/{v}` metadata, rebuilds the broadcast from the header (config + provenance fields), re-serializes and refuses (502) unless the bytes hash to the same sha256, then sets `active`. Upload buffer and round counter untouched.

- [ ] Failing tests: restore sets active to the registry version and `/clusters` shows its sha; tampered blob refused; a base declared on the rejected adapter is then 409.
- [ ] Implement; run cluster + integration tests; commit `cluster: a registry rollback resets the adapter the edges pull`.

### Task 7: Remove placeholder / synthetic demo paths (goal 4)

**Files:** `demo_ui/server.py` (delete `/api/demo/*`, `/api/edge/*`, `_synthetic_tensors_for`, `CLUSTER_OF`-only helpers they need), `demo_ui/static/page{1,2,3}.html` (delete), `demo_ui/static/live.html` (drop the "Earlier panel pages" link), `demo_ui/README.md`, `docs/multi-laptop-demo.md`, `services/cluster/tests/test_server_routes_and_epsilon.py` if it references them.

- [ ] Test first: `tests/integration/test_demo_ui_live.py::test_no_synthetic_routes` asserts every `/api/demo/*` path is 404 and `page1.html` is 404.
- [ ] Delete; run; commit `demo_ui: drop the synthetic panel pages and their placeholder routes`.

### Task 8: Evaluate the live candidate on laptop A (goal 5)

**Files:**
- Create: `scripts/evaluate_candidate.py`: for one cluster, materialize registry vN (candidate) and vN−1 (baseline) via `edge.registry_client.RegistryClient.materialize`, compose `base + α·cluster` with `edge.merge.compose`, score each with `edge.completion_eval.evaluate_completion` on the noise-band client's held-out examples (the same client `demo_ui.evidence.noise_band` names), generate the 20 frozen HumanEval completions per version (`edge.humaneval_baseline.generate_completions`) and score them with `evaluation.eval_harness.guard_anchor.score_samples`/`build_anchor`. Writes `.runtime/live-eval/<cluster>-v<N>.json` in the `CLASP_EVAL_EVIDENCE` shape plus `candidate_version`, `candidate_sha256`, `baseline_version`, `baseline_sha256`.
- Modify: `demo_ui/evidence.py` (`evidence_for(cluster_id, version=None, sha256=None)`: prefer a live-eval file whose version+sha match the candidate; `stale=False` then), `demo_ui/server.py` (`POST /api/live/evaluate/{cluster}` starts the script as a subprocess job, `GET /api/live/evaluate/{cluster}` streams its log/state; promote passes version+sha), `demo_ui/static/live.html` (Evaluate step between Publish and Promote)
- Modify: `services/edge/src/edge/humaneval_baseline.py` only if generation needs an `--adapter` path (it takes a model, so likely no change).
- Test: `tests/integration/test_demo_ui_live.py` (evidence selection by version/sha, stale fallback marked), GPU-gated smoke for the script.

- [ ] Failing tests for evidence selection; implement; GPU run in Task N-1; commit `demo: evaluate the published candidate itself before Promote`.

### Task 9: Security library does its part at the cluster (goal 5)

**Files:** `services/cluster/src/cluster/server.py` (upload: `security.validation.validate_adapter_integrity` → 422 with its errors; aggregate: `security.validation.detect_poisoning` over the round's uploads → flagged clients go to `skipped_clients` as `"poisoning_suspect"`), `services/cluster/pyproject.toml` (depend on `clasp-security`), `.github/workflows/ci.yml` (install `security` before the cluster/integration/e2e legs).

**Ruling needed at task start:** `import security.validation` pulls `security/__init__` → opacus/torch. Check cost in CI; if the cluster leg cannot take torch, import `security.validation.adapter_validator` lazily and fail closed at service start (`cluster.serve` refuses to start without it).

- [ ] Failing tests: NaN upload → 422; 3+ uploads with one 1000× norm → that client skipped and named; normal round unchanged.
- [ ] Implement; run; commit `cluster: validate uploads and screen rounds with the security library`.

### Task 10: Evaluation runs on the Windows laptops (do before Task 8: Task 8 scores HumanEval on laptop A)

**Files:** `services/evaluation/src/evaluation/eval_harness/execution.py` (sandbox env keeps `SYSTEMROOT` on Windows; tree kill via `taskkill /T /F` where there are no process groups), `utils/paths.py` (`relative()` POSIX), `eval_harness/archive.py` (manifest keys POSIX), `utils/io_utils.py` (Paths serialise POSIX), `eval_harness/dependencies.py` (`MIN_PYTHON = (3, 10)` = pyproject), tests: `test_utils.py::test_resolve_path_preserves_absolute` (use a real absolute path), `test_interfaces.py::test_registering_makes_the_edge_backend_reachable` (stub the loader instead of loading 6.7B). Revert the 4 regenerated result files (only timestamp + separators differ).
- [ ] RED already exists (24 failing on Windows); fix each root cause; evaluation suite fully green on this laptop; commit per cause group.

### Task N-1: Live loop on laptop A (real HTTP over the LAN IP, real GPU training)

- [ ] `run_services.py --ui --background` (binds 0.0.0.0). Register clusters through demo_ui's real route.
- [ ] Round 0: short DP training on the bare base for `web/client-requests` and `scientific/client-numpy` (this laptop's GPU), upload both through `edge.webui` instances (ports 8020/8021) dialing the **LAN IP**, aggregate both clusters, publish v-next.
- [ ] Round 1: pull back on each page, train with FedProx on the pulled adapter, upload (base declared), aggregate → `update_rule: fedprox_cumulative`, publish, promote → registry decision. Then an out-of-date upload → 409.
- [ ] Record every response and timing in "Live evidence".

### Task N: Whole-branch review, fixes, push

- [ ] Dispatch one reviewer subagent (requesting-code-review template) over `origin/integration/demo..HEAD`.
- [ ] Fix Critical/Important; rerun all suites + ruff.
- [ ] `git push origin integration/demo`. B/C then `git pull` and run the 3-laptop round (user).

---

## Audit findings

**Baseline (75c90e7, laptop A, `.venv` py3.10):** contracts 22 ✓ · edge 161 ✓ 1 skip · cluster 234 ✓ 2 skip · registry 203 ✓ · integration 28 ✓ · e2e 1 ✓ · ruff ✓. security: collection error (PyJWT missing from `.venv`, declared in pyproject: env gap, installed). evaluation: `jsonschema` missing (env gap, installed), then 24 failures + 1 test that loads the 6.7B model (looks like a hang).

Goal 1 — real connections B/C ↔ A
- OK: edge page → `edge.upload` → `POST /clusters/{id}/uploads` over the URL given (LAN IP); `run_services.py` binds 0.0.0.0; pull-back over the same session. No simulated hop in the live path.
- Limitation (team decision Q3): plain HTTP by default; then `client_id` is self-asserted. Under `--mtls-dir` uploads are bound to the cert CN (`cluster.serve`).
- Legacy-only: `scripts/demo_local.py` (one-laptop fallback, loopback by design), cluster `simulation.py`/`demo*.py` (Flower experiments). Not on the 3-laptop path.

Goal 2 — edge UI + backend
- OK: page lists DP adapters only, trains DP, uploads, tracks 5 steps, pulls back.
- Defect: training had no FedProx term (fixed, Task 3).
- Defect: page auto-pull compares round numbers only; a restarted cluster (or a rollback) serving a different adapter with a round number ≤ the pulled one is never pulled (Task 5b/6, digest compare).
- Defect: bundle's live fine-tune instructions omit `security` (needed for `--dp`), omit `--dp`, write to `live/` not `live-<ts>` (Task 6).
- Finding: DP at seq 1024 peaks 4.447 GB on a 4 GB card (Laptop C, 6.19 s/step) → page should pass `--seq-len 512` under DP (Task 6).

Goal 3 — merge + return
- Defect: cluster replaced C_N with avg(R) (round N lost); user chose FedProx-exact (fixed, Task 5). Edge never declared its base (fixed, Task 4). `/recluster` subtracted C_N from residuals (fixed, Task 5).
- Defect: a D5 ROLLBACK repoints the registry's `active`, but the cluster keeps serving the rejected adapter, so the edges pull and train on it (Task 5b).

Goal 4 — placeholders
- Defect: `demo_ui` `/api/demo/send-to-cluster` uploads **synthetic tensors**; `/api/demo/promote` sends **hardcoded** edit-similarity 0.5/0.45, HumanEval 0.30/0.31; `/api/demo/demonstrate-rollback` posts a **made-up regressed candidate**. Reached from live.html's "Earlier panel pages" link (pages 1–3). (Task 7)

Goal 5 — registry / evaluation / security
- Registry OK: immutable versions, sha256, D5 two-sided promote, rollback cascade, restore, GC, audit trail, mTLS launcher. Ignores unknown publish metadata (contracts v1.0 frozen) → FedProx provenance rides in the safetensors header (Task 5).
- Defect: live Promote's evidence is **stale**: measured on round 1's aggregate (2026-09-06), never on the published candidate; candidate HumanEval guard never scored → every decision is ROLLBACK by construction. All parts to score the real candidate exist (Task 8).
- Defect (Windows): evaluation's sandbox passes an env without `SYSTEMROOT`, so every program dies at interpreter start on Windows ("failed to get random numbers") → HumanEval pass@1 0.0 on laptop A; process-tree kill no-op on Windows; `paths.relative`/`archive` keys/`write_json` write `\` separators (the 4 dirty result files differ only by timestamp + separators); `MIN_PYTHON = (3, 11)` contradicts `requires-python >=3.10` and every laptop's 3.10. Test that eagerly loads the 6.7B model. (Task 10)
- Defect: security library's adapter validation (`validate_adapter_integrity`, `detect_poisoning`) is used by no service, so uploads with NaN/Inf/absurd norms reach the aggregate (Task 9). Middleware/auth/audit unused (report only: not in scope of the loop).
- Finding (D7): per-round DP at ε 8 plus cumulative composition → a 2-round run spends up to 16 > D7's 8 per federated run. Now reported honestly (cluster epsilon composes); budget split across rounds is a policy decision for the team.

## Live evidence

(filled by Task N-1)
