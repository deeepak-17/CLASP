# P4 (State Registry) — Rubric Evidence Checklist

> Per-criterion evidence for `docs/rubric.md`'s Panel Review 1 sheet, scoped to
> P4's module (`services/registry/`). Cites concrete repo paths / test results
> per the rubric generator's own instruction — no unverified assertions.
> Owner: Deepak (P4). Last verified: 2026-08-13 (Week 4 → Week 5 handoff).

## 1. Well-defined Requirements & Feasibility

- **Invariants stated up front, not implied**: [`storage.py:14-20`](../../services/registry/src/registry/storage.py#L14-L20)
  docstring lists the registry's non-negotiables — safetensors never
  overwritten in place, every payload sha256'd for audit.
- **NFR made measurable**: "writes must be atomic" isn't a vague goal — it's
  proven by [`test_storage.py::test_save_leaves_no_orphan_on_crash_between_writes`](../../services/registry/tests/test_storage.py)
  and [`test_save_publishes_both_files_together`](../../services/registry/tests/test_storage.py),
  which simulate the crash window directly rather than trusting the
  implementation by inspection.
- **Feasibility**: registry is stdlib-only (no torch/numpy/safetensors
  dependency — [`storage.py:19-20`](../../services/registry/src/registry/storage.py#L19-L20)),
  keeping it viable on the single-GPU-constrained dev box (D8) since it never
  competes for GPU/CUDA resources.
- **Gap**: no formal SRS doc for the registry specifically — its requirements
  live as docstrings + `MASTER_PLAN.md` D5/D9, not a standalone spec. Acceptable
  for a single-owner module but flag if the panel wants a dedicated SRS artifact.

## 2. Architecture Validation

- **D5 (two-sided promotion) traceable to code**: [`promotion.py:1-13`](../../services/registry/src/registry/promotion.py#L1-L13)
  docstring quotes the D5 rule verbatim and cites it inline; `decide()` is a
  pure function (no filesystem access) specifically so the rule can be unit
  tested apart from the HTTP/storage layers — see `test_promotion.py`.
- **D9 (immutable versions) traceable to code**: [`storage.py:195-227`](../../services/registry/src/registry/storage.py#L195)
  `save()` — atomic stage-then-`os.rename` publish, `VersionExists` on
  collision instead of silent overwrite.
- **Never gates on pass@k alone (D5 explicit anti-pattern)**: enforced at
  [`promotion.py:34-46`](../../services/registry/src/registry/promotion.py#L34-L46)
  `_guard_holds()` — guard is one of *two* required conditions, checked
  alongside `_in_project_improved()`.
- **Gap**: D6 (composite-adapter storage, offline α/β merge at promotion time)
  is architecturally described in `docs/architecture.md` §1 but not yet
  implemented — correctly scheduled for W8, not a current gap.

## 3. Comprehensive Architecture & System Flow

- **Module boundary + ownership**: `docs/architecture.md` line 31 names P4 as
  sole owner of `services/registry/`, stateful-container flagged explicitly.
- **Contracts seam used, not bypassed**: registry imports `AdapterRef`,
  `AdapterMetadata`, `EvalResult`, `GuardMetrics`, `PromotionDecision` from
  `contracts` rather than redefining its own shapes —
  [`app.py:22-33`](../../services/registry/src/registry/app.py#L22-L33).
- **One federated round's Registry leg documented end-to-end**:
  `docs/architecture.md` §3, steps 3–5 (Cluster→Registry snapshot,
  Registry→Evaluation handoff, D5 promotion).
- **Endpoint contract fully documented**: `docs/registry_api.md` — table of
  all 9 endpoints, request/response shapes, and a runnable walkthrough
  ([`demo/save_promote_rollback_demo.sh`](../../services/registry/demo/save_promote_rollback_demo.sh)).

## 4. Process Audit (CI/CD, Dockerization)

- **CI**: registry is one leg of the 6-package `ruff` + `pytest` matrix
  (`.github/workflows/ci.yml`); `ruff check .` on `services/registry` — **0
  issues** (verified this session).
- **Test suite green**: `pytest services/registry/tests` — **37 passed**
  (verified this session; includes the two new atomicity tests above).
- **Dockerized**: [`services/registry/Dockerfile`](../../services/registry/Dockerfile)
  — built and started clean via `docker compose up -d --build registry`
  (verified this session: image built, container reached `healthy`,
  `GET /healthz` returned `200` from the host).
- **Stateful volume actually verified, not just claimed**: `docker-compose.yml`
  comments assert `registry-data` "must survive restarts" (D9) — confirmed by
  running `docker compose down` (no `-v`) then `up` again and re-querying
  `/adapters/demo-django/versions`: all 3 versions and the correct active
  pointer (`v2`, post-rollback) were intact after the restart.

## 5. Works Completed & Plan

- **Save / load / list / promote / rollback all live with tests**: 9 endpoints
  in `app.py`, all covered — `test_api.py`, `test_promote_api.py`,
  `test_storage.py`, `test_promotion.py` (37 tests total, 0 failing).
- **Demoable in isolation**: `docker compose up -d --build registry` +
  `demo/save_promote_rollback_demo.sh` runs the full save→save→promote(good)
  →save→promote(bad→rollback)→lineage→audit-trail cycle against a real
  container with no manual steps — dry-run executed this session, passed
  clean on the first attempt (no gaps to fix).
- **Credible forward plan**: `daily_targets_v2.md` Weeks 6–16 lay out P4's
  path from here (G1 mTLS/live channels W7, D6 composite storage W8, G3 full
  E2E round W9, retention/GC W10) — consistent with `MASTER_PLAN.md` D12 gate
  structure.
- **Known, scheduled gaps (not blockers for Panel 1)**: mTLS (D7, W10),
  composite-adapter storage (D6, W8), retention/GC policy (W10) — all
  correctly listed under "Not yet" in `docs/registry_api.md`, none are Week
  4/5 commitments.
