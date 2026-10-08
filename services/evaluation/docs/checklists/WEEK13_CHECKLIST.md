# CLASP-P5 · Week 13 Completion Checklist

**Scope:** P5 Week-13 targets (`CLASP_Daily_Targets_v2.pdf`, Phase II — debt, coverage, isolation stress).

**Note:** written on 2026-10-08, after the Phase II calendar, while closing out P5's remaining deliverables. Items marked `[x]` cite the file that proves them; team events (panel reviews, all-hands runs, rehearsals) are not code deliverables and are left unticked unless the repo records them.

## Mon: Regression suite complete in CI
- [x] 486 P5 tests run in CI (was 1 smoke test before PR #20)

## Tue: Eval determinism (seeded sampling)
- [x] `generation.seed` + per-task `task_seed`; seeded in the transformers adapter and the mock; recorded in results.json (`tests/test_determinism.py`)

## Wed: Dashboard error states
- [x] Every page distinguishes loading / malformed (names the guard) / not-produced-yet (names the script) — `src/lib/useFeed.ts`, `components/DataState.tsx`

## Thu–Fri
- [x] Contract tests run the committed feeds through the dashboard's runtime guards (`dashboard/tests/feeds.test.ts`)
