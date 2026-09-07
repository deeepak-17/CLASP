# CLASP-P5 · Week 5 Completion Checklist

**Scope:** P5 Week-5 daily targets (`CLASP_Daily_Targets.pdf`, Phase II, "Lock in deliverables for Panel Review 1: every module should be demoable in isolation"). Weeks 1–4 functionality is frozen baseline and untouched.

**Note:** this checklist was written on 2026-08-12, after the Week-5 work itself was already done, to bring Week 5 in line with the Week 1–4 checklist convention. Every item below was re-verified on that date (dashboard rebuilt from a clean `npm run build`, full Python suite re-run), not just read off old docs.

## Monday: Build dashboard wireframes (pages, charts needed)
- [x] Four-page plan settled and documented: Overview, Benchmarks, Pass@k, Evaluation Pipeline (`docs/dashboard.md` §2) — no extra pages added beyond the Week-5 brief
- [x] Chart need identified: exactly one chart (Pass@k vs k, grouped by benchmark) — deliberately not over-built
- [x] Data source decided up front: the existing `evaluation/results/results.json` (Week 3/4 artefact), read-only — no new backend

## Tuesday: Scaffold React app + Recharts skeleton
- [x] `dashboard/` — Vite + React 18 + TypeScript (strict mode) + `react-router-dom` + `recharts`
- [x] `dashboard/src/components/` — `AppShell`, `StatTile`, `ProvenanceBadge`, `MetadataTable`, `PassAtKChart`, `PipelineDiagram`, `DataState`
- [x] `dashboard/src/pages/` — `Overview.tsx`, `Benchmarks.tsx`, `PassAtK.tsx`, `Pipeline.tsx`
- [x] `dashboard/src/App.tsx` — route table; `main.tsx` uses `HashRouter` (static hosting, no server-side routing needed)

## Wednesday: Wire skeleton to read static results.json (placeholder data)
- [x] `dashboard/src/lib/types.ts` — TS types mirroring `interfaces/contracts.py::EvalResult` / `evaluation/results_store.py` field-for-field
- [x] `dashboard/src/lib/loadResults.ts` — fetches `/data/results.json`, runtime-validates the shape, formats missing fields as `N/A` rather than fabricating numbers
- [x] `dashboard/scripts/sync-results.mjs` — copies the repo's real `evaluation/results/results.json` into `dashboard/public/data/` (plain copy, no transform); wired into `predev`/`prebuild`
- [x] REAL vs DEMO_TEST provenance surfaced explicitly (`ProvenanceBadge`, Overview banner) — every current record is `DEMO_TEST` (no GPU/checkpoint in this environment) and the dashboard says so rather than hiding it
- [x] Re-verified 2026-08-12: `npm run build` → `sync-data` copies a real 4 KB `results.json`, `tsc` strict-mode compiles clean, `vite build` succeeds (`dashboard/dist/`)

## Thursday: Rehearse — explain eval pipeline + show wireframe
- [x] `docs/p5_panel_notes.md` §8 — the pipeline diagram (`Dataset -> Prompt -> Merged Model -> ... -> Dashboard`), every stage mapped to a real module, no invented steps
- [x] `PipelineDiagram.tsx` / Pipeline page — the same diagram shown live in the dashboard itself, not just in notes
- [x] `docs/p5_panel_notes.md` — ~300-word / ~3-minute speaking script, written and ready to rehearse

## Friday: Polish Eval/Dashboard slide(s)/notes for panel review
- [x] `docs/p5_panel_notes.md` — full panel notes: P5's responsibility, HumanEval/MBPP provenance, Pass@k, results.json schema, dashboard structure, current limitations stated plainly (mock backend, no historical trend view, process-level sandboxing only)
- [x] `docs/dashboard.md` — companion technical write-up (data flow, React structure, Recharts usage, current limitations) for anyone reviewing the code, not just the talk

## Cross-cutting Week-5 deliverables
- [x] No CSS framework/UI kit/state-management library — plain `useState`/`useEffect` and hand-written CSS custom properties, appropriately scoped to a Week-5 skeleton
- [x] `dashboard/` is fully self-contained: nothing in `corpus/`, `partitions/`, `evaluation/`, or the Python test suite was touched
- [x] Python suite still green: 338 tests passing (Weeks 1–4 baseline, re-run 2026-08-12)
- [x] No secrets/keys in the dashboard or its config

**Week 5 status: COMPLETE.**

**Explicitly out of Week-5 scope (unchanged from `docs/dashboard.md` §9):**
- Live/streaming updates — single static snapshot per page load, per the Week-5 brief ("static results.json (placeholder data)")
- Historical trend view across multiple runs
- Automated frontend test suite (Jest/Vitest) — verification was strict-mode compile + production build + manual browser check
- A REAL (non-DEMO_TEST) result — blocked on P1's real checkpoint + a GPU, outside P5's control
