# CLASP-P5 · Week 7 Completion Checklist

**Scope:** P5 Week-7 targets (`CLASP_Daily_Targets_v2.pdf`, Phase II — G1: Edge↔Cluster live over mTLS).

**Note:** written on 2026-10-08, after the Phase II calendar, while closing out P5's remaining deliverables. Items marked `[x]` cite the file that proves them; team events (panel reviews, all-hands runs, rehearsals) are not code deliverables and are left unticked unless the repo records them.

## Mon: Vite dashboard scaffold in services/evaluation/dashboard
- [x] `services/evaluation/dashboard/` (moved under the module in PR #20)

## Tue: Dashboard reads results.json
- [x] `dashboard/scripts/sync-results.mjs` + `src/lib/loadResults.ts`

## Wed: Pass@k chart component
- [x] `src/components/PassAtKChart.tsx`

## Thu: Adapter-lineage view stub
- [x] Beyond a stub: `eval_harness/lineage.py` → `results/lineage.json` → Adapter Lineage page (client → cluster → composite, registry sha256, click-through provenance)

## Fri: Dashboard deployed in compose
- [x] `dashboard/Dockerfile` (node build → nginx) and a `dashboard` service in `docker-compose.yml` (port 8005)
- [ ] Image built and started — no Docker on the development machine; run `docker compose up -d dashboard`
