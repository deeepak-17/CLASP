# CLASP-P5 · Week 9 Completion Checklist

**Scope:** P5 Week-9 targets (`CLASP_Daily_Targets_v2.pdf`, Phase II — G3: full E2E federated round).

**Note:** written on 2026-10-08, after the Phase II calendar, while closing out P5's remaining deliverables. Items marked `[x]` cite the file that proves them; team events (panel reviews, all-hands runs, rehearsals) are not code deliverables and are left unticked unless the repo records them.

## Mon–Tue: All-hands round
- [ ] All-hands runs (team event). P5 side: the round feed (`scripts/export_round_feed.py`) and the four-seam integration test pass on a trial merge with `integration/panel`

## Wed: Eval-noise write-up; dashboard shows live round
- [x] `reports/eval_noise_report.md` (from `scripts/build_noise_report.py`)
- [x] Federated Rounds page and `scripts/export_round_feed.py` built and contract-tested against `scripts/demo_round.py`'s manifest shape
- [ ] A live round actually shown — the round manifests `demo_round.py` writes are gitignored, so no `rounds.json` is committed and the page shows its empty state

## Thu: Personalization chart on dashboard
- [x] Personalization page: perplexity reduction per client, both rounds; cluster-layer effect before/after D3

## Fri: G3 sign-off
- [ ] Team sign-off (team event)
