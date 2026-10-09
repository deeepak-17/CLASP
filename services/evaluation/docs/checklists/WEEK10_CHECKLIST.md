# CLASP-P5 · Week 10 Completion Checklist

**Scope:** P5 Week-10 targets (`CLASP_Daily_Targets_v2.pdf`, Phase II — hardening + the clustering novelty).

**Note:** written on 2026-10-08, after the Phase II calendar, while closing out P5's remaining deliverables. Items marked `[x]` cite the file that proves them; team events (panel reviews, all-hands runs, rehearsals) are not code deliverables and are left unticked unless the repo records them.

## Mon: Personalization report v1 draft
- [x] `reports/personalization_report.md`

## Tue: Guard thresholds locked from noise data
- [x] D5 in-project noise band reported as D5 defines it — spread of 3 repeated baseline evals (0.0: greedy repeats are identical), computed with `evaluation.completion.noise_band`
- [x] `configs/guard_thresholds.yaml` + `results/noise_report.json`: paired noise floor 13.9 pts at 20 tasks, 4.8 pts at 164; D5's 2-pt tolerance is below it, so drops inside the floor are reported `within_noise` (`eval_harness/noise.py::classify_guard_drop`)
- [ ] Not yet "locked from noise data": the floor rests on an **assumed** discordance (0.10). It becomes measured once a candidate anchor is scored (`build_noise_report.py --candidate-anchor`), and the in-project band needs per-example rows from P1

## Wed: Report v1 review with team
- [ ] Team review (team event)

## Thu: Regression suite skeleton in CI
- [x] The P5 suite runs in the `pytest (evaluation)` CI job (PR #20)

## Fri: Buffer / debt
- [x] Python-version bug in exclude-glob matching found by CI and fixed (PR #20)
