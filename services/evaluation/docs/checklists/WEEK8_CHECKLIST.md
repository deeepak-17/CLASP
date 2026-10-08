# CLASP-P5 · Week 8 Completion Checklist

**Scope:** P5 Week-8 targets (`CLASP_Daily_Targets_v2.pdf`, Phase II — G2: 3-layer composition, 6 clients / 2 clusters).

**Note:** written on 2026-10-08, after the Phase II calendar, while closing out P5's remaining deliverables. Items marked `[x]` cite the file that proves them; team events (panel reviews, all-hands runs, rehearsals) are not code deliverables and are left unticked unless the repo records them.

## Mon: In-project eval harness on merged model
- [x] `eval_harness/in_project.py` scoring with the canonical `evaluation.completion`

## Tue: Eval personalized vs base per client
- [x] `eval_harness/personalization.py` — base / client-only / composite per client, from the edge round manifests (`results/edge_rounds/`)

## Wed: First personalization delta measured
- [x] Round 1: 6/6 clients improved, mean −0.175 ppl (`results/personalization.json`)

## Thu: HumanEval guard check alongside
- [x] Base model scored: pass@1 0.50 on 20 tasks (`results/humaneval_guard/base_anchor.json`)
- [ ] Candidate scored — needs the composite model's samples from the GPU lane (P1)

## Fri: Delta + guard results committed
- [x] `results/personalization.json`, `reports/personalization_report.md`, `results/humaneval_guard/`
