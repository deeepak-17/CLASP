# Reusing the P5 harness for Phase III ablations

Phase III sweeps rank, α/β, ε and the clustering scheme, and needs one quality
measurement per sweep point plus plots that regenerate from a result store.
Most of that already exists; this note maps each need to the code that serves
it and lists the gaps.

## What carries over unchanged

| Phase III need | P5 component | notes |
|---|---|---|
| quality per sweep point (in-project) | `evaluation.completion` + `eval_harness/in_project.py` | same held-out examples (`examples_sha256`) across points, so points are comparable |
| quality per sweep point (HumanEval guard) | `scripts/score_humaneval_samples.py` → anchor | score every point's samples with the same scorer |
| is a difference real? | `eval_harness/noise.py` | `paired_bootstrap_diff_ci` for in-project, `paired_min_detectable_drop` / `classify_guard_drop` for the guard |
| per-client personalization from a round | `eval_harness/personalization.py` | reads any `edge.round` manifest; one manifest per sweep point |
| provenance of every adapter | `eval_harness/lineage.py` | registry versions + sha256 per node |
| reproducible sampling | `generation.seed` + `task_seed` | per-task seeds, independent of task order |
| plots that regenerate from results | `utils/svg_charts.py`, `scripts/build_report_figures.py` | add one figure function per sweep type |
| results integrity | `scripts/archive_results.py` | archive each sweep's results; `--check` before a paper number is quoted |

## Gaps to close first

1. **Per-example in-project rows.** The edit-similarity noise band needs
   per-example scores for both sides of a comparison. `evaluation.completion`
   already produces them (`per_example_rows`); the edge round must write them
   to the manifest.
2. **Training-seed noise.** Every Phase II round used seed 0. Before the rank
   sweep, train the base configuration with 3 seeds; the spread is the noise
   band every other sweep is read against.
3. **Guard size.** At ~1,000 paired tasks a 2-point drop becomes resolvable
   (`reports/eval_noise_report.md`). Score HumanEval (164) + MBPP (500) with
   several samples per task, or relax the tolerance to the measured floor.
4. **Sweep-aware feed.** `personalization.build_feed` keys rounds by round
   number; a sweep needs a sweep-point key (rank, α, ε). Add a `point` field
   rather than overloading `round`.
5. **Result store layout.** P4 owns `runs/`; the exporters take explicit input
   paths, so pointing them at the store needs no code change.

## Suggested order

Seed-noise runs → rank sweep (cheapest, fixes rank for the rest) → α/β grid →
ε sweep → clustering ablation. Each step: export with the scripts above,
archive, then plot.
