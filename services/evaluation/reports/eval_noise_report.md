# Evaluation noise and guard thresholds

_How large a change must be before D5 can tell it from noise_  
**Generated:** 2026-10-09T05:48:56Z

## HumanEval guard

- **baseline anchor:** results/humaneval_guard/base_anchor.json
- **tasks scored:** 20
- **baseline pass@1:** 0.5
- **standard error:** 0.1118
- **95% bootstrap interval:** 0.3 – 0.7
- **D5 tolerance (max allowed drop):** 0.02
- **unpaired minimum detectable drop at this size:** 0.3099

Candidate and baseline are scored on the same tasks, so the relevant noise is paired: only tasks whose outcome flips between the two models contribute. The fraction that flips (discordance) is unknown until a candidate is scored, so the floor is shown for a range of plausible values.

| discordance | min detectable drop @ 20 tasks | @ 164 tasks | tasks needed to resolve 0.02 |
| --- | --- | --- | --- |
| 0.05 | 0.098 | 0.0342 | 481 |
| 0.1 | 0.1386 | 0.0484 | 961 |
| 0.2 | 0.196 | 0.0684 | 1921 |

### Locked thresholds

- D5 in-project noise band = spread of 3 repeated baseline evaluations (computed with evaluation.completion.noise_band); see the in-project section for its value.
- D5 tolerance stays 0.02 (registry-owned); P5 does not loosen it.
- Guard noise floor at the scored 20 tasks (assumed discordance 0.1): 0.1386. A drop above the tolerance but below this is reported `within_noise`, not a pass.
- On the full 164-task benchmark the floor falls to 0.0484 — still above the tolerance, so a 2-point regression is not resolvable from one greedy sample per task on HumanEval alone.
- Resolving the tolerance needs about 961 paired tasks at discordance 0.1 — e.g. HumanEval + MBPP together, or several samples per task.

## In-project completion

**D5 noise band** (spread of 3 repeated baseline evaluations (D5)): **0.0**. Greedy decoding is deterministic, so three repeats of the same baseline are identical and the band is zero: under it, any positive gain counts as an improvement. It measures decode noise only, not the variation between independently trained adapters.

| client | version | edit similarity | exact match | exact-match SE |
| --- | --- | --- | --- | --- |
| werkzeug | v1 — naive avg | 0.5337 | 0.2667 | 0.0571 |
| werkzeug | v2 — exact SVD | 0.547 | 0.2833 | 0.0582 |

At 60 paired examples and an assumed 0.1 discordance, an exact-match change smaller than 0.08 is inside noise. Edit-similarity band: supplementary bootstrap band not measurable from the recorded round: it needs per-example rows (evaluation.completion.per_example_rows) for both versions; evaluation.eval_harness.noise.paired_bootstrap_diff_ci computes it from them.
