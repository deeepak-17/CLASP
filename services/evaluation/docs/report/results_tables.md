# Evaluation results tables

_Every number quoted in the evaluation section_  
**Generated:** 2026-10-09T05:30:09Z

## Personalization — round 1 — clients trained on the bare base

| client | tokens | base | client only | composite | Δ vs base | cluster @ α=0.5 | best α |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scientific/client-numpy | 108026 | 2.736 | 2.653 | 2.653 | -0.083 | +0.011 | 0 |
| scientific/client-pandas | 162647 | 2.91 | 2.718 | 2.718 | -0.192 | +0.010 | 0 |
| scientific/client-scikit-learn | 248968 | 2.584 | 2.371 | 2.371 | -0.213 | +0.014 | 0 |
| web/client-flask | 10549 | 2.452 | 2.269 | 2.269 | -0.183 | +0.000 | 0 |
| web/client-requests | 513 | 3.463 | 3.26 | 3.26 | -0.203 | +0.002 | 0 |
| web/client-werkzeug | 30279 | 2.828 | 2.653 | 2.653 | -0.175 | +0.012 | 0 |

Improved over base: 6/6; mean Δ -0.1748 (range -0.213 to -0.083); cluster layer helps 0/6; sweep chose α = 0 for 6/6.

## Personalization — round 2 — D3 order (client on frozen base + 0.5·cluster)

| client | tokens | base | client only | composite | Δ vs base | cluster @ α=0.5 | best α |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scientific/client-numpy | 108026 | 2.736 | 2.667 | 2.651 | -0.085 | -0.016 | 0.5 |
| scientific/client-pandas | 162647 | 2.91 | 2.736 | 2.714 | -0.196 | -0.022 | 0.5 |
| scientific/client-scikit-learn | 248968 | 2.584 | 2.388 | 2.367 | -0.217 | -0.021 | 0.5 |
| web/client-flask | 10549 | 2.452 | 2.311 | 2.264 | -0.188 | -0.047 | 0.5 |
| web/client-requests | 513 | 3.463 | 3.323 | 3.259 | -0.204 | -0.064 | 0.5 |
| web/client-werkzeug | 30279 | 2.828 | 2.695 | 2.648 | -0.180 | -0.047 | 0.5 |

Improved over base: 6/6; mean Δ -0.1783 (range -0.217 to -0.085); cluster layer helps 6/6; sweep chose α = 0 for 0/6.

## Aggregation error (relative Frobenius vs exact average)

| cluster | naive average | exact SVD | ratio |
| --- | --- | --- | --- |
| web | 0.3287 | 0.1712 | 1.92× |
| scientific | 0.5978 | 0.2777 | 2.15× |

## In-project completion

| client | cluster | aggregate | edit similarity | exact match |
| --- | --- | --- | --- | --- |
| werkzeug | web | v1 — naive avg | 0.5337 | 0.2667 |
| werkzeug | web | v2 — exact SVD | 0.547 | 0.2833 |

## HumanEval guard noise

- **tasks scored:** 20
- **base pass@1:** 0.5
- **standard error:** 0.1118
- **95% bootstrap interval:** 0.3 – 0.7
- **D5 tolerance:** 0.02

| discordance | floor @ 20 | floor @ 164 | tasks for 0.02 |
| --- | --- | --- | --- |
| 0.05 | 0.098 | 0.0342 | 481 |
| 0.1 | 0.1386 | 0.0484 | 961 |
| 0.2 | 0.196 | 0.0684 | 1921 |

## Sources

- `results/edge_rounds/round1_manifest.json`
- `results/edge_rounds/round2_d3_manifest.json`
- `results/humaneval_guard/base_anchor.json`
- `results/in_project_metric.json`
