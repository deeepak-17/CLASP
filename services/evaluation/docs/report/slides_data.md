# Slides data

_Headline numbers for the demo and deck, read from committed results_  
**Generated:** 2026-10-09T06:21:52Z

| headline | value | claim | source |
| --- | --- | --- | --- |
| personalization | 6/6 clients, mean -0.178 ppl | Every client's composite has lower perplexity than the base on its own never-trained-on code. | results/personalization.json · rounds[-1].summary |
| d3_cluster_layer | helps 0/6 → 6/6 clients (vs α = 0) | Supports D3, not conclusive: in round 2, α = 0 removes a layer the client was trained on. | results/personalization.json · rounds[*].summary.n_cluster_helps |
| round_over_round | lower on 6/6 clients, by 0.001–0.005 ppl | Within unmeasured seed noise: one seed, no interval. | results/personalization.json · comparisons[-1] |
| aggregation | web 1.92× · scientific 2.15× | SVD aggregation is ~2× closer to the exact average than naive factor averaging. | results/in_project_metric.json · aggregation_ablation |
| in_project | edit sim 0.534 → 0.547 | The SVD aggregate also completes held-out code better (within noise at 60 examples). | results/in_project_metric.json · in_project_completion |
| guard_baseline | pass@1 0.50 (95% CI 0.30–0.70) | Base model on the 20-task HumanEval guard subset. | results/noise_report.json · humaneval_guard |
| guard_noise | 13.9 pts now, 4.8 pts on full HumanEval | D5's 2-point tolerance is below the guard's noise floor. | results/noise_report.json · locked_thresholds |
