# Slides data

_Headline numbers for the demo and deck, read from committed results_  
**Generated:** 2026-10-09T05:48:56Z

| headline | value | claim | source |
| --- | --- | --- | --- |
| personalization | 6/6 clients, mean -0.178 ppl | Every client's model is better on its own never-trained-on code. | results/personalization.json · rounds[-1].summary |
| d3_cluster_layer | helps 0/6 → 6/6 clients | Training clients on the frozen cluster (D3) turns the cluster layer from harmful to helpful. | results/personalization.json · rounds[*].summary.n_cluster_helps |
| round_over_round | composite better on 6/6 clients | Round 2 beats round 1 for every client. | results/personalization.json · comparisons[-1] |
| aggregation | web 1.92× · scientific 2.15× | SVD aggregation is ~2× closer to the exact average than naive factor averaging. | results/in_project_metric.json · aggregation_ablation |
| in_project | edit sim 0.534 → 0.547 | The SVD aggregate also completes held-out code better (within noise at 60 examples). | results/in_project_metric.json · in_project_completion |
| guard_baseline | pass@1 0.50 (95% CI 0.30–0.70) | Base model on the 20-task HumanEval guard subset. | results/noise_report.json · humaneval_guard |
| guard_noise | 13.9 pts now, 4.8 pts on full HumanEval | D5's 2-point tolerance is below the guard's noise floor. | results/noise_report.json · locked_thresholds |
