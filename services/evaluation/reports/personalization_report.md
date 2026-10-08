# Personalization results

_Held-out perplexity on each client's own never-trained-on files (lower is better)_  
**Generated:** 2026-10-08T07:28:00Z

Each row compares three models on one client's held-out files: the frozen base, the client-only adapter, and the three-layer composite base + α·cluster + β·client (D6). The personalization delta is composite − base; the cluster contribution is composite − client-only at the reference α (negative means the cluster layer helps). Every number is read from the edge round manifest named under Sources.

## round 1 — clients trained on the bare base

- **model:** deepseek-ai/deepseek-coder-1.3b-base
- **seed:** 0
- **manifest written:** 2026-08-18T06:57:12.716892+00:00
- **α grid:** 0.0, 0.5
- **clients improved over base:** 6 / 6
- **mean personalization delta (ppl):** -0.1748
- **cluster layer helps:** 0 / 6 clients
- **sweep chose α = 0 (cluster off):** 6 / 6 clients

| client | held-out tokens | base | client only | composite | Δ vs base | cluster @ α_ref | best α |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scientific/client-numpy | 108026 | 2.736 | 2.653 | 2.653 | -0.083 | +0.011 | 0 |
| scientific/client-pandas | 162647 | 2.91 | 2.718 | 2.718 | -0.192 | +0.010 | 0 |
| scientific/client-scikit-learn | 248968 | 2.584 | 2.371 | 2.371 | -0.213 | +0.014 | 0 |
| web/client-flask | 10549 | 2.452 | 2.269 | 2.269 | -0.183 | +0.000 | 0 |
| web/client-requests | 513 | 3.463 | 3.26 | 3.26 | -0.203 | +0.002 | 0 |
| web/client-werkzeug | 30279 | 2.828 | 2.653 | 2.653 | -0.175 | +0.012 | 0 |

## round 2 — D3 order (client on frozen base + 0.5·cluster)

- **model:** deepseek-ai/deepseek-coder-1.3b-base
- **seed:** 0
- **manifest written:** 2026-10-06T08:34:23.681974+00:00
- **α grid:** 0.0, 0.25, 0.5, 1.0
- **clients improved over base:** 6 / 6
- **mean personalization delta (ppl):** -0.1783
- **cluster layer helps:** 6 / 6 clients
- **sweep chose α = 0 (cluster off):** 0 / 6 clients

| client | held-out tokens | base | client only | composite | Δ vs base | cluster @ α_ref | best α |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scientific/client-numpy | 108026 | 2.736 | 2.667 | 2.651 | -0.085 | -0.016 | 0.5 |
| scientific/client-pandas | 162647 | 2.91 | 2.736 | 2.714 | -0.196 | -0.022 | 0.5 |
| scientific/client-scikit-learn | 248968 | 2.584 | 2.388 | 2.367 | -0.217 | -0.021 | 0.5 |
| web/client-flask | 10549 | 2.452 | 2.311 | 2.264 | -0.188 | -0.047 | 0.5 |
| web/client-requests | 513 | 3.463 | 3.323 | 3.259 | -0.204 | -0.064 | 0.5 |
| web/client-werkzeug | 30279 | 2.828 | 2.695 | 2.648 | -0.180 | -0.047 | 0.5 |

## Round 1 → round 2

Composite perplexity improved on 6 of 6 clients. The cluster-contribution columns show whether the cluster layer moved from harmful (positive) to helpful (negative) once clients were trained in the D3 order.

| client | composite Δ (r→r) | cluster contribution before | after | best α before | after |
| --- | --- | --- | --- | --- | --- |
| scientific/client-numpy | -0.002 | +0.011 | -0.016 | 0 | 0.5 |
| scientific/client-pandas | -0.004 | +0.010 | -0.022 | 0 | 0.5 |
| scientific/client-scikit-learn | -0.004 | +0.014 | -0.021 | 0 | 0.5 |
| web/client-flask | -0.005 | +0.000 | -0.047 | 0 | 0.5 |
| web/client-requests | -0.001 | +0.002 | -0.064 | 0 | 0.5 |
| web/client-werkzeug | -0.005 | +0.012 | -0.047 | 0 | 0.5 |

## Caveats

- 1.3B `dev` profile on a 4 GB RTX 2050 — not comparable to a 6.7B figure.
- client-requests' held-out split is one 513-token block; its row is the noisiest.
- Each cluster adapter aggregates all members, including the client being evaluated (inherent to federated averaging); held-out files were never trained on.
- Perplexity is the edge's held-out training eval. D5's primary signal is in-project edit similarity — see the In-Project page and the noise report.

## Sources

- `results/edge_rounds/round1_manifest.json`
- `results/edge_rounds/round2_d3_manifest.json`
