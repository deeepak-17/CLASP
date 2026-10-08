# P5 demo — evaluation & data

A timed walkthrough of what P5 contributes to the CLASP demo, the data it
runs on, and the fallback if anything live fails. Every number quoted comes
from `results/slides_data.json`; regenerate it rather than editing a slide.

## Demo dataset

| what | where | how it is pinned |
|---|---|---|
| D1 corpus, web / tooling (6 projects, 135 files) | `datasets/partitions/manifest.json` | corpus sha256 in the manifest; repos pinned to commits (`reports/corpus_collection_report.md`) |
| D1 corpus, scientific (3 projects, 733 files) | `datasets/partitions/scientific/manifest.json` | same |
| Held-out split (never trained on) | materialised per client by `scripts/materialize_client_repo.py` | deterministic; file order hashed in each edge manifest |
| Federated rounds 1 and 2 | `results/edge_rounds/` | copied verbatim from the edge lane |
| HumanEval guard samples + anchor | `results/humaneval_guard/` | samples sha256 in the anchor |
| Everything above + every result | `results/ARCHIVE_MANIFEST.json` | `python scripts/archive_results.py --check` |

## Before the demo (once)

```bash
cd services/evaluation
python scripts/rebuild_results.py --label demo      # a few seconds, no GPU / network
python scripts/archive_results.py --check           # must say "archive intact"
cd dashboard && npm install && npm run build        # or: docker compose up -d dashboard (port 8005)
```

## Scenario (≈ 6 minutes)

| # | show | say | headline |
|---|---|---|---|
| 1 | Overview page | P5 owns the data the federation trains on and the measurement that decides whether a round's adapter is promoted. | — |
| 2 | Personalization page, first chart | Each client's model is measured on its own files that were never trained on. | `personalization` |
| 3 | Personalization page, second chart | Round 1: cluster layer off everywhere, slightly harmful. Round 2, clients trained on the frozen cluster (D3): it helps every client. | `d3_cluster_layer`, `round_over_round` |
| 4 | Adapter Lineage page → click `flask r2` | This client was trained on `cluster-web v2`, which was aggregated from the three round-1 web clients; registry sha256 shown. | — |
| 5 | In-Project page | The promotion rule decides on completion quality, not perplexity; SVD aggregation is ~2× more faithful than naive averaging. | `aggregation`, `in_project` |
| 6 | Noise & Guard page | Base pass@1 and its interval; the D5 2-point tolerance is below what 20 — or even 164 — tasks can resolve, so a drop inside the noise floor is reported as "within noise", never as a pass. | `guard_baseline`, `guard_noise` |
| 7 | Federated Rounds page | The D5 decision for each cluster and why it was provisional. | — |

## Fallback

If the dashboard cannot be served: open `docs/report/figures/*.svg` (same
data, static) and `docs/report/slides_data.md`. If a question needs a number
not on a slide, `docs/report/results_tables.md` has every number the
evaluation section quotes, with its source file.

## Likely questions

* **Why perplexity *and* edit similarity?** Perplexity is measured on every
  held-out token and drives the α sweep; edit similarity is what a developer
  experiences and is what D5 gates on. They can disagree (the naive aggregate
  had marginally lower perplexity but completed worse) — that is why D5 reads
  completion.
* **Is the improvement real or noise?** The personalization deltas are an
  order of magnitude larger than the round-1/round-2 differences and hold on
  all six clients; the in-project v1→v2 difference is within noise at 60
  examples, and the noise page says so.
* **Why is the guard provisional?** It is scored on the base model only; the
  composite model's samples come from the GPU lane.
