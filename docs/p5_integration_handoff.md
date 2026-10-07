# P5 (Evaluation & Data) — integration handoff

What P5 provides at each seam, how to run it, what has been verified against
the other modules' real code, and what is still open (with the owner).
Branch: `services/evaluation`. Verified against `origin/integration/panel` and
`origin/main` by a trial merge (not committed anywhere).

## 1. Layout after this branch

| Path | What | Import name |
|---|---|---|
| `services/evaluation/src/evaluation/` | P5's installable package: `completion.py` (the D5 in-project metric, byte-identical to `integration/panel`), `__main__.py` (container entrypoint) | `evaluation` |
| `eval_harness/` | HumanEval/MBPP harness, Pass@k, sandboxed execution, in-project eval, guard anchors, round feed | `eval_harness` (was `evaluation`) |
| `interfaces/registry_wire.py` | contracts v1.0 wire format both ways + real-API registry reader | — |
| `partitions/`, `corpus/`, `utils/`, `interfaces/` | data pipeline + mirrors (unchanged layout) | — |
| `tests/p5/` | P5 test suite (was `tests/`) | — |
| `dashboard/` | React/Recharts dashboard, now with a Federated Rounds page | — |

**Why the rename.** The old top-level `evaluation/` had the same import name as
`services/evaluation`. With the root `conftest.py` putting the repo root on
`sys.path`, a merged tree failed `tests/integration` at collection
(`ModuleNotFoundError: No module named 'evaluation.completion'`). Reproduced on
a trial merge, fixed by the rename; `tests/p5/` keeps P5's conftest from
loading for `tests/e2e` and `tests/integration`.

## 2. Seams

| Seam | P5 side | Counterpart | Verified how |
|---|---|---|---|
| Held-out split → edge completion eval | `partitions.materialize` writes `held_out/`; `eval_harness.in_project` scores with `evaluation.completion` | `edge.completion_eval.client_examples` | Same example set and `examples_sha256` on all 9 real clients on disk |
| C2 payload (P5 metrics → registry) | `registry_wire.eval_result_wire` / `promote_body` | `registry.app._eval_result_from`, `registry.promotion.decide` | Real registry app in-process: PROMOTE, and ROLLBACK on a >2-pt HumanEval drop |
| Registry → P5 (what to evaluate) | `registry_wire.HttpRegistryReadClient` | `GET /adapters/{name}/versions[/{v}]`, `/active` | Real registry app in-process |
| HumanEval guard | `scripts/score_humaneval_samples.py` → `anchor.json` | `edge.promote.resolve_guard`, `promote_candidate` | Real resolve_guard + promote_candidate + registry: decision becomes authoritative |
| Round → dashboard | `scripts/export_round_feed.py` → `rounds.json` → `/rounds` page | `scripts/demo_round.py` manifest | `seam_c2` / `humaneval_guard` from the real producers; TS guard run on exporter output |
| Container | `python -m evaluation` | `docker-compose.yml` `evaluation` service | Entry point run locally and against a live registry; **image not built** (no docker here) |

## 3. Commands for the integration day

```bash
# environment (team convention; CI does the same)
pip install -e contracts -e services/evaluation

# P5 suite (also runs inside a bare `pytest` from the root)
python -m pytest tests/p5 -q

# close the HumanEval guard: score baseline and candidate samples with the SAME tool
python scripts/score_humaneval_samples.py --samples services/edge/eval_out/samples.jsonl \
    --generation-manifest services/edge/eval_out/manifest.json --out eval_out/baseline/anchor.json
python scripts/score_humaneval_samples.py --samples <candidate>/samples.jsonl \
    --generation-manifest <candidate>/manifest.json --out eval_out/candidate/anchor.json
python scripts/demo_round.py --round 1 \
    --baseline-anchor eval_out/baseline/anchor.json --candidate-anchor eval_out/candidate/anchor.json

# dashboard feed after a round
python scripts/export_round_feed.py experiments/w12-integration/results/round1_manifest.json
cd dashboard && npm run sync-data && npm run dev
```

Measured today: the 20 committed base-model samples
(`services/edge/eval_out/samples.jsonl`, sha256 `ad912167…`) score
**base pass@1 = 0.50 (10/20)**; the same 20 tasks' canonical solutions score
20/20 through the same path.

## 4. Trial-merge results (integration/panel + this branch + main)

| CI job | Before this branch's fixes | After |
|---|---|---|
| `tests/integration` (four seams) | **collection error** | 22 passed |
| `tests/e2e` | 1 passed | 1 passed |
| `services/{edge,cluster,registry,evaluation}/tests`, `contracts/tests` | pass | pass |
| `ruff check .` | — | clean |
| bare `pytest` from root (P5 + e2e + integration) | — | all pass |

Git-level merge of this branch into `integration/panel`: no conflicts
(`.gitignore` auto-merges).

## 5. Open items

| Item | Owner | Note |
|---|---|---|
| Candidate-side HumanEval samples (composite model) | P1 | Generation needs the GPU; P5 scores them with the command above |
| `resolve_guard`'s reason text says "from evalplus" | P1 | Inaccurate when the anchor's `scorer` is P5's executor; the number is unaffected |
| `perplexity` on the wire | P1 | contracts v1.0 requires it; `registry_wire.in_project_wire` refuses to send without it |
| Training-seed noise band | P1 + P5 | Greedy repeats give 0.0 (decode noise only). A real band needs adapters trained with different seeds; `evaluation.completion.noise_band` then computes it |
| CI does not run `tests/p5` | P5 + P2 (CI owner) | Not added to the shared `ci.yml` to avoid a conflict with `integration/panel`'s edit. Proposed job: `pip install -e contracts -e services/evaluation pyyaml jsonschema numpy pytest && pytest tests/p5 -q` |
| Compose `evaluation` service has no `CLASP_REGISTRY_URL` | P4 | Without it the container only prints what it provides and exits 0 |
| `docker compose build evaluation` | P4/P5 | Not run — no docker daemon on this machine |
| Rounds page visual check | P5 | Type-checked, built and contract-tested; not rendered in a browser here |
| Historical `WEEK*_CHECKLIST.md`, `reports/*.md`, recorded result JSON | — | Still show the old `evaluation/` paths; they are records of past runs and were left as they were |
