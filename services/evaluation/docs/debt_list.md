# P5 debt list

Review feedback and known gaps, with status. The first block is the review of
PR #13; the second is what is still open after the Phase II work.

## Review feedback (PR #13)

| # | item | status |
|---|---|---|
| 1 | Root `conftest.py` put the repo root ahead of site-packages, shadowing `clasp-evaluation` | fixed — removed (PR #20) |
| 2 | Second package named `evaluation` breaks `evaluation.completion` imports | fixed — one package, P5 code as its subpackages (PR #20) |
| 3 | Root layout vs one directory per module; `utils/`/`interfaces/` collision risk | fixed — everything under `services/evaluation/` (PR #20) |
| 4 | P5 tests never ran in CI; `jsonschema` not installed | fixed — suite under `services/evaluation/tests`, deps declared (PR #20) |
| 5 | Root `pytest.ini` applied repo-wide | fixed — options in the package pyproject (PR #20) |
| 6 | `.gitignore` silently ignoring new partition files | fixed — module `.gitignore` re-includes `datasets/` (PR #20) |
| 7 | Validator returned `ok=True` without `jsonschema` | fixed — fails closed (PR #20) |
| 8 | (found by CI once the suite ran) glob matching wrong on Python < 3.13 | fixed (PR #20) |

## Open

| item | owner | blocked on | note |
|---|---|---|---|
| HumanEval guard on a candidate | P1 → P5 | composite-model samples (GPU) | `build_noise_report.py --candidate-anchor` gives the verdict |
| Edit-similarity noise band | P1 → P5 | per-example rows in the round manifest | bootstrap implemented |
| Training-seed noise band | P1 | adapters trained with ≥ 3 seeds | — |
| D5 tolerance below the guard's noise floor | P4 + P5 | team decision | evidence in `reports/eval_noise_report.md` |
| Dashboard image not built | P5 | a machine with Docker | `docker compose build dashboard` |
| Dashboard pages not viewed in a browser here | P5 | a working browser | type-checked, built, contract-tested |
| Web cluster differs from the D1 plan | team | — | on disk {flask, requests, werkzeug}; plan said {django, flask, requests} |
| `pytest` from the repo root fails to collect | all modules | — | every service names its test package `tests`; CI runs each separately |
