# D1 dataset

The code corpus CLASP's clients train and are evaluated on. Owner: P5. Built
by `services/evaluation` (configs and reports referenced below are under that
directory).

## Decision

A curated multi-repository corpus of permissively licensed open-source
Python, one federated client per repository. It was chosen over CodeSearchNet,
The Stack v2, py150, CodeParrot and a (desirable but unavailable) proprietary
codebase because it is the only candidate with an unambiguous project
boundary — the precondition for project clusters — and per-file licence
certainty (`reports/dataset_survey.md`, weighted score 5.0/5).

## Sources (pinned)

| domain | project | ref | licence |
|---|---|---|---|
| web / tooling | Flask | 3.0.3 | BSD-3-Clause |
| web / tooling | Requests | v2.32.3 | Apache-2.0 |
| web / tooling | Werkzeug | 3.0.3 | BSD-3-Clause |
| web / tooling | Click | 8.1.7 | BSD-3-Clause |
| web / tooling | Colorama | 0.4.6 | BSD-3-Clause |
| web / tooling | Jinja2 | 3.1.4 | BSD-3-Clause |
| scientific | NumPy | v2.1.3 | BSD-3-Clause |
| scientific | pandas | v2.2.3 | BSD-3-Clause |
| scientific | scikit-learn | 1.5.2 | BSD-3-Clause |

Collection (`configs/dataset.yaml`, `configs/dataset_scientific.yaml`): `*.py`
files only, excluding tests, docs, examples, benchmarks, build output,
migrations and generated version files; files under 10 code lines or over
256 KiB dropped; exact-content duplicates dropped.

| corpus | files | code lines | sha256 |
|---|---|---|---|
| web / tooling (6 projects) | 135 | 45,922 | `1d2d6f6e…` |
| scientific (3 projects) | 733 | 403,675 | `5901ae1a…` |

## Partitions and held-out split

- One client per project (project-level partitioning, seed 20260616);
  shards are disjoint by file and by content
  (`reports/partition_validation_report.md`, 9/9 checks pass).
- **10% of each client's files are held out** (seed 20260616) as that
  client's in-project evaluation set and never trained on
  (`configs/materialize_client.yaml`).
- `scripts/materialize_client_repo.py` writes each client to the shared
  `<repo>/datasets/materialized/<client_id>/{repo,held_out}/`, which the edge
  lane trains from and docker-compose's `train` profile mounts.

## Federation used in the rounds

| cluster | clients |
|---|---|
| web | flask, requests, werkzeug |
| scientific | numpy, pandas, scikit-learn |

## Deviation from the D1 decision

D1 fixed the web cluster as **{django, flask, requests}**. The rounds were
run with **{flask, requests, werkzeug}**: Django is not in the collected
corpus and Werkzeug took its place. The reason was not recorded at the time.
D1 changes need all-hands sign-off; this one has none on record yet, and
every result so far (both federated rounds, the integration round) is on
the werkzeug composition. Restoring Django would mean collecting it,
re-partitioning and retraining every web-cluster adapter.
