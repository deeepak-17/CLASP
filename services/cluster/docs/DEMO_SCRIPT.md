# P2 Cluster — end-semester demo script (≈ 6 minutes)

Run everything from `services/cluster` with the package installed
(`pip install -e ../../contracts -e ".[test]"`). Every command below was run in the verification environment; the
expected output is what the code prints, nothing is mocked except where stated. The demos take about 10 s each.

| # | Say | Run | You should see |
|---|---|---|---|
| 1 | "Three FedProx clients, SVD aggregation, one round." | `python -m cluster.demo --seed 42` | stage trace, round metrics block, `FINAL ROUND STATUS: SUCCESS` |
| 2 | "Two clusters, one client mislabelled, a straggler, a flaky link — then re-clustering repairs the label." | `python -m cluster.demo_phase2` | straggler skipped as `timeout`; `attempts>1` for the flaky client; after round 2 `dynamic`, `moved: {cluster-sci/client-2: cluster-web -> cluster-sci}`; `isolation_violations() = []`; `static_fallback` case; `adapters identical = True`; HTTP dry run then apply; `PHASE II DEMO OK` |
| 3 | "Honesty slide." | `docs/INTEGRATION_BOUNDARIES.md` summary table | G1 mTLS and DP µ tuning are **blocked by P3**; the registry end is P4's; toy workload |

Likely questions → answers with the evidence: *Is it reproducible?* (SHA-256 of adapters identical across runs, every seed; `tests/test_demo.py`.)
*Does a client ever see another cluster's adapter?* (`isolation_violations()` = 0; adversarial placement test; HTTP 403.)
*Does it work on the real model?* (No — toy workload.) *Is mTLS done?* (No — blocked by P3; the Cluster side
refuses un-certified clients in a test with a throw-away CA.)
