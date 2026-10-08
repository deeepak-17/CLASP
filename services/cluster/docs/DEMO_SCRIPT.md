# P2 Cluster — end-semester demo script (≈ 6 minutes)

Run everything from `services/cluster` with the package installed
(`pip install -e ../../contracts -e ".[test]"`). Every command below was run in the verification environment; the
expected output is what the code prints, nothing is mocked except where stated. Keep `demo_runs/phase2_cluster_evidence.md`
open as the backup if the live run is slow (the recorded run took about 80 s for evidence, about 10 s for the demo).

| # | Say | Run | You should see |
|---|---|---|---|
| 1 | "Three FedProx clients, SVD aggregation, one round." | `python -m cluster.demo --seed 42` | stage trace, round metrics block, `FINAL ROUND STATUS: SUCCESS` |
| 2 | "Two clusters, one client mislabelled, a straggler, a flaky link — then re-clustering repairs the label." | `python -m cluster.demo_phase2` | straggler skipped as `timeout`; `attempts>1` for the flaky client; after round 2 `dynamic`, `moved: {cluster-sci/client-2: cluster-web -> cluster-sci}`; `isolation_violations() = []`; `static_fallback` case; `adapters identical = True`; HTTP dry run then apply; `PHASE II DEMO OK` |
| 3 | "Why SVD and not averaging A and B." | open `demo_runs/phase2_cluster_evidence.md`, section *3-client training* | SVD vs naive loss per round |
| 4 | "What the clustering buys, and when it refuses." | same file, *Dynamic re-clustering vs static* and *Static fallback* | wrong label: dynamic ARI 1.0 vs static ≈ 0.32; fallback reasons |
| 5 | "Robustness." | same file, *Straggler / dropout / quorum* | timeout skip, dropout and rejoin, quorum failure keeps the adapter |
| 6 | "Honesty slide." | `docs/INTEGRATION_BOUNDARIES.md` summary table | G1 mTLS and DP µ tuning are **blocked by P3**; registry/eval are stubs; toy workload |

Likely questions → answers with the evidence: *Is it reproducible?* (SHA-256 of adapters identical across runs, every seed.)
*Does a client ever see another cluster's adapter?* (`isolation_violations()` = 0; adversarial placement test; HTTP 403.)
*Does it work on the real model?* (No — toy workload; §15 of the report.) *Is mTLS done?* (No — blocked by P3; the Cluster side
refuses un-certified clients in a test with a throw-away CA.)
