"""Multi-cluster demo for P2: ``python -m cluster.demo_phase2``.

One narrated run of the real pipeline on a toy LoRA regression (CPU, ~15 s):

  1. two clusters x three FedProx clients, one client given the WRONG warm label
  2. round 1-2 with a straggler (timeout policy) -> per-cluster SVD aggregation
  3. dynamic re-clustering after round 2 repairs the label; isolation re-checked
  4. redistribution to the new members, with a flaky link that needs retries
  5. the static fallback when there is too little evidence
  6. the same run twice -> byte-identical adapters (reproducibility)
  7. (if httpx is installed) the same decision through ``POST /recluster``

Every number printed is measured in this run; the checks are real assertions, so
the script exits non-zero if a claim stops being true. What it does NOT show is
listed at the end (mTLS and DP are blocked by P3).
"""

from __future__ import annotations

import argparse
import hashlib
import logging

import numpy as np

from cluster.adapter_format import random_adapter
from cluster.clustering import ClusterAssigner
from cluster.federation import MultiClusterFederation
from cluster.simulation import make_clustered_clients
from cluster.straggler import StragglerPolicy

WEB, SCI = "cluster-web", "cluster-sci"
WRONG = "cluster-sci/client-2"  # truly a sci client, mislabelled as web
STRAGGLER = "cluster-sci/client-0"
FLAKY = "cluster-web/client-1"
DIM = 32


def _digest(fed: MultiClusterFederation) -> str:
    h = hashlib.sha256()
    for cid in sorted(fed.cluster_adapters):
        for arr in fed.cluster_adapters[cid].to_ndarrays():
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _build(seed: int, *, flaky: dict[str, int] | None = None, timeout_s: float | None = None):
    clients, truth = make_clustered_clients(seed=seed, dim=DIM)
    clients[STRAGGLER].simulated_delay_s = 30.0  # reports 30 s of "work"
    warm = dict(truth)
    warm[WRONG] = WEB
    failures = dict(flaky or {})

    def send(client_id, payload):
        if failures.get(client_id, 0) > 0:
            failures[client_id] -= 1
            raise ConnectionError("simulated link failure")

    assigner = ClusterAssigner(warm, [WEB, SCI], seed=0)
    fed = MultiClusterFederation(
        clients, assigner, random_adapter(DIM, DIM, seed=0),
        policy=StragglerPolicy(timeout_s=timeout_s), send=send if flaky else None,
    )
    return fed, truth


def run(seed: int = 42, verbose: bool = True) -> dict:
    def say(msg: str = "") -> None:
        if verbose:
            print(msg)

    facts: dict = {"seed": seed}
    say("=" * 70)
    say(f"CLASP P2 Cluster - Multi-cluster demo, seed={seed}")
    say("=" * 70)

    # ---- 1-4: federation with straggler, wrong label, flaky delivery -----
    fed, truth = _build(seed, flaky={FLAKY: 2}, timeout_s=5.0)
    say(f"\n[1] 2 clusters x 3 clients; {WRONG} is really a sci client but starts in {WEB}")
    say(f"    straggler policy: timeout 5 s; {STRAGGLER} reports 30 s")
    for _ in range(4):
        rnd = fed.run_round()
        for cid, res in rnd.clusters.items():
            loss = res.metrics.mean_loss
            retried = {c: n for c, n in res.delivery.attempts.items() if n > 1} or "-"
            shown = "n/a" if loss is None else f"{loss:.3f}"
            say(
                f"[round {rnd.round_id + 1}] {cid:<11} contributors={len(res.contributors)} "
                f"skipped={res.skipped or '-'} mean_loss={shown} attempts>1={retried}"
            )
        if rnd.recluster:
            rec = rnd.recluster
            say(f"    -> re-clustering after round {rec.completed_rounds}: {rec.method} ({rec.reason})")
            say(f"       moved: { {c: f'{a} -> {b}' for c, (a, b) in rec.moved.items()} }")
            compared = {k: round(v, 3) for k, v in (rec.comparison or {}).items()}
            say(f"       cohesion={rec.cohesion:.3f} separation={rec.separation:.3f} vs warm start: {compared}")
    facts["moved"] = {c: list(v) for c, v in fed.assigner.history[-1].moved.items()}
    assert facts["moved"] == {WRONG: [WEB, SCI]}, facts["moved"]
    assert fed.assigner.assignment == truth
    assert fed.rounds[0].clusters[SCI].skipped.get(STRAGGLER) == "timeout"
    assert fed.rounds[0].clusters[WEB].delivery.attempts[FLAKY] == 3  # 2 failures + success
    assert fed.isolation_violations() == []
    say("\n[2] assignment now equals the true groups; isolation_violations() = []")
    facts["isolation_violations"] = fed.isolation_violations()
    facts["digest"] = _digest(fed)

    # ---- 5: static fallback ------------------------------------------------
    fed2, _ = _build(seed)
    fed2.run_round()
    rnd = fed2.run_round(offline=set(truth) - {"cluster-web/client-0"})
    say(
        f"\n[3] only 1 client answers in the re-clustering round: "
        f"{rnd.recluster.method} - {rnd.recluster.reason}"
    )
    assert rnd.recluster.method == "static_fallback"
    assert rnd.assignment_before == rnd.assignment_after
    facts["fallback_reason"] = rnd.recluster.reason

    # ---- 6: reproducibility ------------------------------------------------
    fed3, _ = _build(seed, flaky={FLAKY: 2}, timeout_s=5.0)
    fed3.run(4)
    facts["reproducible"] = _digest(fed3) == facts["digest"]
    say(
        f"\n[4] same seed, second run: adapters identical = {facts['reproducible']} "
        f"(sha256 {facts['digest'][:12]}...)"
    )
    assert facts["reproducible"]

    # ---- 7: the same decision over HTTP -----------------------------------
    facts["http"] = _http_section(seed, say)

    say("\nNOT shown / not claimed: mTLS (G1) and DP mu-tuning are blocked by P3;")
    say("registry and evaluation integration are stubs only (see docs/INTEGRATION_BOUNDARIES.md).")
    say("PHASE II DEMO OK")
    return facts


def _http_section(seed: int, say) -> dict:
    try:
        from fastapi.testclient import TestClient
    except ImportError:  # httpx (a test extra) is missing
        say("\n[5] HTTP re-clustering skipped: install httpx to run it")
        return {"ran": False}
    from cluster import server
    from cluster.schemas.messages import TensorPayload

    saved = (dict(server._clusters), dict(server._membership))
    default = server._state
    try:
        server._clusters.clear()
        server._clusters[server.DEFAULT_CLUSTER_ID] = default
        server._membership.clear()
        api = TestClient(server.app)
        clients, truth = make_clustered_clients(seed=seed, dim=DIM)
        warm = dict(truth)
        warm[WRONG] = WEB
        for cid in (WEB, SCI):
            api.put(f"/clusters/{cid}/members",
                    json={"client_ids": sorted(c for c, g in warm.items() if g == cid)})
        start = random_adapter(DIM, DIM, seed=0)
        for cid in (WEB, SCI):
            for client_id in sorted(c for c, g in warm.items() if g == cid):
                arrays, n, _ = clients[client_id].fit([a.copy() for a in start.to_ndarrays()], {})
                adapter = type(start).from_ndarrays(arrays)
                body = {
                    "client_id": client_id, "round_id": 0, "rank": adapter.rank,
                    "target_modules": list(adapter.target_modules), "alpha": adapter.alpha,
                    "num_layers": 1, "num_examples": int(n),
                    "tensors": [TensorPayload.from_numpy(k, v).model_dump()
                                for k, v in adapter.to_state_dict().items()],
                }
                assert api.post(f"/clusters/{cid}/uploads", json=body).status_code == 201
            assert api.post(f"/clusters/{cid}/aggregate").status_code == 200
        dry = api.post("/recluster").json()
        before = api.get("/clusters").json()["clusters"][WEB]["members"]
        say(
            f"\n[5] POST /recluster (dry run): {dry['method']}, moved={dry['moved']}, "
            f"applied={dry['applied']}"
        )
        assert dry["moved"] == {WRONG: [WEB, SCI]} and not dry["applied"]
        assert api.get("/clusters").json()["clusters"][WEB]["members"] == before
        done = api.post("/recluster", json={"apply": True}).json()
        assert done["applied"] and WRONG in api.get("/clusters").json()["clusters"][SCI]["members"]
        say(f"    POST /recluster {{apply: true}}: {WRONG} now in {SCI}")
        return {"ran": True, "moved": dry["moved"]}
    finally:
        server._clusters.clear()
        server._clusters.update(saved[0])
        server._membership.clear()
        server._membership.update(saved[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.getLogger("flwr").setLevel(logging.ERROR)
    run(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
