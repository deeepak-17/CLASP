"""Phase-II evidence runner for P2 (Weeks 9-13): ``python -m cluster.evidence``.

Runs the real pipeline (FedProx clients -> per-cluster SVD aggregation ->
re-clustering -> redistribution) and writes one JSON record of *measured*
numbers; nothing in the output is hard-coded.

  1. mu_sweep             FedProx mu vs. final loss / stability, per seed (W9 Wed;
                          a toy NON-DP stability experiment — DP tuning is P3-blocked)
  2. dynamic_vs_static    wrong warm-start label: dynamic re-clustering vs the
                          static fallback, same seeds, same rounds (W10 Thu)
  3. reproducibility      the same seeded run twice per seed, adapters byte-compared
                          (G3, W11)
  4. isolation_stress     adversarial clients placed in the wrong cluster (W13 Mon)
  5. three_client_training  one cluster, 3 clients, several rounds, SVD vs naive (G2)
  6. straggler_dropout    timeout straggler, offline client, quorum failure (W7/W13)
  7. static_fallback      too few updates / no group structure / static mode (D12)
  8. summary              aggregates of the above, computed from the records

Requires torch (real clients). Takes well under a minute on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

from cluster.adapter_format import random_adapter
from cluster.clustering import ClusterAssigner, adjusted_rand_index
from cluster.federation import MultiClusterFederation
from cluster.simulation import make_clustered_clients
from cluster.straggler import StragglerPolicy

DIM = 32
LR = 1e-1
SCI_FLIPPED = "cluster-sci/client-2"
WEB, SCI = "cluster-web", "cluster-sci"


def _digest(fed: MultiClusterFederation) -> str:
    h = hashlib.sha256()
    for cid in sorted(fed.cluster_adapters):
        for arr in fed.cluster_adapters[cid].to_ndarrays():
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _cluster_digest(fed: MultiClusterFederation, cluster_id: str) -> str:
    h = hashlib.sha256()
    for arr in fed.cluster_adapters[cluster_id].to_ndarrays():
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _build(
    seed,
    mode="dynamic",
    flip=(),
    mu=0.01,
    lr=LR,
    *,
    groups=None,
    group_shift=1.5,
    policy=None,
    delays=None,
    aggregation="svd",
):
    clients, truth = make_clustered_clients(
        groups, seed=seed, dim=DIM, mu=mu, lr=lr, group_shift=group_shift
    )
    for cid, delay in (delays or {}).items():
        clients[cid].simulated_delay_s = delay
    warm = dict(truth)
    for cid in flip:
        warm[cid] = WEB if truth[cid] == SCI else SCI
    cluster_ids = [c for c in (WEB, SCI) if c in set(truth.values())]
    assigner = ClusterAssigner(warm, cluster_ids, mode=mode, seed=0)
    fed = MultiClusterFederation(
        clients, assigner, random_adapter(DIM, DIM, seed=0), policy=policy, aggregation=aggregation
    )
    return fed, truth


def _mean_loss(rnd) -> float:
    vals = [r.metrics.mean_loss for r in rnd.clusters.values() if r.metrics.mean_loss is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _ari_vs_truth(fed, truth) -> float:
    ids = sorted(truth)
    return adjusted_rand_index([fed.assigner.assignment[c] == WEB for c in ids],
                               [truth[c] == WEB for c in ids])


def mu_sweep(seeds: list[int], rounds: int) -> list[dict]:
    """FedProx mu sweep (toy, no DP). ``stable`` = every round loss finite."""
    out = []
    for mu in (0.0, 0.01, 0.1, 1.0, 10.0, 50.0):
        per_seed = []
        for seed in seeds:
            fed, _ = _build(seed, mode="static", mu=mu)
            rows = fed.run(rounds)
            losses = [_mean_loss(r) for r in rows]
            per_seed.append(
                {
                    "seed": seed,
                    "stable": bool(np.isfinite(losses).all()),
                    "round_mean_losses": losses,
                    "final_mean_loss": losses[-1],
                }
            )
        stable = [r for r in per_seed if r["stable"]]
        out.append(
            {
                "mu": mu,
                "lr_times_mu": mu * LR,
                "stable_in_all_seeds": len(stable) == len(per_seed),
                "mean_final_loss_over_stable_seeds": (
                    float(np.mean([r["final_mean_loss"] for r in stable])) if stable else None
                ),
                "per_seed": per_seed,
            }
        )
    return out


def dynamic_vs_static(seeds: list[int], rounds: int) -> list[dict]:
    out = []
    for seed in seeds:
        for label, flip in (("correct_warm_start", ()), ("one_wrong_label", (SCI_FLIPPED,))):
            rec: dict = {"seed": seed, "scenario": label}
            for mode in ("dynamic", "static"):
                fed, truth = _build(seed, mode=mode, flip=flip)
                rows = fed.run(rounds)
                recl = next((r.recluster for r in rows if r.recluster), None)
                rec[mode] = {
                    "final_mean_loss": _mean_loss(rows[-1]),
                    "ari_vs_true_groups": _ari_vs_truth(fed, truth),
                    "recluster_method": recl.method if recl else None,
                    "recluster_reason": recl.reason if recl else None,
                    "moved": {c: list(v) for c, v in (recl.moved if recl else {}).items()},
                    "comparison": recl.comparison if recl else None,
                    "isolation_violations": fed.isolation_violations(),
                }
            out.append(rec)
    return out


def reproducibility(seeds: list[int], rounds: int) -> list[dict]:
    out = []
    for seed in seeds:
        digests = []
        for _ in range(2):
            fed, _ = _build(seed, flip=(SCI_FLIPPED,))
            fed.run(rounds)
            digests.append(_digest(fed))
        out.append(
            {"seed": seed, "rounds": rounds, "sha256_run1": digests[0],
             "sha256_run2": digests[1], "identical": digests[0] == digests[1]}
        )
    return out


def isolation_stress(seed: int, rounds: int) -> dict:
    flips = (SCI_FLIPPED, "cluster-web/client-0")  # adversarial placements
    fed, truth = _build(seed, flip=flips)
    fed.run(rounds)
    return {
        "seed": seed,
        "wrongly_placed": list(flips),
        "final_assignment_matches_truth": fed.assigner.assignment == truth,
        "ari_vs_true_groups": _ari_vs_truth(fed, truth),
        "isolation_violations": fed.isolation_violations(),
        "recluster_history": [
            {"after_round": h.completed_rounds, "method": h.method, "reason": h.reason,
             "moved": {c: list(v) for c, v in h.moved.items()}}
            for h in fed.assigner.history[1:]
        ],
    }


def three_client_training(seed: int, rounds: int) -> dict:
    """One cluster, 3 real FedProx clients, ``rounds`` rounds, SVD vs naive (G2)."""
    out: dict = {"seed": seed, "rounds": rounds, "clients": 3}
    for agg in ("svd", "naive"):
        fed, _ = _build(seed, mode="static", groups={WEB: 3}, aggregation=agg)
        rows = fed.run(rounds)
        losses = [_mean_loss(r) for r in rows]
        out[agg] = {
            "round_mean_losses": losses,
            "loss_decreased": bool(losses[-1] < losses[0]),
            "contributors_per_round": [len(r.clusters[WEB].contributors) for r in rows],
            "all_delivered": all(r.clusters[WEB].delivery.all_delivered for r in rows),
        }
    return out


def _round_view(rnd) -> dict:
    return {
        cid: {"contributors": len(res.contributors), "aggregated": res.aggregated,
              "skipped": dict(res.skipped)}
        for cid, res in rnd.clusters.items()
    }


def straggler_dropout(seed: int) -> dict:
    out: dict = {"seed": seed}

    # (a) a straggler: client-2 of cluster-web reports 30 s, timeout is 5 s
    fed, _ = _build(seed, mode="static", policy=StragglerPolicy(timeout_s=5.0),
                    delays={"cluster-web/client-2": 30.0})
    out["timeout_straggler"] = {
        "policy": {"timeout_s": 5.0}, "client_delay_s": 30.0,
        "rounds": [_round_view(r) for r in fed.run(3)],
        "isolation_violations": fed.isolation_violations(),
    }

    # (b) dropout: client-1 of cluster-sci is offline in rounds 1-2, back in round 3
    fed, _ = _build(seed, mode="static")
    rounds = [fed.run_round(offline={"cluster-sci/client-1"}) for _ in range(2)]
    rounds.append(fed.run_round())
    out["dropout_and_rejoin"] = {
        "offline_rounds": [0, 1],
        "rounds": [_round_view(r) for r in rounds],
        "isolation_violations": fed.isolation_violations(),
    }

    # (c) quorum failure: min_clients=2 but only 1 cluster-web client answers in round 1
    fed, _ = _build(seed, mode="static", policy=StragglerPolicy(min_clients=2))
    fed.run_round()
    before = {cid: _cluster_digest(fed, cid) for cid in (WEB, SCI)}
    rnd = fed.run_round(offline={"cluster-web/client-1", "cluster-web/client-2"})
    after = {cid: _cluster_digest(fed, cid) for cid in (WEB, SCI)}
    recovered = fed.run_round()
    out["quorum_failure"] = {
        "policy": {"min_clients": 2},
        "failed_round": _round_view(rnd),
        "web_aggregated": rnd.clusters[WEB].aggregated,
        "web_adapter_unchanged": before[WEB] == after[WEB],
        "sci_adapter_updated": before[SCI] != after[SCI],
        "recovered_round": _round_view(recovered),
    }
    return out


def static_fallback(seed: int) -> dict:
    out: dict = {"seed": seed}

    # (a) too few updates at the re-cluster round: only 1 client answers in round 2
    fed, truth = _build(seed, flip=(SCI_FLIPPED,))
    keep = "cluster-web/client-0"
    fed.run_round()
    rnd = fed.run_round(offline=set(truth) - {keep})
    rec = rnd.recluster
    out["too_few_updates"] = {
        "method": rec.method, "reason": rec.reason,
        "assignment_unchanged": rnd.assignment_before == rnd.assignment_after,
    }

    # (b) no group structure (group_shift=0): nothing for k-means to find
    fed, truth = _build(seed, flip=(SCI_FLIPPED,), group_shift=0.0)
    fed.run(3)
    rec = fed.assigner.history[-1]
    out["no_group_structure"] = {
        "method": rec.method, "reason": rec.reason,
        "cohesion": rec.cohesion, "separation": rec.separation,
        "wrong_label_kept": fed.assigner.assignment[SCI_FLIPPED] == WEB,
    }

    # (c) the static baseline never re-clusters
    fed, _ = _build(seed, mode="static", flip=(SCI_FLIPPED,))
    fed.run(4)
    out["static_mode"] = {
        "history_methods": [h.method for h in fed.assigner.history],
        "wrong_label_kept": fed.assigner.assignment[SCI_FLIPPED] == WEB,
    }
    return out


def summarize(result: dict) -> dict:
    """Aggregates computed from the records above (no new measurements)."""
    summary: dict = {}
    for scenario in ("correct_warm_start", "one_wrong_label"):
        rows = [r for r in result["dynamic_vs_static"] if r["scenario"] == scenario]
        entry: dict = {"seeds": len(rows)}
        for mode in ("dynamic", "static"):
            entry[mode] = {
                "mean_ari_vs_true_groups": float(np.mean([r[mode]["ari_vs_true_groups"] for r in rows])),
                "mean_final_loss": float(np.mean([r[mode]["final_mean_loss"] for r in rows])),
                "seeds_recovering_true_groups": sum(
                    r[mode]["ari_vs_true_groups"] == 1.0 for r in rows
                ),
            }
        summary[scenario] = entry
    summary["reproducible_in_all_seeds"] = all(r["identical"] for r in result["reproducibility"])
    summary["isolation_violations_total"] = (
        sum(len(r[m]["isolation_violations"]) for r in result["dynamic_vs_static"]
            for m in ("dynamic", "static"))
        + len(result["isolation_stress"]["isolation_violations"])
        + len(result["straggler_dropout"]["timeout_straggler"]["isolation_violations"])
        + len(result["straggler_dropout"]["dropout_and_rejoin"]["isolation_violations"])
    )
    return summary


def _versions() -> dict[str, str]:
    import flwr
    import torch

    return {"torch": torch.__version__, "flwr": flwr.__version__}


def render_markdown(result: dict) -> str:
    """Human-readable view of ``result`` (every figure is read from the record)."""
    cfg, summ = result["config"], result["summary"]
    env = cfg["environment"]
    lines = [
        "# P2 Cluster - Phase II evidence (generated)",
        "",
        (
            "Generated by `python -m cluster.evidence`; every number below is read from "
            "`phase2_cluster_evidence.json` produced by the same run. Synthetic toy-LoRA "
            f"regression on CPU (dim {cfg['dim']}, rank {cfg['rank']}, lr {cfg['lr']}); "
            "not DP, not the real model."
        ),
        "",
        (
            f"Seeds {cfg['seeds']}, {cfg['rounds']} rounds, python {env['python']}, "
            f"torch {env['torch']}, flwr {env['flwr']}, numpy {env['numpy']}, "
            f"{env['platform']}; run took {result['elapsed_s']:.0f} s."
        ),
        "",
        "## Dynamic re-clustering vs static (mean over seeds)",
        "",
        "| scenario | mode | mean ARI vs true groups | seeds recovering truth | mean final loss |",
        "|---|---|---|---|---|",
    ]
    for scenario in ("correct_warm_start", "one_wrong_label"):
        entry = summ[scenario]
        for mode in ("dynamic", "static"):
            e = entry[mode]
            lines.append(
                f"| {scenario} | {mode} | {e['mean_ari_vs_true_groups']:.3f} | "
                f"{e['seeds_recovering_true_groups']}/{entry['seeds']} | {e['mean_final_loss']:.4f} |"
            )
    lines += [
        "",
        (
            "Reproducible in all seeds (adapters byte-identical across two runs): "
            f"**{summ['reproducible_in_all_seeds']}**. "
            f"Isolation violations found across all runs: **{summ['isolation_violations_total']}**."
        ),
        "",
        "## FedProx mu sweep (toy, non-DP)",
        "",
        "| mu | lr*mu | stable in all seeds | mean final loss (stable seeds) |",
        "|---|---|---|---|",
    ]
    for r in result["mu_sweep"]:
        loss = r["mean_final_loss_over_stable_seeds"]
        lines.append(
            f"| {r['mu']} | {r['lr_times_mu']:.3g} | {r['stable_in_all_seeds']} | "
            f"{'diverged' if loss is None else f'{loss:.4f}'} |"
        )
    t = result["three_client_training"]
    lines += ["", f"## 3-client training (seed {t['seed']}, {t['rounds']} rounds)", "",
              "| aggregation | round mean losses | loss decreased |", "|---|---|---|"]
    for agg in ("svd", "naive"):
        losses = ", ".join(f"{v:.3f}" for v in t[agg]["round_mean_losses"])
        lines.append(f"| {agg} | {losses} | {t[agg]['loss_decreased']} |")
    sd, q = result["straggler_dropout"], result["straggler_dropout"]["quorum_failure"]
    lines += [
        "",
        f"## Straggler / dropout / quorum (seed {sd['seed']})",
        "",
        (
            "- timeout straggler (client-2 of cluster-web, 30 s vs 5 s timeout): contributors "
            f"per round {[r['cluster-web']['contributors'] for r in sd['timeout_straggler']['rounds']]}"
        ),
        (
            "- dropout and rejoin (client-1 of cluster-sci offline in rounds 1-2): sci "
            f"contributors per round "
            f"{[r['cluster-sci']['contributors'] for r in sd['dropout_and_rejoin']['rounds']]}"
        ),
        (
            f"- quorum failure (min_clients=2, 1 answer): web aggregated={q['web_aggregated']}, "
            f"web adapter unchanged={q['web_adapter_unchanged']}, sci adapter updated="
            f"{q['sci_adapter_updated']}, next round web aggregated="
            f"{q['recovered_round']['cluster-web']['aggregated']}"
        ),
    ]
    fb = result["static_fallback"]
    lines += [
        "",
        f"## Static fallback (seed {fb['seed']})",
        "",
        f"- too few updates: {fb['too_few_updates']['method']} - {fb['too_few_updates']['reason']}",
        (
            f"- no group structure: {fb['no_group_structure']['method']} - "
            f"{fb['no_group_structure']['reason']}; wrong label kept: "
            f"{fb['no_group_structure']['wrong_label_kept']}"
        ),
        f"- static mode history: {fb['static_mode']['history_methods']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--output", default="demo_runs/phase2_cluster_evidence.json")
    args = ap.parse_args(argv)
    logging.getLogger("flwr").setLevel(logging.ERROR)
    t0 = time.monotonic()
    first = args.seeds[0]
    result = {
        "config": {
            "seeds": args.seeds, "rounds": args.rounds, "dim": DIM, "rank": 16, "lr": LR,
            "clusters": [WEB, SCI], "clients_per_cluster": 3, "recluster_after_round": 2,
            "note": "synthetic toy-LoRA regression on CPU; not DP, not the real model",
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, **_versions(),
            },
        },
        "mu_sweep": mu_sweep(args.seeds, args.rounds),
        "dynamic_vs_static": dynamic_vs_static(args.seeds, args.rounds),
        "reproducibility": reproducibility(args.seeds, args.rounds),
        "isolation_stress": isolation_stress(first, args.rounds),
        "three_client_training": three_client_training(first, args.rounds),
        "straggler_dropout": straggler_dropout(first),
        "static_fallback": static_fallback(first),
    }
    result["summary"] = summarize(result)
    result["elapsed_s"] = time.monotonic() - t0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
