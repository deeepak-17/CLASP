"""Cluster adapter aggregation via SVD re-factorization (W4 · G2).

**Ownership note.** D2 assigns aggregation to P2 (cluster service, W5+). This is a
LOCAL edge-side implementation of the same algorithm, written so the edge lane
can produce real cluster adapters and close G2's 3-layer composition instead of
merging against a stub. It is a stand-in for P2's service, not a replacement:
when the real one lands, the edge should consume its output and this module
becomes a test oracle.

The algorithm (D2)
------------------
Naive averaging of A and B factors is wrong, and it is worth being precise about
why. Averaging factors gives

    (1/n Σ Bᵢ)·(1/n Σ Aᵢ)

which contains every cross term Bᵢ·Aⱼ for i≠j — products of one client's output
projection with another's input projection. Those pairs are meaningless: the
factorization of each adapter is only defined up to an invertible rank-r
transform, so client i's B and client j's A do not live in a shared basis. The
result is biased and does not approximate the average update.

So instead:

  1. Reconstruct each client's actual update, ΔWᵢ = sᵢ·Bᵢ·Aᵢ.
  2. Average in ΔW space, weighted by sample count: ΔW̄ = Σ wᵢ·ΔWᵢ.
     This is exact — no cross terms, because the sum happens after each product.
  3. Re-factorize ΔW̄ back to rank r with truncated SVD, so the cluster adapter
     is the same shape as a client one:
         U, S, Vᵀ = svd(ΔW̄)
         B = U_r·√S_r ,  A = √S_r·Vᵀ_r   =>  B·A = best rank-r approx of ΔW̄

Step 3 is lossy by construction: three rank-16 adapters average to something of
rank up to 48, and squeezing that back to 16 discards the tail. That loss is
measured and reported per module (`reconstruction_error`) rather than assumed
negligible — it is the price D2 pays to keep the cluster adapter servable at
rank 16, and it should be visible in the record.

Memory: everything streams module by module. A single ΔW is 2048x2048 fp32
(16 MB); holding all 96 for three clients at once would be ~4.6 GB.

Usage:
    python -m edge.aggregate --clients a/adapter b/adapter c/adapter \\
        --weights 199 239 222 --out cluster_scientific/
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from edge.merge import (
    _ab,
    delta_weights,
    load_adapter,
    module_prefixes,
    save_adapter,
    scaling_of,
    validate_compatibility,
)

DEFAULT_RANK = 16
DEFAULT_NITER = 8      # svd_lowrank power iterations; more = closer to exact SVD


def normalize_weights(weights: Sequence[float]) -> List[float]:
    """FedAvg sample weighting: wᵢ = nᵢ / Σn.

    Deliberately the CLIENT'S DATASET SIZE, not its training budget. The sqrt
    step-budget policy changes how much local work each client does; it must not
    also change how much that client counts in the average, or the size bias
    gets applied twice (see `plan_budgets` in edge.chunking).
    """
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"weights must sum to > 0, got {list(weights)}")
    return [w / total for w in weights]


def refactorize(dw: torch.Tensor, rank: int, niter: int = DEFAULT_NITER
                ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Best rank-`rank` factorization of ΔW, plus the relative error incurred.

    Returns (A, B, rel_error) with A: (rank, in), B: (out, rank) so that B·A is
    the rank-r approximation. Error is Frobenius-relative:
    ||ΔW - B·A||_F / ||ΔW||_F — 0.0 means the truncation lost nothing.
    """
    dw = dw.float()
    q = min(rank, min(dw.shape))
    u, s, v = torch.svd_lowrank(dw, q=q, niter=niter)
    sqrt_s = torch.sqrt(s)
    b = u * sqrt_s.unsqueeze(0)          # (out, q)
    a = (v * sqrt_s.unsqueeze(0)).T      # (q, in)

    denom = torch.linalg.norm(dw)
    rel = (torch.linalg.norm(dw - b @ a) / denom).item() if denom > 0 else 0.0

    if q < rank:      # pad so every module reports the same rank
        # device=/dtype= are load-bearing: b and a inherit dw's device, so an
        # unqualified torch.zeros lands on the CPU and torch.cat raises on a
        # CUDA dw. Unreachable on the real path (2048x2048 at rank 16 gives
        # q == rank), but live from this module's CLI with --rank above a
        # matrix dimension.
        b = torch.cat([b, torch.zeros(b.shape[0], rank - q,
                                      device=b.device, dtype=b.dtype)], dim=1)
        a = torch.cat([a, torch.zeros(rank - q, a.shape[1],
                                      device=a.device, dtype=a.dtype)], dim=0)
    return a, b, rel


def aggregate_svd(adapters: Sequence[Tuple[Dict[str, torch.Tensor], Dict]],
                  weights: Sequence[float], rank: int = DEFAULT_RANK,
                  niter: int = DEFAULT_NITER) -> Tuple[Dict, Dict, Dict]:
    """D2 aggregation: exact weighted average in ΔW space, re-factorized to rank r.

    Streams module by module — one ΔW per client alive at a time, not all 96.
    """
    w = normalize_weights(weights)
    prefixes = sorted({p for sd, _ in adapters for p in module_prefixes(sd)})

    out_sd: Dict[str, torch.Tensor] = {}
    errors: Dict[str, float] = {}
    for prefix in prefixes:
        avg: Optional[torch.Tensor] = None
        for (sd, cfg), wi in zip(adapters, w):
            if f"{prefix}.lora_A.weight" not in sd:
                continue
            term = wi * delta_weights(sd, cfg, [prefix])[prefix]
            avg = term if avg is None else avg + term
            del term
        if avg is None:
            continue
        a, b, rel = refactorize(avg, rank, niter)
        out_sd[f"{prefix}.lora_A.weight"] = a
        out_sd[f"{prefix}.lora_B.weight"] = b
        errors[prefix] = rel
        del avg

    cfg = dict(adapters[0][1])
    cfg.update({"r": rank, "lora_alpha": rank,   # scaling exactly 1.0
                "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {},
                "inference_mode": True})

    vals = list(errors.values())
    stats = {
        "n_modules": len(errors),
        "reconstruction_error_mean": round(sum(vals) / len(vals), 6) if vals else 0.0,
        "reconstruction_error_max": round(max(vals), 6) if vals else 0.0,
        "reconstruction_error_min": round(min(vals), 6) if vals else 0.0,
        "rank": rank,
        "svd_niter": niter,
        "weights_normalized": [round(x, 6) for x in w],
    }
    return out_sd, cfg, stats


def aggregate_naive(adapters: Sequence[Tuple[Dict[str, torch.Tensor], Dict]],
                    weights: Sequence[float]) -> Tuple[Dict, Dict]:
    """Naive per-factor averaging — the ABLATION BASELINE only (D2).

    Averages A and B independently. Kept so the bias can be measured rather than
    asserted; never use it to produce a served cluster adapter.
    """
    w = normalize_weights(weights)
    prefixes = sorted({p for sd, _ in adapters for p in module_prefixes(sd)})
    out_sd: Dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        a_sum = b_sum = None
        for (sd, cfg), wi in zip(adapters, w):
            if f"{prefix}.lora_A.weight" not in sd:
                continue
            a, b = _ab(sd, prefix)
            s = scaling_of(cfg)
            a_t, b_t = wi * a.float(), s * b.float()
            a_sum = a_t if a_sum is None else a_sum + a_t
            b_sum = b_t if b_sum is None else b_sum + b_t
        out_sd[f"{prefix}.lora_A.weight"] = a_sum
        out_sd[f"{prefix}.lora_B.weight"] = b_sum
    cfg = dict(adapters[0][1])
    cfg.update({"lora_alpha": cfg["r"], "use_rslora": False, "inference_mode": True})
    return out_sd, cfg


def exact_average_error(adapters: Sequence[Tuple[Dict, Dict]], weights: Sequence[float],
                        sd: Dict[str, torch.Tensor], cfg: Dict) -> Dict[str, float]:
    """How far an aggregate sits from the EXACT weighted average, per module.

    The yardstick for D2's "unit-tested against the exact product average". Used
    for both the SVD result (small error, from rank truncation) and the naive
    baseline (large error, from cross terms).
    """
    w = normalize_weights(weights)
    worst_rel, mean_rel, n = 0.0, 0.0, 0
    for prefix in module_prefixes(sd):
        exact = None
        for (csd, ccfg), wi in zip(adapters, w):
            if f"{prefix}.lora_A.weight" not in csd:
                continue
            term = wi * delta_weights(csd, ccfg, [prefix])[prefix]
            exact = term if exact is None else exact + term
        if exact is None:
            continue
        got = delta_weights(sd, cfg, [prefix])[prefix]
        denom = torch.linalg.norm(exact)
        rel = (torch.linalg.norm(got - exact) / denom).item() if denom > 0 else 0.0
        worst_rel = max(worst_rel, rel)
        mean_rel += rel
        n += 1
        del exact, got
    return {"max_rel_err": round(worst_rel, 6),
            "mean_rel_err": round(mean_rel / n, 6) if n else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate client adapters into a cluster adapter (D2)")
    ap.add_argument("--clients", nargs="+", required=True, help="client adapter directories")
    ap.add_argument("--weights", nargs="+", type=float, required=True,
                    help="sample counts per client (train files or blocks)")
    ap.add_argument("--names", nargs="+", help="labels for the manifest")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--niter", type=int, default=DEFAULT_NITER)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--compare-naive", action="store_true",
                    help="also measure the naive-averaging ablation baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.clients) != len(args.weights):
        raise SystemExit("--clients and --weights must have the same length")
    names = args.names or [Path(c).parent.name for c in args.clients]

    adapters = [load_adapter(Path(c)) for c in args.clients]
    validate_compatibility([cfg for _, cfg in adapters], names)

    sd, cfg, stats = aggregate_svd(adapters, args.weights, args.rank, args.niter)
    svd_err = exact_average_error(adapters, args.weights, sd, cfg)

    naive_err = None
    if args.compare_naive:
        n_sd, n_cfg = aggregate_naive(adapters, args.weights)
        naive_err = exact_average_error(adapters, args.weights, n_sd, n_cfg)

    save_adapter(Path(args.out), sd, cfg)
    meta = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "W4/G2 cluster aggregation (D2, local stand-in for P2's service)",
        "cluster_id": args.cluster_id,
        "clients": names,
        "client_paths": [str(c) for c in args.clients],
        "sample_weights": args.weights,
        **stats,
        "vs_exact_average": svd_err,
        "naive_baseline_vs_exact_average": naive_err,
        "owner_note": ("D2 assigns this to P2 (W5+). Local implementation so the edge "
                       "lane can close G2 with real cluster adapters."),
    }
    (Path(args.out) / "aggregate_manifest.json").write_text(json.dumps(meta, indent=2),
                                                            encoding="utf-8")
    print(f"cluster adapter '{args.cluster_id}' -> {Path(args.out).resolve()}")
    print(f"  clients        : {', '.join(names)}")
    print(f"  weights        : {stats['weights_normalized']}")
    print(f"  rank           : {stats['rank']} ({stats['n_modules']} modules)")
    print(f"  SVD truncation : mean {stats['reconstruction_error_mean']:.4f}, "
          f"max {stats['reconstruction_error_max']:.4f} relative Frobenius")
    print(f"  vs exact avg   : max {svd_err['max_rel_err']:.4f}, "
          f"mean {svd_err['mean_rel_err']:.4f}")
    if naive_err:
        print(f"  naive baseline : max {naive_err['max_rel_err']:.4f}, "
              f"mean {naive_err['mean_rel_err']:.4f}  <-- ablation, biased by cross terms")


if __name__ == "__main__":
    main()
