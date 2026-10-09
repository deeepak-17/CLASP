"""Cluster adapter aggregation, delegated to P2's cluster package (D2).

**Ownership.** D2 belongs to P2. This module used to carry its own copy of the
SVD aggregation — a stand-in written in W4 so the edge lane could close G2
before the cluster service existed, with the stated plan that "when the real
one lands, the edge should consume its output". It has landed
(``cluster.aggregation``), so the edge no longer does any D2 mathematics. What
is left here is the part that genuinely belongs to the edge: getting a PEFT
adapter directory's tensors into the cluster's ``LoRAAdapter`` and the result
back out again under the same PEFT keys, so ``edge.merge`` can compose it.

    PEFT state_dict (torch) --to_cluster_adapter--> cluster.LoRAAdapter
        --cluster.aggregation.aggregate_svd_lowrank--> aggregated LoRAAdapter
        --from_cluster_adapter--> PEFT state_dict (torch), same key names

The one piece of arithmetic this module still does is folding PEFT's scaling
``s = lora_alpha / r`` into ``lora_B`` on the way in. The cluster reconstructs
an update as ``B @ A`` with no scaling (``LoRAAdapter.delta_w``), so an adapter
whose ``s != 1`` would otherwise be mis-weighted in the average. Folding keeps
``ΔW = s·B·A`` exact and lets the cluster's unscaled product mean what it says;
it is the same convention ``edge.wire.upload_payload`` applies on seam A.

``cluster`` is imported lazily, inside the functions that need it, so the rest
of ``edge`` stays importable on a machine (or CI leg) that does not have the
cluster package installed.

Usage:
    python -m edge.aggregate --clients a/adapter b/adapter c/adapter \\
        --weights 199 239 222 --cluster-id scientific --out cluster_scientific/
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from edge.merge import (
    load_adapter,
    save_adapter,
    scaling_of,
    validate_compatibility,
)
from edge.wire import describe, parse_key

DEFAULT_RANK = 16

#: Which of P2's aggregators the served SVD path uses. ``aggregate_svd_lowrank``
#: is the exact D2 pipeline evaluated through the low-rank factors (P2 measured
#: it identical to the dense ``aggregate_svd`` reference to six decimals on the
#: real web cluster, 2.5 s against 865 s).
SVD_AGGREGATOR = "cluster.aggregation.aggregate_svd_lowrank"
NAIVE_AGGREGATOR = "cluster.aggregation.aggregate_naive"

Adapter = Tuple[Dict[str, torch.Tensor], Dict]


def _cluster_aggregation():
    """Import P2's aggregation module, or say exactly what is missing."""
    try:
        from cluster import aggregation
    except ImportError as exc:  # pragma: no cover - exercised only without cluster
        raise ImportError(
            "edge.aggregate delegates D2 aggregation to P2's cluster package, "
            "which is not installed. Install it with "
            "`pip install -e services/cluster`.") from exc
    return aggregation


def normalize_weights(weights: Sequence[float]) -> List[float]:
    """FedAvg sample weighting: wᵢ = nᵢ / Σn — recorded in the manifest.

    The cluster normalizes internally as well; this is kept so the manifest
    states the weights that were in force. They are the clients' DATASET sizes,
    not their training budgets: the sqrt step-budget policy changes how much
    local work a client does, and must not also change how much it counts in
    the average (see ``plan_budgets`` in edge.chunking).
    """
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"weights must sum to > 0, got {list(weights)}")
    return [w / total for w in weights]


def to_cluster_adapter(sd: Dict[str, torch.Tensor], cfg: Dict):
    """A PEFT adapter -> ``cluster.LoRAAdapter`` with its scaling folded into B.

    Layer count, target modules and rank are read off the tensors
    (``edge.wire.describe``), not trusted from the config, so a truncated or
    mis-saved adapter fails here rather than inside the aggregator. The result
    declares ``alpha == rank`` (scaling exactly 1.0), so its ``B @ A`` is the
    adapter's true ΔW.
    """
    from cluster.adapter_format import LoRAAdapter

    arrays = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    info = describe(arrays)
    if info["rank"] != cfg["r"]:
        raise ValueError(f"adapter_config r={cfg['r']} disagrees with the tensors' "
                         f"rank {info['rank']}")
    s = np.float32(scaling_of(cfg))
    folded = {k: (a * s if parse_key(k)[2] == "lora_B" else a) for k, a in arrays.items()}
    return LoRAAdapter.from_state_dict(
        folded, rank=info["rank"], alpha=float(info["rank"]),
        target_modules=tuple(info["target_modules"]), num_layers=info["num_layers"])


def _key_map(adapters: Sequence[Adapter]) -> Dict[Tuple[int, str, str], str]:
    """(layer, module, part) -> the PEFT key name the clients used for it."""
    out: Dict[Tuple[int, str, str], str] = {}
    for sd, _ in adapters:
        for key in sd:
            out.setdefault(parse_key(key), key)
    return out


def from_cluster_adapter(merged, key_map: Dict[Tuple[int, str, str], str]
                         ) -> Dict[str, torch.Tensor]:
    """An aggregated ``LoRAAdapter`` -> fp32 torch tensors under the clients' keys."""
    sd: Dict[str, torch.Tensor] = {}
    for layer in merged.layer_indices:
        for module in merged.target_modules:
            for part in ("lora_A", "lora_B"):
                key = key_map[(layer, module, part)]
                arr = np.ascontiguousarray(merged.modules[layer][module][part],
                                           dtype=np.float32)
                sd[key] = torch.from_numpy(arr)
    return sd


def _aggregated_cfg(reference_cfg: Dict, rank: int) -> Dict:
    cfg = dict(reference_cfg)
    cfg.update({"r": rank, "lora_alpha": rank,   # scaling exactly 1.0
                "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {},
                "inference_mode": True})
    return cfg


def aggregate_svd(adapters: Sequence[Adapter], weights: Sequence[float],
                  rank: int = DEFAULT_RANK) -> Tuple[Dict, Dict, Dict]:
    """D2 SVD aggregation, computed by ``cluster.aggregation.aggregate_svd_lowrank``.

    Returns (state_dict, adapter_config, stats). The reconstruction errors in
    ``stats`` are the cluster's own exact per-module truncation errors.
    """
    agg = _cluster_aggregation()
    w = normalize_weights(weights)
    merged, errors = agg.aggregate_svd_lowrank(
        [to_cluster_adapter(sd, cfg) for sd, cfg in adapters], list(weights), rank=rank)
    sd = from_cluster_adapter(merged, _key_map(adapters))
    vals = list(errors.values())
    stats = {
        "aggregator": SVD_AGGREGATOR,
        "n_modules": len(errors),
        "reconstruction_error_mean": round(sum(vals) / len(vals), 6) if vals else 0.0,
        "reconstruction_error_max": round(max(vals), 6) if vals else 0.0,
        "reconstruction_error_min": round(min(vals), 6) if vals else 0.0,
        "rank": rank,
        "weights_normalized": [round(x, 6) for x in w],
    }
    return sd, _aggregated_cfg(adapters[0][1], rank), stats


def aggregate_naive(adapters: Sequence[Adapter], weights: Sequence[float]
                    ) -> Tuple[Dict, Dict]:
    """Naive per-factor averaging — D2's ABLATION BASELINE, computed by
    ``cluster.aggregation.aggregate_naive``. Never serve its output."""
    agg = _cluster_aggregation()
    merged = agg.aggregate_naive(
        [to_cluster_adapter(sd, cfg) for sd, cfg in adapters], list(weights))
    sd = from_cluster_adapter(merged, _key_map(adapters))
    return sd, _aggregated_cfg(adapters[0][1], merged.rank)


def exact_average_error(adapters: Sequence[Adapter], weights: Sequence[float],
                        sd: Dict[str, torch.Tensor], cfg: Dict) -> Dict[str, float]:
    """How far an aggregate sits from the EXACT weighted-average ΔW, per module.

    Computed by ``cluster.aggregation.layerwise_exact_average_error`` — D2's
    "unit-tested against the exact product average" yardstick. Small for the
    SVD path (rank truncation only), large for the naive one (cross terms).
    """
    agg = _cluster_aggregation()
    return agg.layerwise_exact_average_error(
        [to_cluster_adapter(csd, ccfg) for csd, ccfg in adapters], list(weights),
        to_cluster_adapter(sd, cfg))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate client adapters into a cluster adapter (D2, via P2's cluster package)")
    ap.add_argument("--clients", nargs="+", required=True, help="client adapter directories")
    ap.add_argument("--weights", nargs="+", type=float, required=True,
                    help="sample counts per client (train files or blocks)")
    ap.add_argument("--names", nargs="+", help="labels for the manifest")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
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

    sd, cfg, stats = aggregate_svd(adapters, args.weights, args.rank)
    svd_err = exact_average_error(adapters, args.weights, sd, cfg)

    naive_err = None
    if args.compare_naive:
        n_sd, n_cfg = aggregate_naive(adapters, args.weights)
        naive_err = exact_average_error(adapters, args.weights, n_sd, n_cfg)

    save_adapter(Path(args.out), sd, cfg)
    meta = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "cluster aggregation (D2), computed by P2's cluster package",
        "cluster_id": args.cluster_id,
        "clients": names,
        "client_paths": [str(c) for c in args.clients],
        "sample_weights": args.weights,
        **stats,
        "vs_exact_average": svd_err,
        "naive_baseline_vs_exact_average": naive_err,
    }
    (Path(args.out) / "aggregate_manifest.json").write_text(json.dumps(meta, indent=2),
                                                            encoding="utf-8")
    print(f"cluster adapter '{args.cluster_id}' -> {Path(args.out).resolve()}")
    print(f"  aggregator     : {stats['aggregator']}")
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
