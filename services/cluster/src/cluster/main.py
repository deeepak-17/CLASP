"""
CLASP — P2 Cluster Service  |  Author: Prasanth
Run: python -m cluster.main
"""
from __future__ import annotations

import io
import logging
import sys
import time

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.getLogger("flwr").setLevel(logging.ERROR)

from cluster.adapter_format import random_adapter
from cluster.simulation import make_real_clients, run_federated


def main():
    clients = make_real_clients(n=3, dim=32, local_steps=20, seed=42)
    t0      = time.monotonic()
    rounds  = run_federated(clients, initial_adapter=random_adapter(32, 32, seed=0),
                            num_rounds=4, aggregation="svd")
    dur     = time.monotonic() - t0
    losses  = [r.mean_loss for r in rounds]

    print()
    print("  Federated Training  (3 clients, 4 rounds, FedProx + SVD)")
    print(f"  {'Round':<7} {'Mean Loss':<12} {'C0':<10} {'C1':<10} {'C2':<10}")
    for r in rounds:
        cl = r.client_losses
        print(f"  {r.round_id:<7} {r.mean_loss:<12.4f} "
              f"{cl.get('client-0', 0):<10.4f} "
              f"{cl.get('client-1', 0):<10.4f} "
              f"{cl.get('client-2', 0):<10.4f}")

    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print()
    print(f"  Loss  : {losses[0]:.4f}  ->  {losses[-1]:.4f}  ({improvement:.1f}% improvement)")
    print(f"  Time  : {dur:.2f}s")
    print()

    # ── 2. SVD vs Naive Aggregation Error ─────────────────────────
    import numpy as np

    from cluster.aggregation import aggregate_naive, aggregate_svd, exact_average_delta

    def _trained(seed):
        rng = np.random.default_rng(seed)
        ad  = random_adapter(32, 32, seed=seed)
        for m in ad.target_modules:
            ad.modules[0][m]["lora_B"] = rng.normal(0.1, 0.1, (32, 16)).astype("f4")
        return ad

    ads    = [_trained(10), _trained(11), _trained(12)]
    exact  = exact_average_delta(iter(ads), [1, 1, 1])
    svd_m  = aggregate_svd(iter(ads), [1, 1, 1], rank=16)
    naive_m= aggregate_naive(iter(ads), [1, 1, 1])

    se = sum(np.linalg.norm(svd_m.delta_w(m)   - exact[m]) for m in svd_m.target_modules)
    ne = sum(np.linalg.norm(naive_m.delta_w(m) - exact[m]) for m in svd_m.target_modules)

    print("  SVD vs Naive Aggregation Error")
    print(f"  SVD error   : {se:.6f}  <-- my mathematical method")
    print(f"  Naive error : {ne:.6f}  <-- simple open-source averaging")
    print(f"  Result      : SVD is {ne/se:.1f}x more accurate for LoRA")
    print()

if __name__ == "__main__":
    main()

