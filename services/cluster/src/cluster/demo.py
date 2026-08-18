"""Week 5 demo (P2): 3-client federated aggregation round.

Reuses the existing P2 pipeline end to end and adds nothing new to the
aggregation/training math itself:

    make_real_clients / make_dummy_clients   (simulation.py, client.py)
    -> LoRAClient.fit  == local FedProx training per client (client.py)
    -> run_round                              (simulation.py)
    -> aggregate_svd / truncated_svd_refactor (aggregation.py)
    -> RoundMetrics                           (schemas/messages.py)

This module is presentation + bookkeeping only: it calls the existing
functions, times the round, prints a readable stage-by-stage trace, fills in
the existing ``RoundMetrics`` schema, and writes a JSON artifact so a run can
be kept as backup/dry-run evidence.

Run:
    python -m cluster.demo
    python -m cluster.demo --seed 42 --clients 3 --aggregation svd

Engine selection:
    --engine auto      (default) real FedProx clients (client.py:LoRAClient)
                        if torch is importable, else the Week 1/2 DummyClient
                        round-trip fallback so the demo still completes.
    --engine fedprox    force real FedProx clients; errors clearly if torch
                        is not installed instead of silently degrading.
    --engine dummy      force the DummyClient round-trip fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cluster.adapter_format import DEFAULT_RANK
from cluster.schemas.messages import RoundMetrics
from cluster.simulation import make_dummy_clients, make_real_clients, random_adapter, run_round

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# services/cluster/src/cluster/demo.py -> services/cluster/demo_runs
DEMO_RUNS_DIR = Path(__file__).resolve().parents[2] / "demo_runs"

_BANNER = "=" * 64
_RULE = "-" * 64


def _resolve_engine(requested: str) -> str:
    if requested == "fedprox":
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "engine=fedprox requires torch, which is not installed in this "
                "environment. Install it with `pip install torch` (see "
                "services/cluster/README.md) or run with --engine dummy / "
                "--engine auto to use the Week 1/2 round-trip fallback."
            )
        return "fedprox"
    if requested == "dummy":
        return "dummy"
    # auto
    return "fedprox" if _TORCH_AVAILABLE else "dummy"


def _build_clients(engine: str, n: int, dim: int, rank: int, mu: float,
                    local_steps: int, samples_per_client: int, seed: int):
    if engine == "fedprox":
        return make_real_clients(
            n=n, dim=dim, rank=rank, mu=mu, local_steps=local_steps,
            samples_per_client=samples_per_client, seed=seed,
        )
    return make_dummy_clients(n=n)


def run_demo(
    clients: int = 3,
    seed: int = 42,
    dim: int = 32,
    rank: int = DEFAULT_RANK,
    mu: float = 0.01,
    local_steps: int = 20,
    samples_per_client: int = 64,
    aggregation: str = "svd",
    engine: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one 3-client aggregation round and return a JSON-serializable result.

    Every step below calls straight into the existing P2 implementation; this
    function only sequences the calls, times them, and narrates the result.
    """
    resolved_engine = _resolve_engine(engine)

    def log(msg: str = "") -> None:
        if verbose:
            print(msg)

    log(_BANNER)
    log(" CLASP Cluster Layer (P2) - Week 5 Demo")
    log(" 3-client federated aggregation round")
    log(_BANNER)
    log(f"engine              : {resolved_engine}"
        + ("" if resolved_engine == "fedprox" else
           "  (torch not available -> DummyClient round-trip fallback, W1/W2)"))
    log(f"num_clients         : {clients}")
    log(f"aggregation method  : {aggregation}")
    log(f"svd_rank (requested): {rank}")
    log(f"seed                : {seed}")
    log(_RULE)

    party = _build_clients(
        resolved_engine, n=clients, dim=dim, rank=rank, mu=mu,
        local_steps=local_steps, samples_per_client=samples_per_client, seed=seed,
    )
    log(f"[init] {len(party)} clients initialized ({resolved_engine} engine)")
    for c in party:
        log(f"[init]   - {c.client_id}: ready")

    global_adapter = random_adapter(dim, dim, rank=rank, seed=seed)
    log(f"[broadcast] initial global adapter ready "
        f"(rank={global_adapter.rank}, modules={list(global_adapter.target_modules)})")
    log(_RULE)

    log("[fit] running local training on all clients ...")
    t0 = time.perf_counter()
    result = run_round(
        clients=party,
        global_adapter=global_adapter,
        round_id=0,
        aggregation=aggregation,
    )
    duration_s = time.perf_counter() - t0

    for client_id, loss in result.client_losses.items():
        log(f"[fit]   - {client_id}: local training complete (loss={loss:.6f})")
    if not result.client_losses:
        log(f"[fit]   - {result.num_clients} client(s) completed round-trip "
            f"(dummy engine reports no loss)")
    log(f"[collect] {result.num_clients}/{clients} client updates collected")
    log(f"[aggregate] method={aggregation} rank={result.adapter.rank} ... done")
    log(_RULE)

    status = "success" if result.num_clients == clients else "partial"

    metrics = RoundMetrics(
        round_id=result.round_id,
        num_clients=result.num_clients,
        mean_loss=result.mean_loss,
        duration_s=duration_s,
        aggregation=aggregation,
    )

    log("Round metrics")
    log(f"  round_id           : {metrics.round_id}")
    log(f"  num_clients        : {metrics.num_clients}")
    log(f"  training_time_s    : {metrics.duration_s:.4f}")
    log(f"  client_losses      : {result.client_losses}")
    log(f"  mean_loss          : {metrics.mean_loss}")
    log(f"  aggregation        : {metrics.aggregation}")
    log(f"  svd_rank           : {result.adapter.rank}")
    log(f"  aggregation_status : {status}")
    log(_BANNER)
    log(f"FINAL ROUND STATUS  : {status.upper()}")
    log(_BANNER)

    adapter_summary = {
        module: {
            "lora_A_shape": list(pair["lora_A"].shape),
            "lora_B_shape": list(pair["lora_B"].shape),
        }
        for module, pair in result.adapter.modules.items()
    }

    return {
        "week": 5,
        "role": "P2 Cluster",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "clients": clients,
            "seed": seed,
            "dim": dim,
            "rank": rank,
            "mu": mu,
            "local_steps": local_steps,
            "samples_per_client": samples_per_client,
            "aggregation": aggregation,
            "engine_requested": engine,
        },
        "engine_used": resolved_engine,
        "round_metrics": metrics.model_dump(mode="json"),
        "client_losses": result.client_losses,
        "final_adapter": adapter_summary,
        "status": status,
    }


def _save_backup(result: dict[str, Any], output: Path | None) -> Path:
    DEMO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = DEMO_RUNS_DIR / f"round_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal weight")
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--samples-per-client", type=int, default=64)
    parser.add_argument("--aggregation", choices=["svd", "naive"], default="svd")
    parser.add_argument("--engine", choices=["auto", "fedprox", "dummy"], default="auto")
    parser.add_argument("--output", type=Path, default=None,
                         help="where to write the backup JSON (default: demo_runs/round_<ts>.json)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = run_demo(
            clients=args.clients,
            seed=args.seed,
            dim=args.dim,
            rank=args.rank,
            mu=args.mu,
            local_steps=args.local_steps,
            samples_per_client=args.samples_per_client,
            aggregation=args.aggregation,
            engine=args.engine,
            verbose=not args.quiet,
        )
    except Exception as exc:  # noqa: BLE001 - top-level demo entry point
        print(f"[FAIL] demo round did not complete: {exc}", file=sys.stderr)
        return 1

    path = _save_backup(result, args.output)
    if not args.quiet:
        print(f"backup written to : {path}")

    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
