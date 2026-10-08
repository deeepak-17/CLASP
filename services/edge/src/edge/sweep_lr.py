"""LR / step-config sweep for stable convergence (P1 Edge, W3 · E3.2).

E3.1 only asked that loss go down. E3.2 asks for a config that is *stable* and
*repeatable*: no NaN, no divergence, no memorization collapse, and the same
numbers when you run it again. That is a search, so this runs the grid and
records it rather than leaving the choice to taste.

Each cell is a fresh `python -m edge.train_client` subprocess. Subprocess rather
than an in-process loop on purpose: a diverged run leaves poisoned optimizer and
RNG state behind, and re-using a process would let one cell contaminate the
next. The cost is a ~10 s model reload per cell, which is noise against a
multi-minute run.

Three signals per cell, in priority order:
  1. `ok`            - did it finish, or did the divergence guard abort it
  2. `ppl_delta`     - held-out perplexity vs base. NEGATIVE is good. Positive
                       means the adapter memorized the repo, which is the
                       failure a train-loss curve cannot see
  3. `last_loss`     - train loss at the end, only meaningful once 1 and 2 pass

Base held-out perplexity is measured once per client and then passed to the
remaining cells with `--base-ppl`: the base model is identical in every config,
so re-measuring is pure wall time (~4 min/run on numpy's 22-file held-out set).

Usage:
    python -m edge.sweep_lr --clients web/client-flask --lrs 5e-5 1e-4 2e-4 5e-4
    python -m edge.sweep_lr --clients scientific/client-numpy --lrs 1e-4 2e-4 --repeat 2
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_LRS = (5e-5, 1e-4, 2e-4, 5e-4)
DEFAULT_GRAD_ACCUMS = (1,)


def cell_tag(client: str, lr: float, grad_accum: int, rep: int) -> str:
    return f"{client.split('/')[-1]}__lr{lr:g}__ga{grad_accum}__r{rep}"


def run_cell(client: str, lr: float, grad_accum: int, rep: int, args,
             base_ppl: Optional[float]) -> Dict:
    """Run one training config; return its manifest plus the sweep bookkeeping."""
    tag = cell_tag(client, lr, grad_accum, rep)
    out_dir = Path(args.out_dir) / tag
    cmd = [
        sys.executable, "-m", "edge.train_client",
        "--client", client,
        "--budget-plan", args.budget_plan,
        "--lr", str(lr),
        "--grad-accum", str(grad_accum),
        "--micro-batch", str(args.micro_batch),
        "--seed", str(args.seed),
        "--out-dir", str(out_dir),
        "--log-every", "1000",          # the sweep table is the output, not the curve
        "--no-save",
    ]
    if base_ppl is not None:
        cmd += ["--base-ppl", str(base_ppl)]

    print(f"  -> {tag} ", end="", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    row = {"client": client, "lr": lr, "grad_accum": grad_accum, "rep": rep,
           "tag": tag, "wall_seconds": elapsed, "ok": proc.returncode == 0}

    if proc.returncode != 0:
        # The divergence/NaN guard raises rather than saving a poisoned adapter,
        # so a non-zero exit here is a RESULT, not an infrastructure failure.
        tail = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        row["error"] = tail[-1][:200] if tail else f"exit {proc.returncode}"
        print(f"FAILED ({elapsed}s) — {row['error'][:80]}")
        return row

    manifest_path = next(Path(out_dir).rglob("manifest.json"), None)
    if manifest_path is None:
        row["ok"] = False
        row["error"] = "no manifest.json produced"
        print(f"FAILED ({elapsed}s) — no manifest")
        return row

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    res = m["results"]
    row.update({
        "first_window": res["first_window"],
        "last_window": res["last_window"],
        "loss_slope": res["loss_slope"],
        "loss_decreased": res["loss_decreased"],
        "base_ppl": (res["base_held_out"] or {}).get("perplexity"),
        "final_ppl": (res["final_held_out"] or {}).get("perplexity"),
        "ppl_delta": res["held_out_ppl_delta"],
        "stable": res["stable"],
        "optimizer_steps": res["optimizer_steps"],
        "chunk_budget": m["optimization"]["chunk_budget"],
        "epochs": m["optimization"]["epochs"],
        "peak_vram_gb": m["hardware"]["peak_vram_gb"],
        "max_grad_norm": res.get("max_grad_norm"),
    })
    print(f"ok ({elapsed}s) loss {row['first_window']:.3f}->{row['last_window']:.3f} "
          f"slope {row['loss_slope']:+.5f} ppl_delta {row['ppl_delta']}")
    return row


def pick_winner(rows: List[Dict]) -> Optional[Dict]:
    """Best config = stable, then largest perplexity improvement.

    Train loss is deliberately NOT the tiebreaker. The lowest train loss in a
    sweep like this is usually the most overfit cell, which is the exact
    outcome E3.2 exists to reject.
    """
    ok = [r for r in rows if r.get("ok") and r.get("stable")]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["ppl_delta"] if r["ppl_delta"] is not None else 0.0))


def repeatability(rows: List[Dict]) -> List[Dict]:
    """Group repeats of the same (client, lr, grad_accum) and report the spread.

    'Repeatable' is E3.2's word. torch warns that the memory-efficient attention
    backward is non-deterministic, so bit-identical is not expected — what we
    need is a spread small enough that a real effect is distinguishable from
    run-to-run noise.
    """
    groups: Dict[tuple, List[Dict]] = {}
    for r in rows:
        if r.get("ok"):
            groups.setdefault((r["client"], r["lr"], r["grad_accum"]), []).append(r)
    out = []
    for (client, lr, ga), g in groups.items():
        if len(g) < 2:
            continue
        losses = [r["last_window"] for r in g]
        ppls = [r["final_ppl"] for r in g if r["final_ppl"] is not None]
        out.append({
            "client": client, "lr": lr, "grad_accum": ga, "n": len(g),
            "last_loss_spread": round(max(losses) - min(losses), 5),
            "final_ppl_spread": round(max(ppls) - min(ppls), 5) if len(ppls) > 1 else None,
            "identical": len(set(losses)) == 1,
        })
    return out


def print_table(rows: List[Dict]) -> None:
    hdr = (f"{'client':18s} {'lr':>8s} {'ga':>3s} {'r':>2s} {'steps':>6s} {'ep':>5s} "
           f"{'loss(win)':>14s} {'slope':>9s} {'ppl':>13s} {'dppl':>8s} {'gn':>6s} "
           f"{'stable':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        name = r["client"].split("/")[-1]
        if not r.get("ok"):
            print(f"{name:18s} {r['lr']:8.0e} {r['grad_accum']:3d} {r['rep']:2d} "
                  f"{'—':>6s} {'—':>5s} {'DIVERGED/NaN':>14s} {'—':>9s} {'—':>13s} "
                  f"{'—':>8s} {'—':>6s} {'no':>7s}")
            continue
        loss = f"{r['first_window']:.3f}->{r['last_window']:.3f}"
        slope = f"{r['loss_slope']:+.5f}" if r["loss_slope"] is not None else "—"
        ppl = (f"{r['base_ppl']:.2f}->{r['final_ppl']:.2f}"
               if r["base_ppl"] and r["final_ppl"] else "—")
        dppl = f"{r['ppl_delta']:+.3f}" if r["ppl_delta"] is not None else "—"
        gn = f"{r['max_grad_norm']:.2f}" if r["max_grad_norm"] is not None else "—"
        print(f"{name:18s} {r['lr']:8.0e} {r['grad_accum']:3d} {r['rep']:2d} "
              f"{r['optimizer_steps']:6d} {r['epochs']:5.2f} {loss:>14s} {slope:>9s} "
              f"{ppl:>13s} {dppl:>8s} {gn:>6s} {str(r['stable']):>7s}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LR/step sweep for stable convergence (E3.2)")
    ap.add_argument("--clients", nargs="+", required=True)
    ap.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    ap.add_argument("--grad-accums", nargs="+", type=int, default=list(DEFAULT_GRAD_ACCUMS))
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1, help="runs per cell, for the spread check")
    ap.add_argument("--budget-plan", default="budget_plan.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="sweep_out")
    ap.add_argument("--results", default="sweep_results.json")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    base_ppl_cache: Dict[str, Optional[float]] = {}
    t0 = time.time()

    grid = list(itertools.product(args.clients, args.lrs, args.grad_accums,
                                  range(1, args.repeat + 1)))
    print(f"{len(grid)} cells: {len(args.clients)} clients x {len(args.lrs)} lrs "
          f"x {len(args.grad_accums)} grad-accums x {args.repeat} reps\n")

    for client, lr, ga, rep in grid:
        row = run_cell(client, lr, ga, rep, args, base_ppl_cache.get(client))
        # First successful cell for a client establishes the base perplexity;
        # every later cell reuses it instead of re-measuring.
        if row.get("ok") and client not in base_ppl_cache and row.get("base_ppl"):
            base_ppl_cache[client] = row["base_ppl"]
        rows.append(row)

    print_table(rows)

    reps = repeatability(rows)
    if reps:
        print("\n----- repeatability -----")
        for r in reps:
            print(f"{r['client'].split('/')[-1]:22s} lr={r['lr']:.0e} ga={r['grad_accum']} "
                  f"n={r['n']}  last_loss spread {r['last_loss_spread']}  "
                  f"final_ppl spread {r['final_ppl_spread']}  identical={r['identical']}")

    winner = pick_winner(rows)
    print("\n----- recommendation -----")
    if winner is None:
        print("NO STABLE CONFIG in this grid. Widen the LR range downward or "
              "raise --grad-accums before picking one.")
    else:
        print(f"lr={winner['lr']:g}  grad_accum={winner['grad_accum']}  "
              f"micro_batch={args.micro_batch}  seed={args.seed}")
        print(f"  on {winner['client']}: ppl {winner['base_ppl']:.2f} -> "
              f"{winner['final_ppl']:.2f} ({winner['ppl_delta']:+.3f}), "
              f"train loss {winner['first_window']:.3f} -> {winner['last_window']:.3f}, "
              f"slope {winner['loss_slope']:+.5f}")

    payload = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "E3.2 lr/steps sweep for stable convergence",
        "grid": {"clients": args.clients, "lrs": args.lrs,
                 "grad_accums": args.grad_accums, "micro_batch": args.micro_batch,
                 "repeat": args.repeat},
        "seed": args.seed,
        "budget_plan": args.budget_plan,
        "rows": rows,
        "repeatability": reps,
        "recommended": ({"lr": winner["lr"], "grad_accum": winner["grad_accum"],
                         "micro_batch": args.micro_batch, "seed": args.seed}
                        if winner else None),
        "total_wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.results).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {Path(args.results).resolve()}  "
          f"(total {payload['total_wall_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
