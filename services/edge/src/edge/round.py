"""Full E2E federated round — 6 clients, 2 clusters (W4/G2 + W5/G3).

Runs the whole chain against real artifacts, with no stubs left in it:

    trained client adapters          (6, from edge.train_client)
      -> SVD aggregation per cluster (edge.aggregate, D2)   => 2 REAL cluster adapters
      -> composite per client        (edge.merge, D6)        base + a*cluster + b*client
      -> in-project held-out eval    (D5 primary metric)
      -> alpha sweep, config-driven  (G2's "alpha/beta config-driven")
      -> promotion / rollback        (D5 two-sided rule)
      -> round manifest + timings    (D11: one round <= 30 min)

This is what closes G2: until now the cluster term in every composite was a
stand-in, because P2's aggregation was not available. Here the cluster adapter is
derived from three real client adapters, so `base + a*cluster + b*client` is a
genuine 3-layer composition and the personalization delta is measurable.

What is still NOT closed, stated plainly
----------------------------------------
* **D3 composition order.** The clients were trained on the frozen BASE, not on
  frozen (base + a*cluster) — the cluster adapter did not exist when they trained.
  Fixing it needs a second training pass and is the natural next round.
* **D5's regression guard.** The promotion rule is two-sided: in-project metric
  improves beyond the noise band AND HumanEval pass@1 drops <= 2 points. The
  HumanEval half cannot be scored on Windows (evalplus imports the Unix-only
  `resource` module), so the guard is reported `unavailable` and every decision
  here is PROVISIONAL. It is not authority to promote.
* **The noise band** is P5's W5 deliverable and does not exist yet. `--noise-band`
  defaults to 0.0, which makes "improved" mean "improved by any amount" — too
  permissive to be a real gate. Pass a measured value once there is one.

Eval cost note: a full held-out pass on a scientific client is ~140 blocks. The
alpha sweep therefore runs on a capped subsample (`--grid-max-blocks`) and only
the chosen alpha is re-measured on the full split. The cap is recorded per number
so no capped figure is mistaken for a full one.

Usage:
    python -m edge.round --adapters artifacts/round1 --alpha-grid 0 0.25 0.5 1.0 \\
        --out artifacts/round1_out
"""
import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import transformers
from peft import PeftModel

from edge.aggregate import aggregate_svd, exact_average_error
from edge.chunking import DEFAULT_CORPUS_ROOT, DEFAULT_SEQ_LEN, discover_clients, pack_client
from edge.merge import (
    CONTRACT_HYPERPARAMS,
    compose,
    load_adapter,
    merge_max_error,
    save_adapter,
    validate_compatibility,
)
from edge.model_loader import load_model
from edge.train_client import evaluate, set_determinism

NFR_ROUND_MINUTES = 30.0        # D11
HUMANEVAL_MAX_DROP = 2.0        # D5, absolute pass@1 points


def cluster_of(client_id: str) -> str:
    return client_id.split("/")[0]


def collect_adapters(adapters_root: Path, clients: Dict[str, Path]) -> Dict[str, Path]:
    """Map client id -> its trained adapter directory, skipping any not trained."""
    found = {}
    for cid, cdir in clients.items():
        path = adapters_root / cdir.name / "adapter"
        if (path / "adapter_config.json").exists():
            found[cid] = path
    return found


def eval_split(model, tokenizer, held_chunks: List[List[int]],
               max_blocks: Optional[int]) -> Dict:
    """Perplexity over a held-out split, on whatever adapter is currently active."""
    chunks = held_chunks[:max_blocks] if max_blocks else held_chunks
    return evaluate(model, chunks, tokenizer.pad_token_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full E2E federated round (G2/G3)")
    ap.add_argument("--adapters", required=True, help="root holding <client>/adapter dirs")
    ap.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--alpha-ref", type=float, default=0.5,
                    help="alpha always re-measured on the FULL split, so the cluster "
                         "layer's contribution is a real number even when the sweep "
                         "picks alpha=0 (where it would be zero by construction)")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16, help="cluster adapter rank after SVD")
    ap.add_argument("--grid-max-blocks", type=int, default=40,
                    help="cap held-out blocks during the alpha sweep (0 = no cap)")
    ap.add_argument("--noise-band", type=float, default=0.0,
                    help="perplexity improvement required to count (P5 W5; 0 = any)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+",
                    help="restrict to these clients or clusters (e.g. 'web' or "
                         "'web/client-flask'); used for smoke-testing the pipeline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_round = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_composites"
    scratch.mkdir(exist_ok=True)
    set_determinism(args.seed)

    clients = discover_clients(Path(args.corpus_root))
    trained = collect_adapters(Path(args.adapters), clients)
    if args.only:
        keep = set(args.only)
        trained = {c: p for c, p in trained.items()
                   if c in keep or cluster_of(c) in keep}
    if not trained:
        raise SystemExit(f"no trained adapters under {args.adapters}")
    print(f"round {args.round}: {len(trained)} trained clients across "
          f"{len({cluster_of(c) for c in trained})} clusters\n")

    # ---- phase 1: aggregate each cluster (D2) -------------------------------
    t0 = time.time()
    by_cluster: Dict[str, List[str]] = {}
    for cid in sorted(trained):
        by_cluster.setdefault(cluster_of(cid), []).append(cid)

    tokenizer_only_packs: Dict[str, Dict] = {}
    model, tokenizer, profile = load_model(args.profile)
    model.eval()
    model.config.use_cache = True

    print("packing held-out splits...")
    for cid in sorted(trained):
        train_split, held = pack_client(clients[cid], tokenizer, seq_len=args.seq_len)
        tokenizer_only_packs[cid] = {"held": held, "n_train_blocks": train_split["n_chunks"],
                                     "n_train_files": train_split["n_files"]}
        print(f"  {cid:34s} held-out {held['n_chunks']:4d} blocks "
              f"({held['n_files']} files)")

    print("\naggregating clusters (D2 SVD)...")
    cluster_dirs: Dict[str, Path] = {}
    cluster_stats: Dict[str, Dict] = {}
    for cluster, members in sorted(by_cluster.items()):
        adapters = [load_adapter(trained[m]) for m in members]
        validate_compatibility([c for _, c in adapters], members,
                               contract=CONTRACT_HYPERPARAMS)
        # FedAvg sample weighting: the client's DATASET size, not its step budget.
        weights = [tokenizer_only_packs[m]["n_train_blocks"] for m in members]
        sd, cfg, stats = aggregate_svd(adapters, weights, rank=args.rank)
        err = exact_average_error(adapters, weights, sd, cfg)
        cdir = out_dir / f"cluster-{cluster}"
        save_adapter(cdir, sd, cfg)
        cluster_dirs[cluster] = cdir
        cluster_stats[cluster] = {"members": members, "sample_weights": weights,
                                  **stats, "vs_exact_average": err}
        print(f"  {cluster:12s} <- {len(members)} clients  weights {weights}  "
              f"SVD loss mean {stats['reconstruction_error_mean']:.4f} / "
              f"max {stats['reconstruction_error_max']:.4f}")
    t_aggregate = round(time.time() - t0, 1)

    # ---- phase 2: base perplexity for every client, BEFORE any adapter -----
    # Deliberately done on the unwrapped model. Measuring the base by toggling
    # `disable_adapter()` on a PeftModel means depending on adapter-juggling
    # bookkeeping (and on there being a live adapter to disable at all) for the
    # single number every other number is compared against. The base does not
    # depend on any adapter, so measure it while none exist.
    grid_cap = args.grid_max_blocks or None
    base_ppl: Dict[str, Dict[str, Dict]] = {}
    print("\nbase perplexity (no adapter)...")
    for cid in sorted(trained):
        held = tokenizer_only_packs[cid]["held"]
        if not held["chunks"]:
            continue
        capped = eval_split(model, tokenizer, held["chunks"], grid_cap)
        full = (capped if (grid_cap is None or held["n_chunks"] <= grid_cap)
                else eval_split(model, tokenizer, held["chunks"], None))
        base_ppl[cid] = {"capped": capped, "full": full}
        print(f"  {cid:34s} capped {capped['perplexity']:7.4f}   "
              f"full {full['perplexity']:7.4f}")

    # ---- phase 3: alpha sweep per client, then full eval at the best --------
    results: Dict[str, Dict] = {}
    anchor_loaded = False

    for cid in sorted(trained):
        cluster = cluster_of(cid)
        held = tokenizer_only_packs[cid]["held"]
        if cid not in base_ppl:
            print(f"\n{cid}: no held-out blocks, skipping")
            continue
        print(f"\n----- {cid} -----")
        t_client = time.time()

        cluster_sd, cluster_cfg = load_adapter(cluster_dirs[cluster])
        client_sd, client_cfg = load_adapter(trained[cid])

        # Build every composite on the alpha grid up front (CPU work).
        composites: Dict[float, Path] = {}
        merge_checks: Dict[float, Dict] = {}
        for alpha in args.alpha_grid:
            parts = [(cluster_sd, cluster_cfg, alpha), (client_sd, client_cfg, args.beta)]
            sd, cfg = compose(parts)
            cdir = scratch / f"{cid.replace('/', '_')}_a{alpha:g}"
            save_adapter(cdir, sd, cfg)
            composites[alpha] = cdir
            merge_checks[alpha] = merge_max_error(parts, sd, cfg)

        # Load this client's whole grid at once and switch with set_adapter, which
        # is the cheap path (~0.5 s measured in E3.5). Rank-32 composites are ~50 MB
        # each, so a 4-point grid costs ~200 MB against 3 GB free — affordable, and
        # it means the model is never left with zero adapters.
        # PEFT adapter names become nn.Module keys, which forbid "." — so index
        # the grid rather than embedding the alpha value (a0.5 -> a1).
        safe_cid = cid.replace("/", "_").replace(".", "_")
        names = {}
        for idx, (alpha, cdir) in enumerate(composites.items()):
            name = f"{safe_cid}_a{idx}"
            if not anchor_loaded:
                model = PeftModel.from_pretrained(model, str(cdir), adapter_name=name)
                anchor_loaded = True
            else:
                model.load_adapter(str(cdir), adapter_name=name)
            names[alpha] = name

        grid_rows = []
        for alpha in args.alpha_grid:
            model.set_adapter(names[alpha])
            ev = eval_split(model, tokenizer, held["chunks"], grid_cap)
            delta = round(ev["perplexity"] - base_ppl[cid]["capped"]["perplexity"], 4)
            grid_rows.append({"alpha": alpha, "beta": args.beta,
                              "perplexity": ev["perplexity"], "ppl_delta": delta,
                              "merge_rel_err": merge_checks[alpha]["max_rel_err"]})
            print(f"  a={alpha:<5g} b={args.beta:<4g} ppl {ev['perplexity']:.4f}  "
                  f"delta {delta:+.4f}")

        best = min(grid_rows, key=lambda r: r["ppl_delta"])

        # Full-split re-measure at the winning alpha, plus the alpha=0 ablation
        # (client only) so the CLUSTER contribution can be separated from the
        # client contribution. Without that column the 3-layer claim is unfalsifiable.
        base_full = base_ppl[cid]["full"]
        model.set_adapter(names[best["alpha"]])
        comp_full = eval_split(model, tokenizer, held["chunks"], None)
        client_full = None
        if 0.0 in names:
            model.set_adapter(names[0.0])
            client_full = eval_split(model, tokenizer, held["chunks"], None)

        # Full-split measurement at a FIXED alpha. Without this, "cluster
        # contribution" is zero whenever the sweep picks alpha=0 — true but
        # vacuous. At a fixed nonzero alpha the number can come out positive
        # (cluster helps), negative (interferes), or ~0 (inert), so the 3-layer
        # claim is actually testable.
        ref_full = None
        if args.alpha_ref in names:
            model.set_adapter(names[args.alpha_ref])
            ref_full = eval_split(model, tokenizer, held["chunks"], None)

        personalization_delta = round(comp_full["perplexity"] - base_full["perplexity"], 4)
        cluster_contribution = (round(comp_full["perplexity"] - client_full["perplexity"], 4)
                                if client_full else None)
        # Negative => the cluster layer HELPS at alpha_ref; positive => it hurts.
        cluster_contribution_at_ref = (
            round(ref_full["perplexity"] - client_full["perplexity"], 4)
            if (ref_full and client_full) else None)

        improved = personalization_delta < -abs(args.noise_band)
        results[cid] = {
            "cluster": cluster,
            "n_train_files": tokenizer_only_packs[cid]["n_train_files"],
            "n_train_blocks": tokenizer_only_packs[cid]["n_train_blocks"],
            "n_held_out_blocks": held["n_chunks"],
            "grid_max_blocks": grid_cap,
            "alpha_grid": grid_rows,
            "best_alpha": best["alpha"],
            "beta": args.beta,
            "full_split": {
                "base_ppl": base_full["perplexity"],
                "client_only_ppl": client_full["perplexity"] if client_full else None,
                "composite_ppl": comp_full["perplexity"],
                "alpha_ref": args.alpha_ref,
                "alpha_ref_ppl": ref_full["perplexity"] if ref_full else None,
                "n_tokens": base_full["n_tokens"],
            },
            "personalization_delta_ppl": personalization_delta,
            "cluster_contribution_ppl": cluster_contribution,
            "cluster_contribution_at_alpha_ref": cluster_contribution_at_ref,
            "promotion": {
                "in_project_improved": improved,
                "noise_band": args.noise_band,
                "noise_band_source": "PLACEHOLDER — P5's measured band (W5) does not exist yet",
                "humaneval_guard": "unavailable (evalplus needs Unix `resource`; W1 debt)",
                "humaneval_max_drop_allowed": HUMANEVAL_MAX_DROP,
                "decision": ("PROVISIONAL_PROMOTE" if improved else "ROLLBACK"),
                "decision_is_authoritative": False,
            },
            "wall_seconds": round(time.time() - t_client, 1),
        }
        print(f"  -> best a={best['alpha']:g} | full-split base {base_full['perplexity']:.4f} "
              f"-> composite {comp_full['perplexity']:.4f} "
              f"(delta {personalization_delta:+.4f})")
        if cluster_contribution_at_ref is not None:
            print(f"     cluster layer at a={args.alpha_ref:g}: "
                  f"{cluster_contribution_at_ref:+.4f} ppl vs client-only "
                  f"({'helps' if cluster_contribution_at_ref < 0 else 'hurts/inert'})")

        # Free this client's grid. One adapter is always kept resident (the first
        # ever loaded) so the model never sits in a zero-adapter state.
        for alpha, name in names.items():
            if name in getattr(model, "peft_config", {}) and len(model.peft_config) > 1:
                model.delete_adapter(name)
        for d in composites.values():
            shutil.rmtree(d, ignore_errors=True)

    shutil.rmtree(scratch, ignore_errors=True)
    total_minutes = round((time.time() - t_round) / 60.0, 2)

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "W4/G2 3-layer composition + W5/G3 full E2E round",
        "round": args.round,
        "profile": profile.name,
        "model_id": profile.model_id,
        "seed": args.seed,
        "beta": args.beta,
        "alpha_grid": args.alpha_grid,
        "cluster_rank": args.rank,
        "clusters": cluster_stats,
        "clients": results,
        "timings": {
            "aggregate_seconds": t_aggregate,
            "total_minutes": total_minutes,
            "nfr_round_minutes": NFR_ROUND_MINUTES,
            "nfr_note": ("this round EXCLUDES client training, which ran separately; "
                         "see the round-cost breakdown in the report"),
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                             if torch.cuda.is_available() else None),
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "known_deviations": {
            "d3_composition_order": ("clients trained on frozen base, not on frozen "
                                     "(base + a*cluster) — no cluster adapter existed at "
                                     "training time; needs a second pass"),
            "d5_regression_guard": "HumanEval half unavailable on Windows; decisions provisional",
            "d5_noise_band": "placeholder 0.0; P5's measured band outstanding",
            "d1_web_cluster": "on disk = {flask, requests, werkzeug}; D1 says {django, flask, requests}",
        },
    }
    (out_dir / "round_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                 encoding="utf-8")

    print("\n=== ROUND SUMMARY ===")
    print(f"{'client':34s} {'a*':>4s} {'base':>8s} {'client':>8s} {'compos':>8s} "
          f"{'delta':>8s} {'clust':>7s}  decision")
    for cid, r in sorted(results.items()):
        f = r["full_split"]
        co = f"{f['client_only_ppl']:.3f}" if f["client_only_ppl"] else "—"
        cc = (f"{r['cluster_contribution_at_alpha_ref']:+.3f}"
              if r.get("cluster_contribution_at_alpha_ref") is not None else "—")
        print(f"{cid:34s} {r['best_alpha']:4g} {f['base_ppl']:8.3f} {co:>8s} "
              f"{f['composite_ppl']:8.3f} {r['personalization_delta_ppl']:+8.3f} {cc:>7s}  "
              f"{r['promotion']['decision']}")
    print(f"\nround wall time (aggregate + eval, excl. training): {total_minutes:.2f} min")
    print(f"written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
