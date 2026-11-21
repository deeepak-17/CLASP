"""Evaluate one point of the alpha/beta composition sweep (Phase III, P4).

    Wnew = Wbase + alpha * dW_cluster + beta * dW_client

For one (alpha, beta) and one client this builds the composite adapter with
edge.merge.compose and scores it with edge.completion_eval (D5's primary
metric: greedy next-line completion on the client's held_out/ files, plus
perplexity). (0, 0) is the base model with no adapter: the sweep's baseline.

Run by `registry.runs`, once per sweep point:

    python -m registry.runs run experiments/alpha-beta-sweep/config.yaml -- \\
        python experiments/alpha-beta-sweep/eval_point.py \\
            --alpha {alpha} --beta {beta} --client web/client-requests \\
            --cluster-adapter <dir> --client-adapter <dir> --out-dir {outputs_dir}

Caveat for the paper (D3): the client adapter was trained on top of
base + alpha_train * cluster. Points with alpha != alpha_train evaluate it on a
base it was not trained against.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from edge.merge import compose, load_adapter, save_adapter


def prepare_adapter(alpha: float, beta: float, cluster_dir: Path, client_dir: Path,
                    out_dir: Path) -> Path | None:
    """The adapter this sweep point evaluates, or None for the bare base model."""
    if alpha == 0.0 and beta == 0.0:
        return None
    cluster_sd, cluster_cfg = load_adapter(Path(cluster_dir))
    client_sd, client_cfg = load_adapter(Path(client_dir))
    sd, cfg = compose([(cluster_sd, cluster_cfg, alpha), (client_sd, client_cfg, beta)])
    path = Path(out_dir) / "composite"
    save_adapter(path, sd, cfg)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score one (alpha, beta) composite")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--client", required=True, help="e.g. web/client-requests")
    ap.add_argument("--cluster-adapter", type=Path, required=True)
    ap.add_argument("--client-adapter", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=60)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adapter = prepare_adapter(args.alpha, args.beta, args.cluster_adapter,
                              args.client_adapter, args.out_dir)
    metrics = args.out_dir / "metrics.json"
    cmd = [sys.executable, "-m", "edge.completion_eval", "--client", args.client,
           "--max-examples", str(args.max_examples), "--out", str(metrics)]
    if adapter is not None:
        cmd += ["--adapter", str(adapter)]
    code = subprocess.run(cmd, check=False).returncode

    point = {"alpha": args.alpha, "beta": args.beta, "client": args.client,
             "adapter": "base (no adapter)" if adapter is None else "composite",
             "cluster_adapter": str(args.cluster_adapter),
             "client_adapter": str(args.client_adapter), "eval_exit_code": code}
    (args.out_dir / "point.json").write_text(json.dumps(point, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
