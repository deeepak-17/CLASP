"""Rank sweep pilot: what does LoRA rank cost at serving time? (P1 Edge, Phase III)

For each rank in the config (default r = 4, 8, 16, 32) this script, on one GPU:

  1. loads the NF4 base and attaches a fresh LoRA of that rank (q/k/v/o),
  2. trains it for a few steps on one real client (the training pilot) and
     records seconds per step and peak VRAM, which feed the GPU-hour estimate,
  3. switches to inference and measures, against the SAME loaded model:
       * TTFT with the adapter disabled ("base"; equal to serving a merged composite)
       * TTFT with the adapter active, unmerged ("lora")
       * decode throughput (tokens/s over a fixed number of new tokens)
       * adapter swap time and adapter size on disk.

Latency does not depend on what the adapter learned, only on its shape, so a
short pilot gives real latency numbers without the full 17 GPU-h sweep.

Measurement helpers are imported, not re-implemented:
``edge.ttft.time_ttft`` / ``time_adapter_swap`` and ``edge.train_client.train``.

Usage:
    python -m edge.rank_bench --config experiments/rank-sweep-latency/config.yaml
"""
import argparse
import csv
import gc
import json
import random
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from transformers import AutoTokenizer

from edge.chunking import pack_client
from edge.config import PROFILES, REPO_ROOT
from edge.lora_init import attach_lora
from edge.model_loader import load_model
from edge.train_client import build_schedule, set_determinism, train
from edge.ttft import DEFAULT_PROMPTS, time_adapter_swap, time_ttft


def _gb(n_bytes: int) -> float:
    return round(n_bytes / 1024 ** 3, 3)


def decode_tokens_per_s(model, tokenizer, prompt: str, new_tokens: int, repeats: int) -> float:
    """Median decode throughput: new tokens / wall time of a fixed-length generate."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    rates = []
    for i in range(repeats + 1):                      # first call is warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                           do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        if i > 0:
            rates.append(new_tokens / (time.perf_counter() - t0))
    return round(statistics.median(rates), 2)


def ttft_median(model, tokenizer, prompts: List[str], repeats: int, warmup: int) -> Dict:
    """Median TTFT per prompt length, plus the mean of those medians as one headline."""
    rows = [time_ttft(model, tokenizer, p, repeats, warmup) for p in prompts]
    return {
        "per_prompt": rows,
        "headline_ms": round(statistics.mean(r["median_ms"] for r in rows), 2),
        "worst_p95_ms": max(r["p95_ms"] for r in rows),
    }


def bench_one_rank(rank: int, cfg: Dict, train_chunks: List[List[int]], out_tmp: Path) -> Dict:
    set_determinism(cfg["seed"])
    model, tokenizer, _ = load_model(cfg["profile"])
    model = attach_lora(model, r=rank, lora_alpha=rank,          # alpha = r -> scaling 1.0 at every rank
                        target_modules=tuple(cfg["target_modules"]), dropout=0.0)

    # --- training pilot ---
    torch.cuda.reset_peak_memory_stats()
    steps = cfg["pilot_steps"]
    schedule = build_schedule(len(train_chunks), steps, 1, 1, random.Random(cfg["seed"]))
    hist = train(model, train_chunks, schedule, tokenizer.pad_token_id, lr=cfg["lr"],
                 grad_accum=1, warmup_ratio=0.03, grad_clip=1.0, log_every=steps)
    train_peak = torch.cuda.max_memory_allocated()

    # --- serving ---
    model.eval()
    model.config.use_cache = True
    adapter_dir = out_tmp / f"r{rank}"
    model.save_pretrained(str(adapter_dir))
    size_mb = (adapter_dir / "adapter_model.safetensors").stat().st_size / 1024 ** 2

    prompts, reps, warm = DEFAULT_PROMPTS, cfg["ttft_repeats"], cfg["ttft_warmup"]
    with model.disable_adapter():
        base = ttft_median(model, tokenizer, prompts, reps, warm)
        base_tps = decode_tokens_per_s(model, tokenizer, prompts[1], cfg["decode_tokens"], cfg["decode_repeats"])
    torch.cuda.reset_peak_memory_stats()
    lora = ttft_median(model, tokenizer, prompts, reps, warm)
    lora_tps = decode_tokens_per_s(model, tokenizer, prompts[1], cfg["decode_tokens"], cfg["decode_repeats"])
    serve_peak = torch.cuda.max_memory_allocated()
    swap = time_adapter_swap(model, adapter_dir, "swap", cfg["swap_repeats"])

    row = {
        "rank": rank,
        "trainable_params": hist["n_trainable_params"],
        "adapter_mb": round(size_mb, 2),
        "pilot_steps": hist["optimizer_steps"],
        "s_per_step": round(hist["wall_seconds"] / hist["optimizer_steps"], 3),
        "loss_first_window": hist["first_window"],
        "loss_last_window": hist["last_window"],
        "train_peak_vram_gb": _gb(train_peak),
        "serve_peak_vram_gb": _gb(serve_peak),
        "ttft_base_ms": base["headline_ms"],
        "ttft_lora_ms": lora["headline_ms"],
        "ttft_overhead_ms": round(lora["headline_ms"] - base["headline_ms"], 2),
        "ttft_lora_worst_p95_ms": lora["worst_p95_ms"],
        "decode_base_tok_s": base_tps,
        "decode_lora_tok_s": lora_tps,
        "swap_median_s": swap["median_s"],
        "swap_p95_s": swap["p95_s"],
    }
    detail = {"summary": row, "ttft_base": base, "ttft_lora": lora, "swap": swap,
              "train_convergence": {k: v for k, v in hist.items() if k != "history"}}

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    bench = cfg["bench"] | {"seed": cfg["seed"], "profile": cfg["profile"],
                            "target_modules": cfg["target_modules"], "lr": cfg["lr"]}

    corpus = REPO_ROOT / cfg["corpus_root"]
    tokenizer = AutoTokenizer.from_pretrained(PROFILES[cfg["profile"]].model_id)
    train_split, _ = pack_client(corpus / cfg["pilot_client"], tokenizer, cfg["seq_len"])
    print(f"pilot client {cfg['pilot_client']}: {len(train_split['chunks'])} train blocks")

    results_dir = cfg_path.parent / "results"
    results_dir.mkdir(exist_ok=True)
    t_start = time.time()
    details = []
    with tempfile.TemporaryDirectory() as tmp:
        for rank in cfg["sweep"]["rank"]:
            print(f"\n===== rank {rank} =====")
            details.append(bench_one_rank(rank, bench, train_split["chunks"], Path(tmp)))
            print(json.dumps(details[-1]["summary"], indent=2))
    gpu_hours = (time.time() - t_start) / 3600

    rows = [d["summary"] for d in details]
    with open(results_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "gpu_hours_actual": round(gpu_hours, 4),
        "ranks": details,
    }
    (results_dir / "rank_bench.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {results_dir / 'summary.csv'} and rank_bench.json "
          f"({gpu_hours * 60:.1f} GPU-min)")


if __name__ == "__main__":
    main()
