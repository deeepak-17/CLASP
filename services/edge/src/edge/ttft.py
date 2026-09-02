"""TTFT and adapter-swap benchmark, composite vs base (P1 Edge, W3 · E3.5).

D11 turns the vague "personalization must not feel slow" NFR into two numbers
this script produces:

    TTFT overhead vs base  <= 200 ms
    adapter swap           <= 2 s

Definitions, because both are easy to measure wrongly
-----------------------------------------------------
**TTFT** = wall time from the generate call to the first generated token, i.e.
prefill plus one decode step. Measured as `generate(max_new_tokens=1)`. It is
NOT total generation time, and it is NOT a single forward pass.

**Adapter swap** = time to point an already-loaded base model at a different
composite. The base stays resident; only the adapter moves. Measuring a full
cold model load here would report ~10 s and fail an NFR that was never about
that.

Three measurement traps this avoids:

  * **CUDA is asynchronous.** Without `torch.cuda.synchronize()` on both sides
    of the timer you measure kernel-launch time, which is roughly zero and
    would make any NFR pass.
  * **The first call is a lie.** Kernel autotuning, cuBLAS workspace allocation
    and NF4 dequant caches all land on call one. Warmup iterations are discarded.
  * **The mean is the wrong statistic.** A single GC pause or a Windows
    scheduler hiccup drags it. Median is the headline; p95 is reported next to
    it so a long tail cannot hide behind a good median.

Usage:
    python -m edge.ttft --composite composite/ --repeats 20
    python -m edge.ttft --composite composite/ --swap-to other_composite/
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import transformers
from peft import PeftModel

from edge.model_loader import load_model

# D11's measurable NFRs.
NFR_TTFT_OVERHEAD_MS = 200.0
NFR_SWAP_SECONDS = 2.0

# Fixed prompts. Held constant across base and composite so the comparison is
# like-for-like; varied in length because prefill cost scales with prompt tokens
# and a single short prompt would flatter the composite.
DEFAULT_PROMPTS = [
    "def ",
    "def parse_config(path):\n    ",
    ("import numpy as np\n\n\n"
     "def normalize(matrix, axis=0):\n"
     "    \"\"\"Scale each row to unit norm.\"\"\"\n    "),
]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ttft(model, tokenizer, prompt: str, repeats: int, warmup: int) -> Dict:
    """Median/p95 TTFT in milliseconds for one prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    samples: List[float] = []

    for i in range(warmup + repeats):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=1,        # first token only — that is the metric
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:                  # discard warmup
            samples.append(elapsed_ms)

    ordered = sorted(samples)
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
        "stdev_ms": round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0,
        "n": len(samples),
    }


def bench_prompts(model, tokenizer, prompts: List[str], repeats: int, warmup: int,
                  label: str) -> List[Dict]:
    rows = []
    for prompt in prompts:
        row = time_ttft(model, tokenizer, prompt, repeats, warmup)
        row["prompt"] = prompt if len(prompt) <= 40 else prompt[:37] + "..."
        rows.append(row)
        print(f"  [{label:9s}] {row['prompt_tokens']:3d} tok -> "
              f"median {row['median_ms']:7.2f} ms   p95 {row['p95_ms']:7.2f} ms")
    return rows


def time_adapter_swap(model, adapter_path: Path, adapter_name: str,
                      repeats: int) -> Dict:
    """Time loading an additional adapter onto a resident PeftModel and
    activating it. The base model is never reloaded."""
    samples = []
    for i in range(repeats):
        name = f"{adapter_name}_{i}"
        _sync()
        t0 = time.perf_counter()
        model.load_adapter(str(adapter_path), adapter_name=name)
        model.set_adapter(name)
        _sync()
        samples.append(time.perf_counter() - t0)
        model.delete_adapter(name)      # keep VRAM flat across repeats
    ordered = sorted(samples)
    return {
        "median_s": round(statistics.median(samples), 4),
        "p95_s": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "max_s": round(max(samples), 4),
        "n": len(samples),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TTFT: composite vs base (E3.5)")
    ap.add_argument("--composite", required=True, help="composite adapter directory")
    ap.add_argument("--swap-to", help="a second adapter, to measure swap time")
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--swap-repeats", type=int, default=5)
    ap.add_argument("--out", default="ttft_results.json")
    args = ap.parse_args()

    composite_dir = Path(args.composite)
    merge_manifest = composite_dir / "merge_manifest.json"
    merge_meta = (json.loads(merge_manifest.read_text(encoding="utf-8"))
                  if merge_manifest.exists() else None)

    # --- base ---------------------------------------------------------------
    model, tokenizer, profile = load_model(args.profile)
    model.eval()
    model.config.use_cache = True
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n----- base -----")
    base_rows = bench_prompts(model, tokenizer, DEFAULT_PROMPTS,
                              args.repeats, args.warmup, "base")
    base_vram = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                 if torch.cuda.is_available() else None)

    # --- composite ----------------------------------------------------------
    # Same process and same resident base weights, so the delta isolates the
    # adapter. Reloading the model here would fold load-time variance into it.
    t0 = time.perf_counter()
    model = PeftModel.from_pretrained(model, str(composite_dir))
    _sync()
    first_load_s = round(time.perf_counter() - t0, 4)
    model.eval()

    print("\n----- composite -----")
    comp_rows = bench_prompts(model, tokenizer, DEFAULT_PROMPTS,
                              args.repeats, args.warmup, "composite")
    comp_vram = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                 if torch.cuda.is_available() else None)

    # --- swap ---------------------------------------------------------------
    swap = None
    if args.swap_to:
        print("\n----- adapter swap -----")
        swap = time_adapter_swap(model, Path(args.swap_to), "swapped", args.swap_repeats)
        print(f"  median {swap['median_s']:.3f} s   p95 {swap['p95_s']:.3f} s")

    # --- verdict ------------------------------------------------------------
    overheads = [c["median_ms"] - b["median_ms"] for b, c in zip(base_rows, comp_rows)]
    worst_overhead = max(overheads)
    ttft_pass = worst_overhead <= NFR_TTFT_OVERHEAD_MS
    swap_pass = swap is None or swap["median_s"] <= NFR_SWAP_SECONDS

    results = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "E3.5 TTFT composite vs base",
        "profile": profile.name,
        "model_id": profile.model_id,
        "composite": str(composite_dir),
        "composite_merge_manifest": merge_meta,
        "repeats": args.repeats,
        "warmup_discarded": args.warmup,
        "base": base_rows,
        "composite_ttft": comp_rows,
        "overhead_ms_per_prompt": [round(o, 2) for o in overheads],
        "worst_overhead_ms": round(worst_overhead, 2),
        "first_adapter_load_s": first_load_s,
        "adapter_swap": swap,
        "nfr": {
            "ttft_overhead_budget_ms": NFR_TTFT_OVERHEAD_MS,
            "ttft_pass": ttft_pass,
            "swap_budget_s": NFR_SWAP_SECONDS,
            "swap_pass": swap_pass,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "base_peak_vram_gb": base_vram,
            "composite_peak_vram_gb": comp_vram,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== E3.5 acceptance (D11) ===")
    for b, c, o in zip(base_rows, comp_rows, overheads):
        print(f"{b['prompt_tokens']:3d} tok | base {b['median_ms']:7.2f} ms | "
              f"composite {c['median_ms']:7.2f} ms | overhead {o:+7.2f} ms")
    print(f"worst TTFT overhead : {worst_overhead:+.2f} ms "
          f"(budget {NFR_TTFT_OVERHEAD_MS:.0f} ms) -> {'PASS' if ttft_pass else 'FAIL'}")
    if swap:
        print(f"adapter swap median : {swap['median_s']:.3f} s "
              f"(budget {NFR_SWAP_SECONDS:.0f} s) -> {'PASS' if swap_pass else 'FAIL'}")
    print(f"first adapter load  : {first_load_s:.3f} s (cold, not the swap metric)")
    print(f"peak VRAM           : base {base_vram} GB -> composite {comp_vram} GB")
    print(f"written to          : {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
