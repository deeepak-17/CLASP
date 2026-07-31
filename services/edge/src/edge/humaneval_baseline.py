"""HumanEval-subset baseline probe (P1 Edge, W1 · E1.3/E1.4).

Produces the *reproducible pass@1 anchor* for a base model profile. Everything
that could move the number is frozen and written to a manifest so the anchor can
be defended months later (D5's promotion rule is measured against this floor).

Pipeline:
  1. Load the base NF4 model via `edge.model_loader.load_model`.
  2. Take a FROZEN subset of HumanEval problems (first-N by sorted task id).
  3. Greedy-decode a completion per problem (deterministic — no sampling).
  4. Truncate each completion at the end of the target function.
  5. Score with the pinned `evalplus` CLI (execution stays inside its harness).
  6. Write `manifest.json` (exact config) and `anchor.json` (the number).

WARNING: scoring executes model-generated code. That runs inside evalplus, but
the sandboxing gap (P5 owns the docker design) is NOT closed here. Run on a
throwaway/dev box until P5's sandbox lands.

Usage:
    python -m edge.humaneval_baseline --profile dev --n 20
"""
import argparse
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from evalplus.data import get_human_eval_plus

from edge.model_loader import load_model

# Standard HumanEval stop sequences: a base model keeps generating past the
# target function, so cut at the first token that starts new top-level code.
STOP_SEQUENCES = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n@", "\nassert ")

# Frozen decoding defaults. Do not change without minting a NEW anchor.
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_SEED = 0


def set_determinism(seed: int) -> None:
    """Pin every RNG that could perturb generation. Greedy is already
    deterministic, but we set these so the manifest's seed is honest."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_subset(n: int) -> list[str]:
    """Return the first `n` HumanEval task ids in stable numeric order.

    Frozen by construction: same n -> same ids on any machine. The full id
    list is still written to the manifest so the choice is explicit, not
    implied by a magic number.
    """
    problems = get_human_eval_plus()
    ids = sorted(problems, key=lambda t: int(t.split("/")[1]))
    return ids[:n]


def truncate_completion(text: str) -> str:
    """Keep only the generated target function body; drop anything the base
    model rambled on to afterwards."""
    cut = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def generate_completions(model, tokenizer, problems: dict, task_ids: list[str],
                         max_new_tokens: int) -> list[dict]:
    """Greedy-decode one completion per task. Returns evalplus sample dicts:
    {"task_id", "solution"} where solution = prompt + truncated completion."""
    samples = []
    for i, tid in enumerate(task_ids, 1):
        prompt = problems[tid]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy -> deterministic
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens, not the echoed prompt.
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        solution = prompt + truncate_completion(gen)
        samples.append({"task_id": tid, "solution": solution})
        print(f"[{i}/{len(task_ids)}] {tid} generated")
    return samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_evalplus(samples_path: Path) -> Path:
    """Score with the pinned evalplus CLI. Execution of model-generated code
    happens here, inside evalplus's harness. Returns the results json path."""
    subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate",
         "--dataset", "humaneval", "--samples", str(samples_path)],
        check=True,
    )
    results = samples_path.with_name(samples_path.stem + "_eval_results.json")
    if not results.exists():
        raise FileNotFoundError(f"evalplus did not produce {results}")
    return results


def compute_pass_at_1(results_path: Path, task_ids: list[str]) -> dict:
    """Parse evalplus results and compute pass@1 over the subset ourselves —
    fraction of tasks whose single sample passes. Reported for both the base
    HumanEval tests and the EvalPlus-extended tests."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    eval_map = data.get("eval", data)
    base_pass = plus_pass = 0
    for tid in task_ids:
        records = eval_map.get(tid) or []
        rec = records[0] if isinstance(records, list) and records else {}
        if rec.get("base_status") == "pass":
            base_pass += 1
        if rec.get("plus_status") == "pass":
            plus_pass += 1
    n = len(task_ids)
    return {
        "n_problems": n,
        "base_pass_at_1": round(base_pass / n, 4),
        "plus_pass_at_1": round(plus_pass / n, 4),
        "base_solved": base_pass,
        "plus_solved": plus_pass,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="HumanEval-subset baseline pass@1 anchor")
    ap.add_argument("--profile", default="dev", help="model profile key (config.PROFILES)")
    ap.add_argument("--n", type=int, default=20, help="number of HumanEval problems (frozen subset)")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default="eval_out", help="where to write samples + anchor")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(args.seed)
    problems = get_human_eval_plus()
    task_ids = select_subset(args.n)

    model, tokenizer, profile = load_model(args.profile)
    samples = generate_completions(model, tokenizer, problems, task_ids,
                                   args.max_new_tokens)

    samples_path = out_dir / "samples.jsonl"
    write_jsonl(samples_path, samples)
    results_path = run_evalplus(samples_path)
    score = compute_pass_at_1(results_path, task_ids)

    # Manifest: every knob that could move the number. This is the anchor's
    # provenance — without it the pass@1 figure is undefendable.
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "model_id": profile.model_id,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "seed": args.seed,
        "subset": {"n": args.n, "task_ids": task_ids},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "anchor.json").write_text(json.dumps({**manifest, **score}, indent=2))

    print("\n=== BASELINE ANCHOR ===")
    print(f"profile      : {profile.name} ({profile.model_id})")
    print(f"subset       : first {args.n} HumanEval problems")
    print(f"base pass@1  : {score['base_pass_at_1']}  ({score['base_solved']}/{score['n_problems']})")
    print(f"plus pass@1  : {score['plus_pass_at_1']}  ({score['plus_solved']}/{score['n_problems']})")
    print(f"written to   : {out_dir}")


if __name__ == "__main__":
    main()
