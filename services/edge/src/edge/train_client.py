"""Real client LoRA training on a materialized partition (P1 Edge, W3 · E3.1/E3.2).

The toy loop proved gradients reach the LoRA params on 8 hand-written functions.
This trains the same adapter on a real client repo, packed into `seq_len` blocks
by `edge.chunking`, under D8's compute envelope.

What this adds over `edge.toy.toy_training_loop`:

  * real packed blocks instead of one-function examples (E3.1)
  * a step BUDGET per client rather than a flat epoch count, so a 60-block
    client and a 1900-block client are not trained to wildly different depths
  * warmup + cosine LR schedule, gradient clipping, gradient accumulation —
    the knobs E3.2 needs to make "loss went down" into "converged stably"
  * a NaN/divergence guard that aborts rather than writing a poisoned adapter
  * held-out perplexity before and after, which is what actually distinguishes
    learning from memorization: train loss falling while held-out perplexity
    rises IS the collapse the toy run showed, and no train-loss curve can see it
  * VRAM + wall-time instrumentation and a full run manifest (D9)

Acceptance for E3.1 is: loss decreases over a real run under D8 caps without
OOM. Acceptance for E3.2 is `stable: true` in the printed summary — no NaN, no
divergence, and held-out perplexity not worse than base.

This week the client trains on the frozen BASE alone. D3 wants it trained on
frozen (base + alpha*cluster), but P2's SVD aggregation does not produce a
cluster adapter until W5+, so that composition order is a known, temporary
deviation — recorded in the manifest as `d3_deviation`.

Usage:
    python -m edge.train_client --client web/client-flask --max-steps 76
    python -m edge.train_client --client scientific/client-numpy --budget-plan budget_plan.json
"""
import argparse
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers

from edge.chunking import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_SEQ_LEN,
    pack_client,
    split_summary,
)
from edge.lora_init import attach_lora
from edge.model_loader import load_model

# Pinned LoRA hyperparameters (D8 rank cap; target modules per lora_init).
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_DROPOUT = 0.0

# Training defaults. These are E3.2's subject matter — change them via the CLI
# and record what worked; do not edit them until a config is actually validated.
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_MICRO_BATCH = 1         # 4 GB VRAM: one 1024-token block at a time
DEFAULT_GRAD_ACCUM = 1          # raise for E3.2 stability; see step accounting below
DEFAULT_SEED = 0

# Abort thresholds for the divergence guard.
DIVERGENCE_FACTOR = 3.0         # loss > 3x the opening loss means it is running away
MAX_LOSS_SANITY = 20.0          # a code LM at seq 1024 should never sit up here


def set_determinism(seed: int) -> None:
    """Pin every RNG that could perturb the run, so a loss curve is repeatable.

    Matches edge.humaneval_baseline.set_determinism. `warn_only=True` because
    some bitsandbytes kernels have no deterministic implementation — we want the
    warning in the log, not a hard failure.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def collate(chunks: List[List[int]], pad_token_id: int, device) -> Dict[str, torch.Tensor]:
    """Stack blocks into a batch, padding short ones and masking the padding.

    Train blocks are all exactly seq_len so no padding happens there. The
    held-out split keeps its short final block (`drop_last=False`), and that one
    MUST be masked: pad == eos == 32014 for this tokenizer, so leaving pad
    positions in `labels` trains/scores the model on emitting EOS-as-padding.
    -100 is the ignore_index HF's loss uses.

    labels is a straight copy of input_ids — HF's LlamaForCausalLM.forward does
    the shift internally. Pre-shifting here would shift twice.
    """
    width = max(len(c) for c in chunks)
    input_ids, attention_mask, labels = [], [], []
    for c in chunks:
        pad = width - len(c)
        input_ids.append(c + [pad_token_id] * pad)
        attention_mask.append([1] * len(c) + [0] * pad)
        labels.append(c + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def build_schedule(total_chunks: int, budget: int, micro_batch: int, grad_accum: int,
                   rng: random.Random) -> List[List[int]]:
    """Turn a block budget into a list of micro-batches of block INDICES.

    The budget is expressed in blocks consumed, not optimizer steps, because
    that is the unit the epoch arithmetic is done in. Optimizer steps then fall
    out of the accumulation setting:

        blocks_per_step = micro_batch * grad_accum
        optimizer_steps = budget / blocks_per_step

    With the defaults (1 x 1) one block == one optimizer step, which keeps the
    numbers directly comparable to D8's "<= 200 steps/client/round".

    Blocks are drawn by reshuffling the full index list each pass, so a client
    whose budget exceeds its corpus revisits blocks in a different order every
    epoch, and one whose budget is a fraction of its corpus covers a different
    slice each round.
    """
    order: List[int] = []
    while len(order) < budget:
        epoch = list(range(total_chunks))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:budget]
    return [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]


@torch.no_grad()
def evaluate(model, chunks: List[List[int]], pad_token_id: int,
             micro_batch: int = 1) -> Optional[Dict[str, float]]:
    """Mean token-level loss and perplexity over a split.

    Batches are token-weighted, not batch-weighted: a 300-token tail block must
    not count the same as a full 1024-token one, or the number drifts with how
    the split happens to chunk.
    """
    if not chunks:
        return None
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(chunks), micro_batch):
        batch = collate(chunks[i:i + micro_batch], pad_token_id, model.device)
        out = model(**batch)
        # HF averages over non-ignored positions; recover the sum. The -1 is the
        # causal shift: n tokens yield n-1 predictions.
        n_tok = int((batch["labels"] != -100).sum().item()) - batch["labels"].shape[0]
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    if was_training:
        model.train()
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 4),
            "perplexity": round(math.exp(min(mean, 20.0)), 3),
            "n_tokens": total_tokens}


def convergence(step_losses: List[float], window_frac: float = 0.2) -> Dict:
    """Summarize a loss curve robustly enough to make a decision on.

    Comparing the FIRST and LAST single-step losses does not work here. At
    micro-batch 1 each step sees one 1024-token block, and blocks vary wildly in
    difficulty — a docstring-heavy block scores far below a dense-code one. The
    first observed run showed 0.669 -> 1.188 (an apparent regression) while
    held-out perplexity *improved*: the endpoints were noise, not trend.

    So compare the mean of the first 20% of steps against the mean of the last
    20%, and report an OLS slope over the whole curve as a second opinion. Both
    are needed: the windows say "did it end lower", the slope says "was it
    heading down the whole way" — a run that plunges then climbs back fails the
    second while passing the first.
    """
    n = len(step_losses)
    if n == 0:
        return {"first_window": None, "last_window": None, "loss_slope": None,
                "loss_decreased": False, "n_steps_recorded": 0}

    w = max(1, math.ceil(n * window_frac))
    first_window = sum(step_losses[:w]) / w
    last_window = sum(step_losses[-w:]) / w

    # OLS slope of loss against step index, in loss-units per step.
    mean_x = (n - 1) / 2
    mean_y = sum(step_losses) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    slope = (sum((i - mean_x) * (y - mean_y) for i, y in enumerate(step_losses)) / denom
             if denom else 0.0)

    return {
        "first_window": round(first_window, 4),
        "last_window": round(last_window, 4),
        "window_steps": w,
        "loss_slope": round(slope, 6),
        "loss_decreased": bool(last_window < first_window and slope < 0),
        "n_steps_recorded": n,
    }


def train(model, chunks: List[List[int]], schedule: List[List[int]], pad_token_id: int,
          lr: float, grad_accum: int, warmup_ratio: float, grad_clip: float,
          log_every: int) -> Dict:
    """LoRA training loop over packed blocks.

    Returns a history dict; raises RuntimeError on NaN/divergence rather than
    letting a poisoned adapter reach `save_pretrained`.
    """
    device = model.device
    model.train()
    # Checkpointing (enabled by prepare_model_for_kbit_training) needs the KV
    # cache off or the two conflict and checkpointing silently no-ops.
    model.config.use_cache = False

    # Optimizer over TRAINABLE params only — the frozen 4-bit base has no grads,
    # so handing AdamW the full param set wastes optimizer state and can error.
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    opt_steps = math.ceil(len(schedule) / grad_accum)
    warmup_steps = max(1, int(opt_steps * warmup_ratio))
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=opt_steps
    )
    print(f"optimizer over {n_trainable:,} trainable params")
    print(f"{len(schedule)} micro-batches -> {opt_steps} optimizer steps "
          f"({warmup_steps} warmup)\n")

    history: List[Dict] = []
    step_losses: List[float] = []       # exactly one entry per optimizer step
    micro_acc, micro_n = 0.0, 0         # micro-batches within the current step
    # Tracked over every step, not just logged ones: the pre-clip gradient norm
    # is the earliest warning that an LR is too hot, and it usually spikes
    # between log points rather than on them.
    max_grad_norm = 0.0
    step = 0
    t0 = time.time()

    # Divergence reference. A SINGLE opening batch is far too noisy to compare
    # against at micro-batch 1 — one easy block would arm a false tripwire — so
    # the reference is the mean over the first window, and until that many steps
    # have run only the absolute sanity cap applies.
    ref_window = max(1, math.ceil(opt_steps * 0.2))
    reference: Optional[float] = None

    for i, idxs in enumerate(schedule):
        batch = collate([chunks[j] for j in idxs], pad_token_id, device)
        loss = model(**batch).loss

        val = loss.item()
        if not math.isfinite(val):
            raise RuntimeError(
                f"non-finite loss ({val}) at micro-batch {i}. Lower the LR or "
                f"raise grad-accum; do NOT save this adapter."
            )
        if val > MAX_LOSS_SANITY or (reference and val > reference * DIVERGENCE_FACTOR):
            raise RuntimeError(
                f"loss diverged: {val:.4f} vs opening-window mean "
                f"{reference if reference else float('nan'):.4f} at micro-batch {i}. "
                f"Lower the LR."
            )

        # Scale so accumulated grads average rather than sum across the window.
        (loss / grad_accum).backward()
        micro_acc += val
        micro_n += 1

        if (i + 1) % grad_accum == 0 or i == len(schedule) - 1:
            # Clip on the LoRA params only — the base is frozen. This is the
            # main guard against a single pathological block spiking the update.
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            max_grad_norm = max(max_grad_norm, float(grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            step_losses.append(micro_acc / max(micro_n, 1))
            micro_acc, micro_n = 0.0, 0
            if reference is None and step >= ref_window:
                reference = sum(step_losses[:ref_window]) / ref_window

            if step % log_every == 0 or step == opt_steps:
                window = step_losses[-log_every:]
                avg = sum(window) / len(window)
                history.append({
                    "step": step,
                    "loss": round(avg, 4),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": round(float(grad_norm), 4),
                })
                print(f"step {step:4d}/{opt_steps} | loss {avg:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | gnorm {float(grad_norm):.3f}")

    return {
        "history": history,
        **convergence(step_losses),
        "optimizer_steps": step,
        "max_grad_norm": round(max_grad_norm, 4),
        "micro_batches": len(schedule),
        "n_trainable_params": n_trainable,
        "wall_seconds": round(time.time() - t0, 1),
    }


def resolve_budget(args, n_chunks: int, client_id: str) -> int:
    """Pick this client's block budget: explicit flag > plan file > D8 default."""
    if args.max_steps:
        return args.max_steps
    if args.budget_plan:
        plan = json.loads(Path(args.budget_plan).read_text(encoding="utf-8"))
        row = plan.get("clients", {}).get(client_id)
        if row is None:
            raise SystemExit(
                f"{client_id} not in {args.budget_plan}. Regenerate it with "
                f"`python -m edge.chunking --plan`."
            )
        return int(row["chunk_budget"])
    return min(200, n_chunks * 2)  # D8's flat cap, with the 2-epoch guard


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a client LoRA on a real partition")
    ap.add_argument("--client", required=True, help="'cluster/client-x' or an absolute client dir")
    ap.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    ap.add_argument("--profile", default="dev", help="model profile key (config.PROFILES)")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--max-steps", type=int, help="explicit block budget (overrides --budget-plan)")
    ap.add_argument("--budget-plan", help="budget_plan.json from `edge.chunking --plan`")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    ap.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out-dir", default="train_out", help="adapter + manifest destination")
    ap.add_argument("--skip-base-eval", action="store_true",
                    help="skip the pre-training held-out eval (faster, but no delta)")
    ap.add_argument("--base-ppl", type=float,
                    help="reuse a known base held-out perplexity instead of measuring it "
                         "(sweeps: the base model is identical across configs)")
    ap.add_argument("--no-save", action="store_true",
                    help="write the manifest but not the adapter (sweeps: ~25 MB/run)")
    args = ap.parse_args()

    client_id = args.client.replace("\\", "/").strip("/")
    client_dir = (Path(args.client) if Path(args.client).is_absolute()
                  else Path(args.corpus_root) / args.client)
    out_dir = Path(args.out_dir).resolve() / client_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(args.seed)
    model, tokenizer, profile = load_model(args.profile)

    train_split, held_split = pack_client(client_dir, tokenizer, seq_len=args.seq_len)
    budget = resolve_budget(args, train_split["n_chunks"], client_id)
    epochs = budget / train_split["n_chunks"]
    print(f"\n----- {client_dir.name} -----")
    print(f"train : {train_split['n_files']} files, {train_split['n_tokens']:,} tokens, "
          f"{train_split['n_chunks']} blocks")
    print(f"held  : {held_split['n_files']} files, {held_split['n_tokens']:,} tokens, "
          f"{held_split['n_chunks']} blocks")
    print(f"budget: {budget} blocks ({epochs:.2f} epochs)\n")

    # Base held-out perplexity, measured BEFORE the adapter is attached. This is
    # the floor every later number is compared against (D5).
    pad_id = tokenizer.pad_token_id
    if args.base_ppl is not None:
        # Supplied by the sweep runner: the base model is the same in every
        # config, so re-measuring it per run buys nothing but wall time.
        base_eval = {"loss": round(math.log(args.base_ppl), 4),
                     "perplexity": args.base_ppl, "reused": True}
    else:
        base_eval = None if args.skip_base_eval else evaluate(model, held_split["chunks"], pad_id)
    if base_eval:
        print(f"base held-out : loss {base_eval['loss']:.4f}  ppl {base_eval['perplexity']:.2f}\n")

    model = attach_lora(model, r=LORA_R, lora_alpha=LORA_ALPHA,
                        target_modules=LORA_TARGET_MODULES, dropout=LORA_DROPOUT)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    schedule = build_schedule(train_split["n_chunks"], budget, args.micro_batch,
                              args.grad_accum, random.Random(args.seed))
    result = train(model, train_split["chunks"], schedule, pad_id,
                   lr=args.lr, grad_accum=args.grad_accum,
                   warmup_ratio=args.warmup_ratio, grad_clip=args.grad_clip,
                   log_every=args.log_every)

    peak_vram_gb = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                    if torch.cuda.is_available() else None)

    final_eval = evaluate(model, held_split["chunks"], pad_id)

    # E3.2's real gate. Train loss falling is necessary but not sufficient: if
    # held-out perplexity ROSE, the adapter memorized the repo instead of
    # learning it, which is precisely the toy run's failure mode.
    loss_decreased = result["loss_decreased"]
    ppl_delta = (round(final_eval["perplexity"] - base_eval["perplexity"], 3)
                 if base_eval and final_eval else None)
    memorized = ppl_delta is not None and ppl_delta > 0
    stable = bool(loss_decreased and not memorized)

    if not args.no_save:
        model.save_pretrained(save_directory=str(out_dir / "adapter"))
        tokenizer.save_pretrained(save_directory=str(out_dir / "adapter"))

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "E3.1/E3.2 client LoRA on real partition",
        "client_id": client_id,
        "client_dir": str(client_dir),
        "profile": profile.name,
        "model_id": profile.model_id,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "lora": {
            "r": LORA_R, "lora_alpha": LORA_ALPHA,
            "target_modules": list(LORA_TARGET_MODULES), "lora_dropout": LORA_DROPOUT,
        },
        "data": {
            "seq_len": args.seq_len,
            "train": split_summary(train_split),
            "held_out": split_summary(held_split),
        },
        "optimization": {
            "lr": args.lr, "scheduler": "cosine_with_warmup",
            "warmup_ratio": args.warmup_ratio, "grad_clip": args.grad_clip,
            "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
            "chunk_budget": budget, "epochs": round(epochs, 3),
            "budget_plan": args.budget_plan,
        },
        "seed": args.seed,
        "results": {
            **result,
            "base_held_out": base_eval,
            "final_held_out": final_eval,
            "held_out_ppl_delta": ppl_delta,
            "loss_decreased": loss_decreased,
            "stable": stable,
        },
        "hardware": {
            "peak_vram_gb": peak_vram_gb,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        # D3 wants the client trained on frozen (base + alpha*cluster). P2's SVD
        # aggregation produces no cluster adapter until W5+, so this run is
        # base-only. Temporary and expected; recorded so no one reads a W3
        # number as a D3-compliant one.
        "d3_deviation": "trained on frozen base only; no cluster adapter exists yet (P2 W5+)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== E3.1/E3.2 acceptance ===")
    print(f"client            : {client_id}")
    print(f"blocks / budget   : {train_split['n_chunks']} / {budget} ({epochs:.2f} epochs)")
    print(f"optimizer steps   : {result['optimizer_steps']}")
    print(f"loss window mean  : {result['first_window']:.4f} -> {result['last_window']:.4f} "
          f"(first/last {result['window_steps']} steps)")
    print(f"loss slope        : {result['loss_slope']:+.6f} / step")
    print(f"max grad norm     : {result['max_grad_norm']:.3f}")
    print(f"loss decreased    : {loss_decreased}")
    if base_eval and final_eval:
        print(f"held-out ppl      : {base_eval['perplexity']:.2f} -> "
              f"{final_eval['perplexity']:.2f}  (delta {ppl_delta:+.3f})")
        print(f"memorization flag : {memorized}")
    print(f"stable (E3.2)     : {stable}")
    print(f"peak VRAM         : {peak_vram_gb} GB")
    print(f"wall time         : {result['wall_seconds']} s "
          f"({result['wall_seconds'] / max(result['optimizer_steps'], 1):.2f} s/step)")
    print(f"written to        : {out_dir}")


if __name__ == "__main__":
    main()
