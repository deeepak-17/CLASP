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

Composition order (D3)
----------------------
D3: ΔW_client is trained on the FROZEN merged (W_base + α·ΔW_cluster), not on
W_base alone. Pass ``--cluster-adapter`` (a cluster adapter P2's service
aggregated, pulled out of the registry by ``edge.registry_client``) and
``--alpha``, and the client trains with the cluster layer frozen underneath it:

    forward = W_base·x + α·B_c·A_c·x + B_l·A_l·x      only B_l, A_l get gradients

The cluster adapter is loaded as a second, frozen PEFT adapter with its scaling
multiplied by α, and both adapters are active at once — exact, and with no
dequantize/requantize of the NF4 base. Only the client adapter is saved. Without
``--cluster-adapter`` the client trains on the bare base, and the manifest says
so under ``d3_deviation``.

Differential privacy (D7)
-------------------------
``--dp`` trains with DP-SGD from P3's security library (``security.DPConfig`` +
``security.make_private``); the edge does not implement any of it. The ε spent
is recorded in the manifest as a ``contracts.PrivacySpec`` so it can ride the
upload envelope (``edge.wire.upload_payload(privacy=...)``) into the cluster and
the registry metadata. See ``train_dp`` for the constraints this imposes.

Usage:
    python -m edge.train_client --client web/client-flask --max-steps 76
    python -m edge.train_client --client scientific/client-numpy --budget-plan budget_plan.json
    python -m edge.train_client --client web/client-flask --budget-plan budget_plan.json \\
        --cluster-adapter pulled/cluster-web/v2 --alpha 0.5          # D3
    python -m edge.train_client --client web/client-requests --dp --seq-len 512   # D7
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
from edge.config import portable_path
from edge.lora_init import attach_lora
from edge.merge import CONTRACT_HYPERPARAMS, validate_compatibility
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

# A rising train loss is only disqualifying if it rises FAST. Healthy sub-epoch
# runs sit around +2e-4 (measured on client-numpy at lr 1e-3, a run whose
# held-out perplexity improved), so the tolerance sits an order of magnitude above.
SLOPE_DIVERGENCE_TOL = 2e-3

# D3: the PEFT adapter name the frozen cluster layer is loaded under. The
# client adapter stays "default", so save_pretrained writes it at the root of
# the output directory exactly as a base-only run does.
CLIENT_ADAPTER_NAME = "default"
CLUSTER_ADAPTER_NAME = "cluster"
DEFAULT_CLUSTER_ALPHA = 0.5     # the alpha the integration round composes with

# D7 defaults: the budget D7 fixes (epsilon <= 8, delta = 1e-5 per run) and a
# logical batch big enough for Poisson subsampling to amplify privacy.
DEFAULT_DP_EPSILON = 8.0
DEFAULT_DP_DELTA = 1e-5
DEFAULT_DP_MAX_GRAD_NORM = 1.0
DEFAULT_DP_BATCH_SIZE = 8


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


def stack_frozen_cluster(model, cluster_dir: Path, alpha: float) -> Dict:
    """D3: put α·ΔW_cluster underneath the client adapter, frozen.

    ``model`` is the PeftModel ``attach_lora`` returned (client adapter
    "default", freshly initialized). The cluster adapter is checked against
    the client's config and the frozen contract BEFORE it touches the model —
    a rank or target-module mismatch would otherwise train the client on top
    of a silently wrong base (``AdapterCompatibilityError``).

    Mechanism, and why it is exact:

      * ``load_adapter(..., is_trainable=False)`` adds the cluster as a second
        LoRA adapter; ``set_scale(name, α)`` multiplies its PEFT scaling by α,
        so its forward contribution is α·s_c·B_c·A_c — exactly α·ΔW_cluster.
      * ``set_adapter([client, cluster])`` makes both active. A LoRA layer sums
        the active adapters' outputs, so the forward pass is
        W_base·x + α·ΔW_cluster·x + ΔW_client·x.
      * ``set_adapter`` re-enables grads on every adapter it activates, so the
        cluster's parameters are frozen again explicitly afterwards, and the
        trainable set is asserted to be the client adapter and nothing else.

    Measured on the GPU (1.3B, cluster-web v2, α=0.5): with the client adapter
    freshly initialized (B = 0) the stacked logits equal those of a single
    pre-merged α·cluster composite with a max absolute difference of 0.0.
    """
    from peft.tuners.lora import LoraLayer

    cluster_dir = Path(cluster_dir)
    cluster_cfg = json.loads((cluster_dir / "adapter_config.json").read_text(encoding="utf-8"))
    client_cfg = model.peft_config[CLIENT_ADAPTER_NAME].to_dict()
    compat = validate_compatibility([cluster_cfg, client_cfg], ["cluster", "client"],
                                    contract=CONTRACT_HYPERPARAMS)

    model.load_adapter(str(cluster_dir), adapter_name=CLUSTER_ADAPTER_NAME, is_trainable=False)
    n_scaled = 0
    for module in model.modules():
        if isinstance(module, LoraLayer) and CLUSTER_ADAPTER_NAME in module.scaling:
            module.set_scale(CLUSTER_ADAPTER_NAME, alpha)
            n_scaled += 1
    model.base_model.set_adapter([CLIENT_ADAPTER_NAME, CLUSTER_ADAPTER_NAME])
    for name, param in model.named_parameters():
        if f".{CLUSTER_ADAPTER_NAME}." in name:
            param.requires_grad_(False)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    stray = [n for n in trainable if f".{CLIENT_ADAPTER_NAME}." not in n]
    if not trainable or stray:
        raise RuntimeError(f"D3 freeze failed: trainable parameters outside the client "
                           f"adapter: {stray[:5]}")

    provenance = None
    manifest = cluster_dir / "materialize_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        provenance = {k: m.get(k) for k in ("registry_url", "adapter", "version", "kind",
                                            "cluster_id", "aggregation", "round",
                                            "source_clients", "sha256_verified")}
    return {
        "composition_order": "D3: client trained on frozen (base + alpha*cluster)",
        "cluster_adapter": str(cluster_dir),
        "alpha": alpha,
        "cluster_rank": cluster_cfg["r"],
        "cluster_scaling_after_alpha": alpha * cluster_cfg["lora_alpha"] / cluster_cfg["r"],
        "lora_layers_scaled": n_scaled,
        "n_trainable_tensors": len(trainable),
        "compatibility": compat,
        "registry_provenance": provenance,
    }


class _Blocks(torch.utils.data.Dataset):
    """Packed train blocks as a map-style dataset — what Opacus samples from."""

    def __init__(self, chunks: List[List[int]]) -> None:
        lengths = {len(c) for c in chunks}
        if len(lengths) != 1:
            raise ValueError(f"DP training needs equal-length blocks, got lengths {sorted(lengths)}")
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.tensor(self.chunks[i], dtype=torch.long)


def _security_dp():
    """P3's DP-SGD API, or a precise statement of what is missing."""
    try:
        from security import DPConfig, EpsilonTracker, make_private
    except ImportError as exc:
        raise RuntimeError(
            "--dp needs P3's security library (security.DPConfig, security.make_private, "
            "security.EpsilonTracker). The installed `security` package does not provide "
            "them — integrate the security branch first.") from exc
    return DPConfig, EpsilonTracker, make_private


def train_dp(model, chunks: List[List[int]], lr: float, warmup_ratio: float,
             log_every: int, *, epochs: int, batch_size: int, target_epsilon: float,
             delta: float, max_grad_norm: float) -> Dict:
    """DP-SGD over the client's packed blocks, via ``security.make_private``.

    Everything privacy-related is P3's: ``DPConfig`` describes the run,
    ``make_private`` builds the Opacus engine and calibrates the noise to hit
    ``target_epsilon`` at ``delta`` over ``epochs``. The edge supplies the model,
    the optimizer, the data and the loop. Four constraints, all measured or
    read off the library rather than assumed:

    * ``grad_sample_mode="hooks"``. DPConfig defaults to ``"ghost"``, but in
      ghost mode Opacus 1.6's ``make_private`` returns FOUR objects (module,
      optimizer, criterion, loader) and ``security.make_private`` unpacks
      three, so the default crashes. Hooks mode returns three. With LoRA
      only (6.3 M params) per-sample gradients are cheap at physical batch 1.
    * Gradient checkpointing is switched OFF for DP runs. With it on, Opacus's
      per-sample hooks never populate ``grad_sample`` ("Per sample gradient is
      not initialized") — measured on this model. The memory it saved has to
      come from a shorter ``--seq-len`` instead.
    * ``security.make_private`` is used, not ``make_private_lora``: the latter
      sets ``requires_grad=True`` on every parameter whose name contains
      "lora_", which would UNFREEZE the D3 cluster adapter. Freezing is done
      by ``attach_lora`` / ``stack_frozen_cluster`` and asserted there.
    * The privacy unit is one packed ``seq_len`` block (record-level DP-SGD),
      which is what Opacus provides. D7 asks for client-level DP; that is a
      property of the federated protocol, not of one client's loop, and is
      recorded as such rather than claimed.

    ε is reported two ways: by Opacus's RDP accountant over the (σ, q, steps)
    this run actually took — the accountant ``security.make_private`` builds
    its engine with — and by P3's standalone ``security.EpsilonTracker``. They
    disagree (the tracker uses a looser bound, 1.3-3x higher on the settings
    checked); both are recorded and neither is hidden.
    """
    from opacus.accountants import RDPAccountant
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    DPConfig, EpsilonTracker, make_private = _security_dp()

    model.train()
    model.config.use_cache = False
    # Opacus's per-sample hooks do not survive checkpoint recomputation.
    model.base_model.model.gradient_checkpointing_disable()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    loader = torch.utils.data.DataLoader(_Blocks(chunks), batch_size=batch_size)
    cfg = DPConfig(target_epsilon=target_epsilon, target_delta=delta,
                   max_grad_norm=max_grad_norm, noise_multiplier=None, epochs=epochs,
                   batch_size=batch_size, physical_batch_size=1,
                   grad_sample_mode="hooks", accountant_type="rdp")
    dp_model, dp_opt, dp_loader = make_private(model, optimizer, loader, cfg)
    sigma = float(dp_opt.noise_multiplier)
    q = float(dp_loader.sample_rate)

    # Count real (noised) logical steps through Opacus's public step hook: it
    # fires only when a logical batch completes, never on the skipped
    # accumulation steps of BatchMemoryManager. Any hook already attached
    # (e.g. an accountant) is chained, not replaced.
    noised_steps = [0]
    prior_hook = dp_opt.step_hook

    def _count_step(opt) -> None:
        if prior_hook is not None:
            prior_hook(opt)
        noised_steps[0] += 1

    dp_opt.attach_step_hook(_count_step)

    planned_steps = epochs * len(dp_loader)
    warmup_steps = max(1, int(planned_steps * warmup_ratio))
    scheduler = transformers.get_cosine_schedule_with_warmup(
        dp_opt, num_warmup_steps=warmup_steps, num_training_steps=planned_steps)
    print(f"DP-SGD (security.make_private): sigma {sigma:.4f}, sample rate {q:.4f}, "
          f"clip {max_grad_norm}, {planned_steps} logical steps planned")

    history: List[Dict] = []
    step_losses: List[float] = []
    micro_losses: List[float] = []
    steps, empty_batches = 0, 0
    t0 = time.time()
    with BatchMemoryManager(data_loader=dp_loader, max_physical_batch_size=1,
                            optimizer=dp_opt) as memory_loader:
        for _ in range(epochs):
            for x in memory_loader:
                if x.shape[0] == 0:
                    # Poisson sampling can draw an empty logical batch. There is
                    # nothing to forward; the step is skipped and NOT accounted.
                    empty_batches += 1
                    continue
                x = x.to(model.device)
                loss = dp_model(input_ids=x, attention_mask=torch.ones_like(x), labels=x).loss
                val = loss.item()
                if not math.isfinite(val):
                    raise RuntimeError(f"non-finite loss ({val}) under DP-SGD at step {steps}; "
                                       f"do NOT save this adapter")
                loss.backward()
                micro_losses.append(val)
                before = noised_steps[0]
                dp_opt.step()
                if noised_steps[0] > before:              # a real logical step
                    scheduler.step()
                    steps += 1
                    step_losses.append(sum(micro_losses) / len(micro_losses))
                    micro_losses = []
                    if steps % log_every == 0 or steps == planned_steps:
                        window = step_losses[-log_every:]
                        history.append({"step": steps, "loss": round(sum(window) / len(window), 4),
                                        "lr": scheduler.get_last_lr()[0]})
                        print(f"dp step {steps:4d}/{planned_steps} | loss {history[-1]['loss']:.4f}")
                dp_opt.zero_grad(set_to_none=True)

    accountant = RDPAccountant()
    accountant.history = [(sigma, q, steps)] if steps else []
    eps_opacus = float(accountant.get_epsilon(delta)) if steps else 0.0
    tracker = EpsilonTracker(noise_multiplier=sigma, sample_rate=q, delta=delta)
    eps_tracker = float(tracker.step(steps)) if steps else 0.0

    privacy = {
        # contracts.PrivacySpec fields — what rides the upload envelope.
        "epsilon": round(eps_opacus, 6),
        "delta": delta,
        "noise_multiplier": round(sigma, 6),
        "max_grad_norm": max_grad_norm,
    }
    return {
        "history": history,
        **convergence(step_losses),
        "optimizer_steps": steps,
        "planned_steps": planned_steps,
        "empty_logical_batches": empty_batches,
        "max_grad_norm": None,       # per-sample clipping replaces global clipping
        "micro_batches": None,
        "n_trainable_params": n_trainable,
        "wall_seconds": round(time.time() - t0, 1),
        "privacy": privacy,
        "dp": {
            "library": "security.make_private (P3) over Opacus",
            "grad_sample_mode": "hooks",
            "accountant": "opacus RDPAccountant over the steps actually taken",
            "epsilon_opacus_rdp": round(eps_opacus, 6),
            "epsilon_security_tracker": round(eps_tracker, 6),
            "target_epsilon": target_epsilon,
            "within_d7_budget": eps_opacus <= DEFAULT_DP_EPSILON,
            "sample_rate": q,
            "logical_batch_size": batch_size,
            "epochs": epochs,
            "privacy_unit": "one packed block (record-level DP-SGD); client-level DP "
                            "is a federated-protocol property and is not claimed here",
            "gradient_checkpointing": False,
        },
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
    ap.add_argument("--cluster-adapter",
                    help="D3: cluster adapter directory (e.g. materialized from the registry "
                         "by edge.registry_client); the client trains on frozen base + "
                         "alpha*cluster. Omit to train on the bare base.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_CLUSTER_ALPHA,
                    help="D3 composition coefficient for the frozen cluster layer")
    ap.add_argument("--dp", action="store_true",
                    help="D7: train with DP-SGD from P3's security library")
    ap.add_argument("--dp-epsilon", type=float, default=DEFAULT_DP_EPSILON)
    ap.add_argument("--dp-delta", type=float, default=DEFAULT_DP_DELTA)
    ap.add_argument("--dp-max-grad-norm", type=float, default=DEFAULT_DP_MAX_GRAD_NORM)
    ap.add_argument("--dp-batch-size", type=int, default=DEFAULT_DP_BATCH_SIZE,
                    help="logical (Poisson-expected) batch size; physical batch is 1")
    ap.add_argument("--dp-epochs", type=int,
                    help="passes over the data under DP (default: the block budget "
                         "rounded to whole epochs, at least 1)")
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

    # D3: the frozen cluster layer goes in underneath the (still zero) client
    # adapter, and the starting point — base + alpha*cluster — is measured, so
    # the client's own contribution on top of it is a number, not an inference.
    d3 = None
    start_eval = None
    if args.cluster_adapter:
        d3 = stack_frozen_cluster(model, Path(args.cluster_adapter), args.alpha)
        print(f"D3: frozen cluster layer {args.cluster_adapter} at alpha={args.alpha} "
              f"({d3['lora_layers_scaled']} LoRA layers)")
        if not args.skip_base_eval:
            start_eval = evaluate(model, held_split["chunks"], pad_id)
            print(f"base + alpha*cluster held-out : loss {start_eval['loss']:.4f}  "
                  f"ppl {start_eval['perplexity']:.2f}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dp_epochs = None
    if args.dp:
        dp_epochs = args.dp_epochs or max(1, round(epochs))
        result = train_dp(model, train_split["chunks"], lr=args.lr,
                          warmup_ratio=args.warmup_ratio, log_every=args.log_every,
                          epochs=dp_epochs, batch_size=args.dp_batch_size,
                          target_epsilon=args.dp_epsilon, delta=args.dp_delta,
                          max_grad_norm=args.dp_max_grad_norm)
    else:
        schedule = build_schedule(train_split["n_chunks"], budget, args.micro_batch,
                                  args.grad_accum, random.Random(args.seed))
        result = train(model, train_split["chunks"], schedule, pad_id,
                       lr=args.lr, grad_accum=args.grad_accum,
                       warmup_ratio=args.warmup_ratio, grad_clip=args.grad_clip,
                       log_every=args.log_every)

    peak_vram_gb = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                    if torch.cuda.is_available() else None)

    final_eval = evaluate(model, held_split["chunks"], pad_id)
    # With D3 the final number is base + alpha*cluster + client; how much of the
    # gain the client adds over the frozen layer it trained on is this delta.
    ppl_delta_vs_start = (round(final_eval["perplexity"] - start_eval["perplexity"], 3)
                          if start_eval and final_eval else None)

    # E3.2's real gate. Train loss falling is necessary but not sufficient: if
    # held-out perplexity ROSE, the adapter memorized the repo instead of
    # learning it, which is precisely the toy run's failure mode.
    loss_decreased = result["loss_decreased"]
    ppl_delta = (round(final_eval["perplexity"] - base_eval["perplexity"], 3)
                 if base_eval and final_eval else None)
    memorized = ppl_delta is not None and ppl_delta > 0

    # Gradient clipping saturating is the mechanistic tell that an LR is too hot,
    # and it fires BEFORE the loss curve shows anything. Observed directly: at
    # lr=5e-3 on client-flask the training loss still fell with a negative slope
    # (so loss_decreased was True) and held-out perplexity moved -0.006, i.e.
    # nothing — while max grad norm hit 2.03 against a clip of 1.0. Judging that
    # run on its loss curve alone would have passed a config that learned
    # nothing generalizable. In the healthy band the same metric sits near 0.3.
    # Not defined under DP: per-sample clipping replaces the global clip there.
    clip_saturated = (result["max_grad_norm"] is not None
                      and result["max_grad_norm"] > 2.0 * args.grad_clip)

    # Train loss is not the right learning signal at the budgets D8 allows, and
    # two measured runs show why:
    #
    #   client-numpy  0.23 epochs: slope +0.00022 (rising) while held-out
    #                 perplexity improved 0.084. Every step saw a fresh block,
    #                 so "train loss" was out-of-sample loss tracking block
    #                 difficulty, not fitting.
    #   client-requests 1.135 epochs: slope -0.000196 (falling) but the endpoint
    #                 window means rose, because 89% of blocks are still seen
    #                 exactly once at 1.1 epochs. Held-out perplexity improved
    #                 0.194 — unambiguously a good run.
    #
    # So an epoch threshold is the wrong discriminator; at every budget we can
    # afford, train loss is dominated by which blocks landed where. D5 already
    # designates the correct primary metric: in-project held-out perplexity on
    # the client's own held-out files. Use it as the learning signal, and demote
    # the train-loss slope to what it is genuinely good for — catching runs that
    # are actively blowing up rather than merely noisy.
    diverging = (result["loss_slope"] or 0.0) > SLOPE_DIVERGENCE_TOL
    improved = ppl_delta is not None and ppl_delta < 0
    stable = bool(improved and not clip_saturated and not diverging)

    if not args.no_save:
        # Only the client adapter is this client's to publish; under D3 the
        # frozen cluster layer is the registry's, and is not re-saved here.
        model.save_pretrained(save_directory=str(out_dir / "adapter"),
                              selected_adapters=[CLIENT_ADAPTER_NAME])
        tokenizer.save_pretrained(save_directory=str(out_dir / "adapter"))

    # contracts.PrivacySpec: epsilon None means DP was off (the ablation).
    privacy = result.pop("privacy", None) or {
        "epsilon": None, "delta": args.dp_delta, "noise_multiplier": None,
        "max_grad_norm": None}

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "E3.1/E3.2 client LoRA on real partition",
        "client_id": client_id,
        "client_dir": portable_path(client_dir),
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
            "dp_sgd": args.dp, "dp_epochs": dp_epochs,
        },
        "seed": args.seed,
        "d3": d3,
        "privacy": privacy,
        "results": {
            **result,
            "base_held_out": base_eval,
            "cluster_start_held_out": start_eval,
            "final_held_out": final_eval,
            "held_out_ppl_delta": ppl_delta,
            "held_out_ppl_delta_vs_cluster_start": ppl_delta_vs_start,
            "loss_decreased": loss_decreased,
            "clip_saturated": clip_saturated,
            "grad_clip_threshold": args.grad_clip,
            "diverging": diverging,
            "improved_held_out": improved,
            "learned_signal": "held_out_ppl (D5 primary metric)",
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
        # Recorded so no one reads a base-only number as a D3-compliant one.
        "d3_deviation": (None if d3 else
                         "trained on frozen base only (no --cluster-adapter given)"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== E3.1/E3.2 acceptance ===")
    print(f"client            : {client_id}")
    print(f"composition       : "
          f"{'D3 frozen base + %g*cluster' % args.alpha if d3 else 'frozen base only'}")
    print(f"blocks / budget   : {train_split['n_chunks']} / {budget} ({epochs:.2f} epochs)")
    print(f"optimizer steps   : {result['optimizer_steps']}")
    if result["first_window"] is not None:
        print(f"loss window mean  : {result['first_window']:.4f} -> {result['last_window']:.4f} "
              f"(first/last {result['window_steps']} steps)")
        print(f"loss slope        : {result['loss_slope']:+.6f} / step")
    if result["max_grad_norm"] is not None:
        print(f"max grad norm     : {result['max_grad_norm']:.3f} "
              f"(clip {args.grad_clip}){'  <-- SATURATED, lr too hot' if clip_saturated else ''}")
    if args.dp:
        dp = result["dp"]
        print(f"DP-SGD epsilon    : {dp['epsilon_opacus_rdp']:.4f} (opacus RDP) / "
              f"{dp['epsilon_security_tracker']:.4f} (security.EpsilonTracker) at "
              f"delta {args.dp_delta:g}, sigma {privacy['noise_multiplier']}")
    print(f"loss decreased    : {loss_decreased}")
    if base_eval and final_eval:
        print(f"held-out ppl      : {base_eval['perplexity']:.2f} -> "
              f"{final_eval['perplexity']:.2f}  (delta {ppl_delta:+.3f})")
        if start_eval:
            print(f"vs base+a*cluster : {start_eval['perplexity']:.2f} -> "
                  f"{final_eval['perplexity']:.2f}  (delta {ppl_delta_vs_start:+.3f})")
        print(f"memorization flag : {memorized}")
    print(f"stable (E3.2)     : {stable}")
    print(f"peak VRAM         : {peak_vram_gb} GB")
    print(f"wall time         : {result['wall_seconds']} s "
          f"({result['wall_seconds'] / max(result['optimizer_steps'], 1):.2f} s/step)")
    print(f"written to        : {out_dir}")


if __name__ == "__main__":
    main()
