import random

import torch
from transformers import AutoTokenizer

from edge.model_loader import load_model, sanity_check
from edge.lora_init import attach_lora
from edge.dataset import toy_dataset

SEED = 0
EPOCHS = 10
LR = 2e-4

def encode(text: str, tokenizer: AutoTokenizer) -> dict:
    """One code string -> one training example.

    - add_special_tokens=True prepends BOS.
    - EOS is appended manually: HF does NOT add it for causal LM, and without
      it the model never learns where a function ends (it will ramble at
      generation time).
    - labels is a straight copy. HF's LlamaForCausalLM.forward does the shift:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
      Pre-shifting here would shift twice and train on garbage.
    """
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]

    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)

    input_ids = torch.tensor([ids], dtype=torch.long)  # [1, seq_len]

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }

def set_seed(seed: int) -> None:
    """Pin the RNGs so 'loss went down' is a repeatable claim, not luck."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, examples: list[dict], epochs: int, lr: float) -> tuple[float, float]:
    """Minimal LoRA training loop over the toy examples (batch size 1).

    Proves the backward pass reaches the LoRA params — NOT real convergence.
    On 8 tiny functions the model simply memorizes them, so loss drops fast.

    :return: (first_epoch_avg_loss, last_epoch_avg_loss)
    """
    device = model.device
    model.train()
    # Checkpointing (already enabled by prepare_model_for_kbit_training) needs
    # the KV cache off, or the two conflict and it silently no-ops.
    model.config.use_cache = False

    # Optimizer over TRAINABLE params only — the frozen 4-bit base has no grads,
    # so handing AdamW the full param set would waste state (and can error).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    print(f"optimizer over {sum(p.numel() for p in trainable):,} trainable params\n")

    first_avg = last_avg = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for ex in examples:
            batch = {k: v.to(device) for k, v in ex.items()}
            out = model(**batch)                # forward with labels -> HF loss
            loss = out.loss
            loss.backward()                     # grads flow into LoRA A/B
            optimizer.step()
            optimizer.zero_grad()               # or grads accumulate and lie
            epoch_loss += loss.item()

        avg = epoch_loss / len(examples)
        print(f"epoch {epoch:2d} | avg loss {avg:.4f}")
        if first_avg is None:
            first_avg = avg
        last_avg = avg

    return first_avg, last_avg


if __name__ == "__main__":
    set_seed(SEED)
    model, tokenizer, profile = load_model("dev")
    target_modules = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    model = attach_lora(model, r=16, target_modules=target_modules)

    examples = [encode(s, tokenizer) for s in toy_dataset]
    lengths = [e["input_ids"].shape[1] for e in examples]
    print(f"\nexamples      : {len(examples)}  | token lengths {lengths} (max {max(lengths)})")

    first_avg, last_avg = train(model, examples, epochs=EPOCHS, lr=LR)
    model.save_pretrained(save_directory="./cache")
    tokenizer.save_pretrained(save_directory="./cache")
    
    print("\n=== E2.2 acceptance ===")
    print(f"first-epoch avg loss : {first_avg:.4f}")
    print(f"last-epoch  avg loss : {last_avg:.4f}")
    print(f"loss decreased       : {last_avg < first_avg}")

    # Switching to inference mode
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    print("\n Fine-tuned model output:")
    prompt = "def fibonacci(n):\n    "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=profile.max_new_tokens, do_sample=False
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))
