"""In-project completion evaluation on the edge (Integration Sprint · B5).

Generation half of D5's primary metric. The scoring half — example selection,
edit similarity, exact match — lives in ``evaluation.completion`` (P5's module,
pure Python, no torch) so it can be unit-tested in CPU-only CI; this module is
what needs a GPU and a model.

    held-out .py files
      -> deterministic next-line problems      (evaluation.completion)
      -> greedy completion, one line each      (here)
      -> edit similarity / exact match         (evaluation.completion)
      -> perplexity                            (edge.train_client.evaluate)
      -> contracts.InProjectMetrics            -> seam C2

Decoding is greedy (``do_sample=False``), so a re-run on the same adapter and
the same examples reproduces the same numbers exactly. Every completion is cut
at the first newline: this is NEXT-LINE completion, and a model that keeps
writing the rest of the function must not be scored on it.

Usage:
    python -m edge.completion_eval --client web/client-flask \\
        --adapter artifacts/round1/client-flask/adapter --max-examples 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from evaluation.completion import (
    CompletionExample,
    collect_examples,
    examples_fingerprint,
    in_project_dict,
    per_example_rows,
    score,
)

#: Tokens of file context kept in front of a target line. The prefix is
#: truncated from the LEFT so the lines nearest the target always survive.
DEFAULT_PROMPT_TOKENS = 512
#: A source line longer than this is an outlier, not a completion problem.
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_MAX_EXAMPLES = 60
DEFAULT_BATCH_SIZE = 4


def _encode_prompts(tokenizer, examples: Sequence[CompletionExample],
                    prompt_tokens: int) -> List[List[int]]:
    """Tokenize each prompt and keep its final ``prompt_tokens`` tokens."""
    out: List[List[int]] = []
    for ex in examples:
        ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        out.append(ids[-prompt_tokens:] if len(ids) > prompt_tokens else ids)
    return out


def _first_line(text: str) -> str:
    """The generated line, cut at the first newline."""
    return text.split("\n", 1)[0]


@torch.no_grad()
def generate_completions(model, tokenizer, examples: Sequence[CompletionExample], *,
                         prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
                         max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                         batch_size: int = DEFAULT_BATCH_SIZE,
                         ) -> List[str]:
    """One greedy next-line completion per example, on the ACTIVE adapter.

    Batched with left padding so the last real token of every prompt sits at
    the end of the window — right padding would make the model continue from
    pad tokens and produce garbage for every short prompt in the batch.
    """
    was_training = model.training
    model.eval()
    encoded = _encode_prompts(tokenizer, examples, prompt_tokens)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    device = model.device
    completions: List[str] = []

    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention[row, width - len(ids):] = 1
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
        for row in range(len(batch)):
            new_tokens = out[row, width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(_first_line(text))
    if was_training:
        model.train()
    return completions


def evaluate_completion(model, tokenizer, examples: Sequence[CompletionExample],
                        perplexity: float, **gen_kw) -> Dict:
    """Full in-project measurement for whatever adapter is currently active."""
    t0 = time.time()
    preds = generate_completions(model, tokenizer, examples, **gen_kw)
    scored = score(preds, examples)
    return {
        "in_project": in_project_dict(scored, perplexity),
        "examples_sha256": examples_fingerprint(examples),
        "rows": per_example_rows(preds, examples),
        "wall_seconds": round(time.time() - t0, 1),
        "decoding": {"strategy": "greedy", "do_sample": False,
                     "max_new_tokens": gen_kw.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                     "prompt_tokens": gen_kw.get("prompt_tokens", DEFAULT_PROMPT_TOKENS),
                     "truncated_at": "first newline"},
    }


def client_examples(client_dir: Path, max_examples: int = DEFAULT_MAX_EXAMPLES,
                    stride: int = 7) -> List[CompletionExample]:
    """Next-line problems from one client's ``held_out/`` split (D5's eval set).

    ``held_out/`` is a sibling of ``repo/`` and was never trained on — see
    ``edge.chunking.pack_client``, which relies on the same separation.
    """
    held = Path(client_dir) / "held_out"
    if not held.is_dir():
        return []
    return collect_examples(held, max_examples=max_examples, stride=stride)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    from edge.chunking import DEFAULT_CORPUS_ROOT, DEFAULT_SEQ_LEN, discover_clients, pack_client
    from edge.model_loader import load_model
    from edge.train_client import evaluate, set_determinism

    ap = argparse.ArgumentParser(description="In-project next-line completion eval (B5)")
    ap.add_argument("--client", required=True, help="e.g. web/client-flask")
    ap.add_argument("--adapter", help="PEFT adapter directory; omit for the base model")
    ap.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the full result JSON here")
    args = ap.parse_args()

    set_determinism(args.seed)
    clients = discover_clients(Path(args.corpus_root))
    if args.client not in clients:
        raise SystemExit(f"unknown client {args.client!r}; have {sorted(clients)}")
    client_dir = clients[args.client]

    examples = client_examples(client_dir, args.max_examples, args.stride)
    if not examples:
        raise SystemExit(f"{args.client}: no usable held-out completion examples")

    model, tokenizer, _profile = load_model(args.profile)
    model.config.use_cache = True
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))

    _train, held = pack_client(client_dir, tokenizer, seq_len=DEFAULT_SEQ_LEN)
    ppl = evaluate(model, held["chunks"], tokenizer.pad_token_id)
    result = evaluate_completion(
        model, tokenizer, examples, ppl["perplexity"],
        prompt_tokens=args.prompt_tokens, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size)
    result["client"] = args.client
    result["adapter"] = args.adapter
    result["n_held_out_files"] = held["n_files"]
    result["perplexity_source"] = "edge.train_client.evaluate over the held-out split"

    m = result["in_project"]
    print(f"{args.client}  ({len(examples)} examples from {held['n_files']} held-out files)")
    print(f"  edit_similarity : {m['edit_similarity']:.4f}")
    print(f"  exact_match     : {m['exact_match']:.4f}")
    print(f"  perplexity      : {m['perplexity']:.4f}")
    print(f"  wall            : {result['wall_seconds']}s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
