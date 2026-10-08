"""Client-partition packing + per-client step budgeting (P1 Edge, W3 · E3.1).

Replaces the toy one-function-one-example dataset with real `seq_len` blocks
packed from a client's materialized partition.

Two jobs:

1. **Packing.** Concatenate a client's `.py` files into one EOS-separated token
   stream and slice it into fixed-size blocks. Fixed size means every training
   example is exactly full — no padding, so the `-100` label-masking trap never
   fires on the train split.

2. **Budgeting.** The six clients differ ~30x in size (requests ~60 blocks,
   scikit-learn ~1900). A flat "200 steps each" (D8) therefore means ~3.3 epochs
   for the smallest and ~0.11 for the largest: small clients memorize while
   large ones barely read their corpus once. `plan_budgets` redistributes a
   FIXED total budget so the epoch spread collapses without spending more
   GPU-hours.

Determinism matters more here than it looks. Files are concatenated and then
sliced at fixed offsets, so file ORDER decides every block boundary in the
dataset. `load_files` sorts, and `order_hash` pins the result into the run
manifest so a reordering becomes visible instead of silent.

Usage:
    python -m edge.chunking --client web/client-flask
    python -m edge.chunking --plan --policy sqrt --total-chunks 1200
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from edge.config import PROFILES
from transformers import AutoTokenizer

# Root of P5's materialized partitions (D1). Overridable on the CLI.
DEFAULT_CORPUS_ROOT = Path("C:/Users/admin/Desktop/git/CLASP/datasets/materialized")

DEFAULT_SEQ_LEN = 1024          # D8 cap
DEFAULT_TOTAL_CHUNKS = 1200     # 6 clients x 200 blocks, i.e. D8's total spend
DEFAULT_MAX_EPOCHS = 2.0        # ceiling for tiny clients — anti-memorization
BUDGET_POLICIES = ("uniform", "sqrt", "linear")


def load_files(path: Path):
    """Yield text and file path"""
    for f in sorted(path.rglob("*.py")):
        yield f.relative_to(path).as_posix(), f.read_text(encoding="utf-8")


def tokenize_repo(repo_dir: Path, tokenizer: AutoTokenizer) -> Dict:
    """Returns all the token ids generated"""
    all_tokens = []
    order = []
    for rel, code in load_files(repo_dir):
        tokens = tokenizer(code, add_special_tokens=False)["input_ids"]
        order.append(rel)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id) # 32014 for deepseek-coder-base-1.3b
    return {"tokens": all_tokens, "file_order": order}


def chunk_tokens(tokens: List[int], seq_len=1024, drop_last: bool = True) -> List[List[int]]:
    """Slice a token stream into fixed `seq_len` blocks.

    drop_last=True (training): every block is exactly full, so no padding and no
    label masking. Costs at most seq_len-1 tokens.

    drop_last=False (held-out eval): keeps the remainder as a short final block.
    The caller must pad it and mask the pad positions with -100 — `collate` in
    edge.train_client does. Needed because a small held-out split (flask holds
    2 files) can be shorter than one block entirely, and dropping the tail would
    silently produce an empty eval set.
    """
    chunks = []

    for i in range(0, len(tokens) - seq_len + 1, seq_len):
        chunk = tokens[i : i + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)
        else:
            break

    if not drop_last:
        rest = len(tokens) % seq_len
        if rest:
            chunks.append(tokens[-rest:])
    return chunks


def order_hash(file_order: List[str]) -> str:
    """SHA-256 over the exact file sequence that produced the token stream.

    Same hash => same blocks. A changed hash means the dataset moved even if the
    file COUNT is identical — exactly the failure an unsorted walk would hide.
    """
    h = hashlib.sha256()
    for name in file_order:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def pack_split(directory: Path, tokenizer, seq_len: int, drop_last: bool) -> Dict:
    """Tokenize + chunk one directory, carrying its provenance along."""
    packed = tokenize_repo(directory, tokenizer)
    chunks = chunk_tokens(packed["tokens"], seq_len=seq_len, drop_last=drop_last)
    return {
        "chunks": chunks,
        "n_files": len(packed["file_order"]),
        "n_tokens": len(packed["tokens"]),
        "n_chunks": len(chunks),
        "file_order": packed["file_order"],
        "order_sha256": order_hash(packed["file_order"]),
        "dropped_tail_tokens": len(packed["tokens"]) - sum(len(c) for c in chunks),
    }


def pack_client(client_dir: Path, tokenizer, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[Dict, Dict]:
    """Pack one client's train (`repo/`) and eval (`held_out/`) splits.

    The 10% held-out split is D5's in-project eval set and is never trained on:
    `repo/` and `held_out/` are siblings, so an rglob over `repo/` cannot reach
    it. Verified zero leakage by path and by content hash at materialization.
    """
    repo_dir = client_dir / "repo"
    held_dir = client_dir / "held_out"
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"no repo/ under {client_dir}")

    train = pack_split(repo_dir, tokenizer, seq_len, drop_last=True)
    if not train["chunks"]:
        raise ValueError(
            f"{client_dir.name}: {train['n_tokens']} tokens is less than one "
            f"{seq_len}-token block — nothing to train on."
        )
    held = (pack_split(held_dir, tokenizer, seq_len, drop_last=False) if held_dir.is_dir()
            else {"chunks": [], "n_files": 0, "n_tokens": 0, "n_chunks": 0,
                  "file_order": [], "order_sha256": order_hash([]),
                  "dropped_tail_tokens": 0})
    return train, held


def split_summary(split: Dict) -> Dict:
    """Manifest-safe view of a packed split: provenance minus the token payload."""
    return {k: v for k, v in split.items() if k != "chunks"}


def discover_clients(corpus_root: Path) -> Dict[str, Path]:
    """Map 'cluster/client-x' -> path for every materialized client.

    `_extra/` is skipped: it holds candidate repos beyond D1's locked six.
    """
    found = {}
    for cluster_dir in sorted(p for p in corpus_root.iterdir() if p.is_dir()):
        if cluster_dir.name.startswith("_"):
            continue
        for client_dir in sorted(p for p in cluster_dir.iterdir() if p.is_dir()):
            if client_dir.name.startswith("_"):
                continue
            if (client_dir / "repo").is_dir():
                found[f"{cluster_dir.name}/{client_dir.name}"] = client_dir
    return found


def plan_budgets(chunk_counts: Dict[str, int],
                 total_chunks: int = DEFAULT_TOTAL_CHUNKS,
                 policy: str = "sqrt",
                 max_epochs: float = DEFAULT_MAX_EPOCHS,
                 min_chunks: int = 20) -> Dict[str, Dict]:
    """Split a fixed total block budget across clients.

    Canonical FedAvg gives every client E local EPOCHS, so larger clients
    naturally take more steps; D8's flat per-client step cap is the deviation,
    and at a 30x size spread it is the one that hurts. This keeps D8's total
    compute constant and only redistributes it.

    Policies:
      uniform - D8 as written; the 30x epoch spread stays.
      sqrt    - weight by sqrt(n). Compresses 30x to ~5.6x for the same spend.
                Standard imbalance compromise: acknowledges size without letting
                one client dominate. Default.
      linear  - weight by n, i.e. equal epochs. Faithful FedAvg, but the three
                scientific clients then eat ~90% of the round.

    `max_epochs` caps tiny clients regardless of policy — the memorization guard
    E3.2 is looking for. `min_chunks` stops a client being starved into a
    meaningless update that would still carry aggregation weight in D2.

    NOTE: this is the LOCAL WORK knob only. D2's aggregation weighting stays
    proportional to sample count. Scaling both would square the size bias.
    """
    if policy not in BUDGET_POLICIES:
        raise ValueError(f"policy must be one of {BUDGET_POLICIES}, got {policy!r}")

    if policy == "uniform":
        weights = {c: 1.0 for c in chunk_counts}
    elif policy == "sqrt":
        weights = {c: math.sqrt(n) for c, n in chunk_counts.items()}
    else:
        weights = {c: float(n) for c, n in chunk_counts.items()}

    total_w = sum(weights.values()) or 1.0
    plan = {}
    for client, n in chunk_counts.items():
        raw = total_chunks * weights[client] / total_w
        ceiling = max(n * max_epochs, min_chunks)   # never exceed max_epochs...
        alloc = int(round(min(raw, ceiling)))
        alloc = max(alloc, min(min_chunks, n))      # ...but never starve either
        plan[client] = {
            "n_chunks": n,
            "chunk_budget": alloc,
            "epochs": round(alloc / n, 3) if n else 0.0,
        }
    return plan


def _cmd_pack(args, tokenizer) -> None:
    client_dir = (Path(args.client) if Path(args.client).is_absolute()
                  else Path(args.corpus_root) / args.client)
    train, held = pack_client(client_dir, tokenizer, seq_len=args.seq_len)
    print(f"-----{client_dir.name}-----")
    print(f"train : {train['n_files']:4d} files  {train['n_tokens']:9,d} tokens  "
          f"{train['n_chunks']:5d} blocks  (dropped tail {train['dropped_tail_tokens']})")
    print(f"held  : {held['n_files']:4d} files  {held['n_tokens']:9,d} tokens  "
          f"{held['n_chunks']:5d} blocks")
    print(f"order_sha256(train) : {train['order_sha256'][:16]}...")


def _cmd_plan(args, tokenizer) -> None:
    clients = discover_clients(Path(args.corpus_root))
    if not clients:
        raise SystemExit(f"no clients found under {args.corpus_root}")

    counts = {}
    for name, path in clients.items():
        train, _ = pack_client(path, tokenizer, seq_len=args.seq_len)
        counts[name] = train["n_chunks"]
        print(f"  counted {name}: {counts[name]} blocks")

    plan = plan_budgets(counts, total_chunks=args.total_chunks,
                        policy=args.policy, max_epochs=args.max_epochs)

    print(f"\n-----budget plan (policy={args.policy}, total={args.total_chunks} blocks)-----")
    print(f"{'client':32s} {'blocks':>8s} {'budget':>8s} {'epochs':>8s}")
    for name, row in sorted(plan.items()):
        print(f"{name:32s} {row['n_chunks']:8d} {row['chunk_budget']:8d} {row['epochs']:8.2f}")
    spent = sum(r["chunk_budget"] for r in plan.values())
    eps = [r["epochs"] for r in plan.values() if r["epochs"]]
    print(f"{'TOTAL':32s} {sum(counts.values()):8d} {spent:8d}")
    if eps:
        print(f"epoch spread: {max(eps) / min(eps):.1f}x   (uniform-200 is ~30x)")

    out = Path(args.out)
    out.write_text(json.dumps({
        "policy": args.policy,
        "total_chunks": args.total_chunks,
        "max_epochs": args.max_epochs,
        "seq_len": args.seq_len,
        "model_id": PROFILES[args.profile].model_id,
        "clients": plan,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack client partitions / plan step budgets")
    ap.add_argument("--client", help="'cluster/client-x' or an absolute client dir")
    ap.add_argument("--plan", action="store_true", help="compute the budget plan for all clients")
    ap.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    ap.add_argument("--profile", default="dev", help="model profile key (config.PROFILES)")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--policy", default="sqrt", choices=BUDGET_POLICIES)
    ap.add_argument("--total-chunks", type=int, default=DEFAULT_TOTAL_CHUNKS)
    ap.add_argument("--max-epochs", type=float, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--out", default="budget_plan.json")
    args = ap.parse_args()

    if not args.plan and not args.client:
        ap.error("pass --client <id> or --plan")

    # Tokenizer only — no model weights, so this runs fine without a GPU.
    tokenizer = AutoTokenizer.from_pretrained(PROFILES[args.profile].model_id)
    if args.plan:
        _cmd_plan(args, tokenizer)
    else:
        _cmd_pack(args, tokenizer)


if __name__ == "__main__":
    main()
