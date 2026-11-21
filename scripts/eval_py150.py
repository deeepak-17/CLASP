"""Report an adapter's PY150 line-completion score (D5: reported only, never a gate).

    python scripts/eval_py150.py --data <path>/test.json [--adapter <peft adapter dir>]

Reads a local copy of CodeXGLUE's line-level Python completion test set
(``dataset/py150/line_completion/test.json`` in github.com/microsoft/CodeXGLUE,
built on the ETH py150 corpus). The script never downloads anything: fetch the
file once yourself. Commit no PY150 data.

Record format (CodeXGLUE ``Code-Code/CodeCompletion-line`` README): JSON Lines,
one object per line, ``{"input": "<s> tok tok <EOL> tok <EOL>", "gt": "tok tok"}``.
``input`` is the code before the line (pre-tokenized, tokens separated by
spaces, ``<EOL>`` ending each line, ``<s>`` at the start); ``gt`` is the line
to produce. The prompt is rebuilt by splitting on ``<EOL>``; the tokenized
spacing and the lost indentation are kept as they are, so scores are NOT
comparable with the guard's scores on real source files.

Licence: CodeXGLUE's code is MIT and its datasets are under the Computational
Use of Data Agreement (C-UDA, CC0 text); py150's repositories were filtered to
permissive licences (MIT/BSD/Apache). That permits computational research use;
check the licence before reporting results.

Scoring is the guard's: ``edge.completion_eval.evaluate_completion``, greedy,
via the model-loading helpers of ``scripts/evaluate_candidate.py``. A seeded
300-example subset (seed 0) is scored and written to
``services/evaluation/results/py150/<adapter sha256>.json`` (``base.json``
without an adapter). This never feeds ``/promote``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBSET_SIZE = 300
SUBSET_SEED = 0
OUT_DIR = REPO_ROOT / "services" / "evaluation" / "results" / "py150"


def parse_record(record: dict, index: int):
    """One CodeXGLUE record -> CompletionExample, or None when it has no target."""
    from evaluation.completion import CompletionExample

    gt = str(record.get("gt", "")).split("<EOL>")[0].strip()
    if not gt:
        return None
    text = str(record.get("input", "")).replace("<s>", " ")
    lines = [ln.strip() for ln in text.split("<EOL>")]
    if lines and not lines[-1]:
        lines.pop()  # the input ends with <EOL>
    prompt = "".join(ln + "\n" for ln in lines)
    return CompletionExample(file=f"py150/line_completion/test.json:{index}",
                             line_no=len(lines), prompt=prompt, target=gt)


def load_examples(path: Path, size: int = SUBSET_SIZE, seed: int = SUBSET_SEED) -> List:
    """The seeded subset of the local test file."""
    examples = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                ex = parse_record(json.loads(line), i)
                if ex is not None:
                    examples.append(ex)
    if len(examples) > size:
        examples = sorted(random.Random(seed).sample(examples, size), key=lambda e: e.file)
    return examples


def adapter_sha256(adapter_dir: Optional[Path]) -> str:
    if adapter_dir is None:
        return "base"
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = Path(adapter_dir) / name
        if f.exists():
            return hashlib.sha256(f.read_bytes()).hexdigest()
    raise SystemExit(f"no adapter_model.* under {adapter_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PY150 line completion, reported only")
    ap.add_argument("--data", type=Path, required=True, help="local CodeXGLUE py150 line_completion test.json")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="PEFT adapter directory (e.g. a composite from evaluate_candidate); default: base model")
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if not args.data.is_file():
        raise SystemExit(f"{args.data} not found: download the CodeXGLUE file yourself, this script never fetches")
    examples = load_examples(args.data)
    if not examples:
        raise SystemExit("no usable records in the data file")
    sha = adapter_sha256(args.adapter)

    from edge.completion_eval import evaluate_completion
    from edge.model_loader import load_model
    from edge.train_client import set_determinism
    from evaluation.completion import examples_fingerprint

    set_determinism(0)
    model, tokenizer, _ = load_model(args.profile)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    scored = evaluate_completion(model, tokenizer, examples, float("nan"))
    m = scored["in_project"]
    doc = {"benchmark": "PY150-line-completion (CodeXGLUE)", "reported_only": True,
           "source": f"scripts/eval_py150.py on {platform.node()}, "
                     f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "adapter_sha256": sha, "n_examples": len(examples), "seed": SUBSET_SEED,
           "examples_sha256": examples_fingerprint(examples),
           "edit_similarity": float(m["edit_similarity"]), "exact_match": float(m["exact_match"]),
           "wall_seconds": scored["wall_seconds"], "decoding": scored["decoding"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{sha}.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"PY150 edit similarity {doc['edit_similarity']:.4f}, exact match {doc['exact_match']:.4f} "
          f"({len(examples)} examples) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
