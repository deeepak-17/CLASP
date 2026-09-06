"""In-project next-line completion metric (Integration Sprint · B5, D5's primary signal).

The D5 promotion rule promotes on ``edit_similarity``. Until now the edge lane
measured perplexity and nothing else, so ``contracts.InProjectMetrics`` could
not be filled honestly and seam C2 had nothing real to carry. This module is
the missing metric.

**What it measures.** Next-line completion on a client's held-out files: take
the file up to some line, ask the model for the following line, compare against
what the file actually says. Three numbers come out, matching
``InProjectMetrics`` field for field:

    edit_similarity  1 - levenshtein(pred, gold) / max(|pred|, |gold|), averaged
                     over examples. 1.0 is a character-exact match. This is the
                     line-completion convention (RepoBench / CodeXGLUE use the
                     same Levenshtein ratio).
    exact_match      fraction of examples where the stripped prediction equals
                     the stripped gold line.
    perplexity       supplied by the caller from the existing token-level
                     evaluation (``edge.train_client.evaluate``) — not
                     recomputed here, so there is exactly one perplexity
                     implementation in the repo.

**Why stripped comparison.** The prompt ends at a newline, so leading
indentation is part of what the model must produce; but a model that emits the
right code with one space of drift is not wrong in a way this metric should
punish at full weight. Both fields are computed on ``line.strip()``, which is
the RepoBench convention. Indentation correctness is therefore NOT measured
here — stated rather than hidden.

Everything in this module is pure Python over strings: no torch, no model, no
network. That keeps the metric unit-testable in CPU-only CI and keeps the
generation half (which does need a model) in ``edge.completion_eval``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: A line must have at least this many non-whitespace characters to be a
#: target. Completing "" or ")" is not a measurement of anything.
MIN_TARGET_CHARS = 4
#: Lines of file context required before a target, so the model has something
#: to condition on.
MIN_PREFIX_LINES = 3


@dataclass(frozen=True)
class CompletionExample:
    """One next-line completion problem drawn from a held-out file."""

    file: str            # path relative to the held-out root
    line_no: int         # 0-based index of the target line within the file
    prompt: str          # the file text up to and including the newline before it
    target: str          # the ground-truth line, without its trailing newline


def levenshtein(a: str, b: str) -> int:
    """Edit distance with the usual unit costs, two rows of DP.

    Written out rather than pulled from `rapidfuzz`/`python-Levenshtein` so the
    metric has no dependency that CI would have to install and so the number is
    reproducible from this file alone.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def edit_similarity(pred: str, gold: str) -> float:
    """1 - normalized edit distance, in [0, 1]. Two empty strings score 1.0."""
    if not pred and not gold:
        return 1.0
    return 1.0 - levenshtein(pred, gold) / max(len(pred), len(gold))


def is_usable_target(line: str) -> bool:
    """Would completing this line measure anything?

    Blank lines, comment-only lines and one-token closers are excluded: a model
    scores near 1.0 on them regardless of whether it learned the project, which
    inflates edit similarity without saying anything about personalization.
    """
    stripped = line.strip()
    if len(stripped) < MIN_TARGET_CHARS:
        return False
    if stripped.startswith("#"):
        return False
    return True


def extract_examples(text: str, file_name: str, *, max_per_file: int,
                     stride: int = 1, min_prefix_lines: int = MIN_PREFIX_LINES,
                     ) -> List[CompletionExample]:
    """Deterministic next-line problems from one file.

    Candidates are walked in file order and taken every ``stride``-th usable
    line, so the same file always yields the same examples — no RNG anywhere in
    the selection path, which is what lets two runs be compared at all.
    """
    lines = text.splitlines()
    out: List[CompletionExample] = []
    seen = 0
    for idx in range(min_prefix_lines, len(lines)):
        if not is_usable_target(lines[idx]):
            continue
        if seen % stride == 0:
            prompt = "\n".join(lines[:idx]) + "\n"
            out.append(CompletionExample(file=file_name, line_no=idx,
                                         prompt=prompt, target=lines[idx]))
            if len(out) >= max_per_file:
                seen += 1
                break
        seen += 1
    return out


def collect_examples(held_out_dir: Path | str, *, max_examples: int,
                     max_per_file: Optional[int] = None, stride: int = 7,
                     pattern: str = "*.py") -> List[CompletionExample]:
    """Examples across a client's held-out split, files walked in sorted order.

    ``stride`` spreads the picks through each file instead of taking the first
    N lines, which would over-sample imports and module docstrings. Files are
    sorted (same rule ``edge.chunking.load_files`` uses) so the example set is a
    pure function of the directory contents.
    """
    held_out_dir = Path(held_out_dir)
    files = sorted(held_out_dir.rglob(pattern))
    if not files:
        return []
    per_file = max_per_file if max_per_file is not None else max(
        1, -(-max_examples // max(len(files), 1)) * 2)
    examples: List[CompletionExample] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        examples.extend(extract_examples(
            text, path.relative_to(held_out_dir).as_posix(),
            max_per_file=per_file, stride=stride))
        if len(examples) >= max_examples:
            break
    return examples[:max_examples]


def examples_fingerprint(examples: Sequence[CompletionExample]) -> str:
    """SHA-256 over (file, line_no) of the example set.

    Two evaluations are only comparable if they scored the SAME problems. The
    fingerprint goes in the manifest so a changed eval set becomes visible
    instead of silently shifting the numbers the promotion rule reads.
    """
    h = hashlib.sha256()
    for ex in examples:
        h.update(f"{ex.file}:{ex.line_no}\n".encode("utf-8"))
    return h.hexdigest()


def score(predictions: Iterable[str], examples: Sequence[CompletionExample],
          ) -> Dict[str, float]:
    """Aggregate edit similarity and exact match over scored examples."""
    sims: List[float] = []
    exact = 0
    n = 0
    for pred, ex in zip(predictions, examples):
        p, g = pred.strip(), ex.target.strip()
        sims.append(edit_similarity(p, g))
        exact += int(p == g)
        n += 1
    if n == 0:
        raise ValueError("no completion examples were scored")
    return {
        "edit_similarity": sum(sims) / n,
        "exact_match": exact / n,
        "n_examples": n,
    }


def per_example_rows(predictions: Sequence[str], examples: Sequence[CompletionExample],
                     ) -> List[Dict[str, object]]:
    """One row per example, for the results file — the evidence behind the mean."""
    rows: List[Dict[str, object]] = []
    for pred, ex in zip(predictions, examples):
        p, g = pred.strip(), ex.target.strip()
        rows.append({
            "file": ex.file,
            "line_no": ex.line_no,
            "gold": g,
            "pred": p,
            "edit_similarity": round(edit_similarity(p, g), 6),
            "exact_match": p == g,
        })
    return rows


def to_in_project_metrics(scored: Dict[str, float], perplexity: float):
    """Build ``contracts.InProjectMetrics`` from a scored run.

    Perplexity is passed in rather than computed: it comes from the existing
    token-level evaluation so the repo keeps one perplexity implementation.
    """
    from contracts import InProjectMetrics

    return InProjectMetrics(
        edit_similarity=float(scored["edit_similarity"]),
        exact_match=float(scored["exact_match"]),
        perplexity=float(perplexity),
        n_examples=int(scored["n_examples"]),
    )


def in_project_dict(scored: Dict[str, float], perplexity: float) -> Dict[str, float]:
    """The same thing as a JSON-safe dict for the registry's promote payload."""
    m = to_in_project_metrics(scored, perplexity)
    return {
        "edit_similarity": m.edit_similarity,
        "exact_match": m.exact_match,
        "perplexity": m.perplexity,
        "n_examples": m.n_examples,
    }


def noise_band(repeats: Sequence[Dict[str, float]]) -> Tuple[float, str]:
    """Spread of edit similarity across repeated identical baseline evaluations.

    D5 wants a measured band, not the 0.0 placeholder that makes "improved"
    mean "improved by any amount". With greedy decoding the same adapter on the
    same examples is deterministic, so repeats of one configuration give a band
    of exactly 0.0 — which is a true statement about decoding, NOT the sampling
    noise D5 actually wants (that needs repeated TRAINING seeds, which this
    sprint does not run). The returned note says so, so the number cannot be
    quoted as something it is not.
    """
    if len(repeats) < 2:
        return 0.0, ("PLACEHOLDER — fewer than 2 baseline repeats supplied; "
                     "no band measured")
    vals = [r["edit_similarity"] for r in repeats]
    band = max(vals) - min(vals)
    return band, (
        f"spread of {len(vals)} repeated baseline evaluations "
        f"({min(vals):.6f}..{max(vals):.6f}); greedy decoding is deterministic, so "
        f"this is decode-level noise only and NOT the training-seed band D5 wants")
