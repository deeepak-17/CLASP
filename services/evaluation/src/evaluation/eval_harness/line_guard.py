"""The frozen line-completion promotion guard (replaces the HumanEval guard).

The guard is a fixed set of next-line completion problems drawn from three
Python repositories (click, jinja, colorama) that no CLASP cluster trains on.
A candidate adapter that loses edit similarity here has forgotten general
Python, whatever it gained in-project. The set is built once by
``scripts/build_line_guard.py`` and committed, because the source repositories
live under the gitignored ``datasets/materialized`` tree.

Problem extraction is ``evaluation.completion.collect_examples``; this module
only samples, stores and compares.
"""
from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

from evaluation.completion import CompletionExample, collect_examples

GUARD_BENCHMARK = "LineCompletion"
GUARD_SIZE = 300
GUARD_SEED = 0
GUARD_SOURCES = ("client-click", "client-jinja", "client-colorama")
GUARD_FILE = Path(__file__).resolve().parents[3] / "datasets" / "guard" / "line_guard_v1.jsonl"

_UNBOUNDED = 10 ** 9


def _quotas(capacities: Sequence[int], size: int) -> List[int]:
    """``size // n`` each, remainder to the earliest sources, shortfall refilled evenly in order."""
    n = len(capacities)
    if size > sum(capacities):
        raise ValueError(f"guard set of {size} requested but sources only hold {sum(capacities)}")
    base, rem = divmod(size, n)
    quotas = [min(cap, base + (1 if i < rem else 0)) for i, cap in enumerate(capacities)]
    missing = size - sum(quotas)
    while missing > 0:
        for i in range(n):
            if missing and quotas[i] < capacities[i]:
                quotas[i] += 1
                missing -= 1
    return quotas


def build_guard_set(source_dirs: Sequence[Path], *, size: int = GUARD_SIZE,
                    seed: int = GUARD_SEED) -> List[CompletionExample]:
    """Seeded sample of ``size`` examples spread evenly across the sources.

    Each ``source_dir`` holds a ``repo/`` directory; the directory's name
    prefixes every example's ``file``. Output order (source order, then file
    order within a source) is a pure function of ``(sources, size, seed)``.
    """
    rng = random.Random(seed)
    pools: List[List[CompletionExample]] = []
    for src in source_dirs:
        src = Path(src)
        found = collect_examples(src / "repo", max_examples=_UNBOUNDED)
        pools.append([replace(e, file=f"{src.name}/{e.file}") for e in found])
    quotas = _quotas([len(p) for p in pools], size)
    out: List[CompletionExample] = []
    for pool, quota in zip(pools, quotas):
        picked = sorted(rng.sample(range(len(pool)), quota))
        out.extend(pool[i] for i in picked)
    return out


def save_guard_set(examples: Sequence[CompletionExample], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for e in examples:
            fh.write(json.dumps({"file": e.file, "line_no": e.line_no,
                                 "prompt": e.prompt, "target": e.target},
                                ensure_ascii=False) + "\n")


def load_guard_set(path: Path = GUARD_FILE) -> List[CompletionExample]:
    out: List[CompletionExample] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(CompletionExample(file=d["file"], line_no=d["line_no"],
                                             prompt=d["prompt"], target=d["target"]))
    return out


def guard_metrics(scored: Dict[str, float], n_examples: int) -> Dict[str, object]:
    """The ``GuardMetrics.to_json`` shape for a scored guard run."""
    return {"benchmark": GUARD_BENCHMARK,
            "edit_similarity": float(scored["edit_similarity"]),
            "exact_match": float(scored["exact_match"]),
            "n_examples": int(n_examples)}


def paired_noise(base_rows: Sequence[float], cand_rows: Sequence[float], *,
                 n_boot: int = 1000, seed: int = 0) -> float:
    """Half-width of the 95 % bootstrap interval of the mean paired difference.

    Example indices are resampled with replacement, the same indices for both
    runs. Identical (or all-equal) differences give exactly 0.0.
    """
    if len(base_rows) != len(cand_rows) or not base_rows:
        raise ValueError("paired_noise needs equal-length, non-empty rows")
    diffs = [c - b for b, c in zip(base_rows, cand_rows)]
    if all(d == diffs[0] for d in diffs):
        return 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(round(0.975 * (n_boot - 1)))]
    return (hi - lo) / 2.0
