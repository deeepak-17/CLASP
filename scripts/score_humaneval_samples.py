#!/usr/bin/env python3
"""Score HumanEval samples into a D5 guard anchor that edge.promote.resolve_guard reads.

The edge lane writes greedy completions as evalplus samples
(``edge.humaneval_baseline`` -> ``eval_out/samples.jsonl``) but could not
score them on Windows, so D5's HumanEval guard stayed unavailable. This runs
each sample against the published HumanEval tests on P5's executor and writes
``anchor.json`` with ``base_pass_at_1`` and the frozen ``subset``.

Score the candidate and the baseline with this same script, then hand both
anchors to the round::

    python scripts/score_humaneval_samples.py \
        --samples services/edge/eval_out/samples.jsonl \
        --generation-manifest services/edge/eval_out/manifest.json \
        --out eval_out/baseline/anchor.json

    python scripts/demo_round.py --round 1 \
        --baseline-anchor eval_out/baseline/anchor.json \
        --candidate-anchor eval_out/candidate/anchor.json

Executes model-generated code (process-level isolation only — see
eval_harness/execution.py). Run it on completions from the team's own model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, run_cli, setup_logging
from eval_harness.guard_anchor import build_anchor, load_samples, score_samples
from eval_harness.harness import load_evaluation_config
from eval_harness.registry import build_adapter
from interfaces.contracts import BenchmarkName
from utils.io_utils import read_json
from utils.timing import Stopwatch


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--samples", type=Path, required=True, help="evalplus samples JSONL (task_id + solution).")
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=None,
        help="edge.humaneval_baseline's manifest.json (model, decoding, frozen subset) to merge into the anchor.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where to write anchor.json.")
    parser.add_argument("--config", type=Path, default=None, help="Evaluation config (HumanEval task file).")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-program execution timeout, seconds.")
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_evaluation_config(args.config)
    adapter = build_adapter(BenchmarkName.HUMANEVAL, config)
    tasks = {task.task_id: task for task in adapter.load_tasks()}
    if len(tasks) < 164:
        emit([f"  WARNING: only {len(tasks)} HumanEval tasks loaded — run scripts/fetch_benchmark_data.py first."])

    samples = load_samples(args.samples)
    manifest = read_json(args.generation_manifest) if args.generation_manifest else None

    with Stopwatch("score") as watch:
        outcomes = score_samples(samples, tasks, timeout_seconds=args.timeout)
    anchor = build_anchor(outcomes, samples_path=args.samples, generation_manifest=manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(anchor, indent=2) + "\n", encoding="utf-8")

    failed = [o for o in outcomes if not o.passed]
    lines = [
        "",
        "CLASP-P5 · HumanEval guard anchor",
        "=" * 68,
        f"  samples        {args.samples}  ({len(samples)} sample(s), sha256 {anchor['samples']['sha256'][:12]}…)",
        f"  problems       {anchor['n_problems']}",
        f"  base pass@1    {anchor['base_pass_at_1']:.4f}  ({anchor['base_solved']}/{anchor['n_problems']} solved)",
        "  plus pass@1    not computed (EvalPlus extended tests not run)",
        f"  elapsed        {watch.elapsed_seconds:.2f}s",
        f"  anchor         {args.out}",
    ]
    if failed:
        lines.append(f"  failed         {', '.join(o.task_id for o in failed)}")
    timeouts = [o.task_id for o in outcomes if o.timed_out]
    if timeouts:
        lines.append(f"  timed out      {', '.join(timeouts)}")
    lines.append("")
    emit(lines)
    return EXIT_OK if anchor["n_problems"] else EXIT_FAILURE


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
