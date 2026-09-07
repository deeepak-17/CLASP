#!/usr/bin/env python3
"""Baseline Pass@k run, stored to results.json.

Thu: "Run Pass@k on the Week-1 baseline model; sanity-check the numbers."
Fri: "Store results in a simple results.json the dashboard can read."

What "the Week-1 baseline model" means for P5
-----------------------------------------------
P1's Week-1 deliverable was "load DeepSeek-Coder-6.7B in 4-bit; run baseline
probe; record pass@1 as the anchor number" — that model and that number are
P1's, not reproduced here. What *this* script runs is P5's own baseline: the
harness exactly as configured for Week 1/2 (``backend.kind: mock``), now with
real scoring switched on, against the real HumanEval/MBPP problem sets when
present. That is the correct Week-3 baseline for this repository's own
pipeline, and its purpose is different from a capability measurement: it
proves the scoring path is correct end to end and gives a reproducible
reference point to diff future *real* backend runs against.

Whether a real model is available is checked explicitly (see
``evaluation.dependencies``); when it is not — as in this environment, no
GPU, no downloaded DeepSeek-Coder-6.7B weights, no P1 adapter — this script
refuses to silently substitute a mock and call it a measurement. Every
artefact it writes is labelled ``DEMO_TEST`` and says why.

Usage::

    python scripts/run_baseline_eval.py
    python scripts/run_baseline_eval.py --limit 20
    python scripts/run_baseline_eval.py --num-samples 5 --k 1 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.harness import EvaluationHarness, load_evaluation_config
from evaluation.results_store import GenerationSnapshot, ResultRecord, append_results, default_results_path
from interfaces.contracts import AdapterKind, AdapterRef
from interfaces.edge_client import MockEdgeInferenceClient
from utils.io_utils import read_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport

_LOG = get_logger(__name__)

_BASELINE_MODEL_ID = "mock/deepseek-coder-6.7b-base"
_DEMO_NOTE = (
    "DEMO/TEST — P1's real merged DeepSeek-Coder-6.7B model is not available in this "
    "environment (no GPU, no downloaded weights, no LoRA adapter from the Registry). "
    "Completions come from interfaces.edge_client.MockEdgeInferenceClient, a deterministic "
    "placeholder generator that always produces a syntactically valid but incorrect body "
    "(`raise NotImplementedError(...)`). Pass@k computed against it is therefore expected to "
    "be 0.0 for every task and every k — that is the CORRECT output of a working scorer given "
    "a generator that never produces a passing program, not a bug. It is NOT a measurement of "
    "any model's coding ability. See scripts/sanity_check_scoring.py for a REAL end-to-end "
    "check (real HumanEval/MBPP + real execution + canonical reference solutions, no mock)."
)


def _dataset_split(benchmark_dir: Path) -> tuple[str, int, str | None]:
    """Describe which task set was actually loaded, from its provenance manifest."""
    manifest_path = benchmark_dir / "tasks_manifest.json"
    tasks_path = benchmark_dir / "tasks.jsonl"
    if manifest_path.is_file() and tasks_path.is_file():
        manifest = read_json(manifest_path)
        return manifest["split"], manifest["tasks_written"], manifest["source_url"]
    return (
        "bundled sample fixture (5 tasks) — NOT the published benchmark; "
        "run scripts/fetch_benchmark_data.py for the real problem set",
        5,
        None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="Evaluation config. Defaults to configs/evaluation.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Tasks per benchmark. Defaults to all loaded tasks.")
    parser.add_argument("--num-samples", type=int, default=None, help="Samples per task. Overrides run.num_samples_per_task.")
    parser.add_argument("--k", type=int, nargs="+", default=None, help="k values for Pass@k. Overrides scoring.pass_at_k.")
    parser.add_argument(
        "--output", type=Path, default=None, help="results.json path. Defaults to evaluation/results/results.json."
    )
    return parser


def main(args: argparse.Namespace) -> int:
    config = load_evaluation_config(args.config)
    config = replace(config, scoring=replace(config.scoring, execution_enabled=True))
    if args.num_samples:
        config = replace(config, run=replace(config.run, num_samples_per_task=args.num_samples))
    if args.k:
        config = replace(config, scoring=replace(config.scoring, pass_at_k=sorted(set(args.k))))

    client = MockEdgeInferenceClient(model_id=_BASELINE_MODEL_ID)
    harness = EvaluationHarness(config, client=client)
    run = harness.run(limit=args.limit, write_artifact=True)

    paths = project_paths()
    records: list[ResultRecord] = []
    sanity_failures: list[str] = []

    for summary in run.summaries:
        benchmark_dir = paths.evaluation / summary.benchmark.value.lower()
        split, expected_problems, source_url = _dataset_split(benchmark_dir)

        # --- sanity checks (Week 3 Thursday) --------------------------------
        if summary.tasks_attempted > expected_problems and args.limit is None:
            sanity_failures.append(
                f"{summary.benchmark.value}: attempted {summary.tasks_attempted} tasks but the "
                f"dataset manifest declares {expected_problems}"
            )
        for k, rate in summary.pass_at_k.items():
            if not 0.0 <= rate <= 1.0:
                sanity_failures.append(f"{summary.benchmark.value}: pass@{k}={rate} outside [0, 1]")
        if summary.tasks_succeeded > summary.tasks_attempted:
            sanity_failures.append(f"{summary.benchmark.value}: succeeded > attempted")

        adapter_ref = AdapterRef(name="baseline", version=0, kind=AdapterKind.CLIENT)
        eval_result_pass_at_k = {k: v for k, v in summary.pass_at_k.items()}
        from interfaces.contracts import EvalResult

        eval_result = EvalResult(
            adapter=adapter_ref,
            benchmark=summary.benchmark,
            pass_at_k=eval_result_pass_at_k,
            num_tasks=summary.tasks_attempted,
            num_samples_per_task=config.run.num_samples_per_task,
            created_at=run.created_at,
            run_id=run.run_id,
        )
        records.append(
            ResultRecord(
                eval_result=eval_result,
                model_checkpoint=_BASELINE_MODEL_ID,
                dataset_split=split,
                num_problems=summary.tasks_attempted,
                seed=None,
                generation=GenerationSnapshot(
                    max_new_tokens=config.generation.max_new_tokens,
                    temperature=config.generation.temperature,
                    stop_sequences=list(config.generation.stop_sequences),
                ),
                provenance="DEMO_TEST",
                provenance_note=_DEMO_NOTE,
                raw_artifact_path=paths.relative(run.artifact_path) if run.artifact_path else None,
                benchmark_data_source=source_url,
            )
        )

    output_path = args.output or default_results_path()
    written_path = append_results(records, output_path)

    lines = [
        "",
        "CLASP-P5 · Baseline Pass@k Run",
        "=" * 68,
        f"  run id:    {run.run_id}",
        f"  model:     {_BASELINE_MODEL_ID}  (DEMO/TEST — see notes)",
        "",
    ]
    for summary in run.summaries:
        lines.append(
            f"    {summary.benchmark.value:<10} tasks={summary.tasks_attempted:<4} "
            f"samples/task={config.run.num_samples_per_task:<3} pass@k={summary.pass_at_k}"
        )
    lines.append("")
    lines.append("  Sanity checks:")
    if sanity_failures:
        for failure in sanity_failures:
            lines.append(f"    FAIL  {failure}")
    else:
        lines.append("    ok    passed <= attempted for every benchmark")
        lines.append("    ok    every pass@k in [0, 1]")
        lines.append("    ok    task counts match the loaded dataset/limit")
    lines.append("")
    lines.append(f"  results.json: {paths.relative(written_path)}")
    if run.artifact_path:
        lines.append(f"  raw artefact: {paths.relative(run.artifact_path)}")
    lines.append("")
    lines.append("  NOTE: pass@k == 0.0 above is EXPECTED — see the DEMO/TEST note in results.json.")
    lines.append("  For a real (non-mock) sanity check, run scripts/sanity_check_scoring.py.")
    lines.append("")

    if not args.no_report:
        report = _render_report(run, records, sanity_failures, config)
        path = report.write(paths.reports / "baseline_eval_report.md")
        lines.append(report_line(path))
        lines.append("")

    emit(lines)
    return EXIT_OK if not sanity_failures else EXIT_FAILURE


def _render_report(run, records: list[ResultRecord], sanity_failures: list[str], config) -> MarkdownReport:
    report = MarkdownReport(
        title="CLASP-P5 · Baseline Pass@k Report",
        subtitle="baseline run, sanity-checked, stored to results.json",
    )
    report.heading("1. Verdict")
    report.status_line(not sanity_failures, f"{len(sanity_failures)} sanity-check failure(s)")
    if sanity_failures:
        report.bullets(sanity_failures)

    report.heading("2. REAL vs DEMO/TEST")
    report.paragraph(_DEMO_NOTE)

    report.heading("3. Results")
    report.table(
        ["Benchmark", "Dataset split", "Problems", "Samples/task", "Pass@k"],
        [
            [
                r.eval_result.benchmark.value,
                r.dataset_split,
                r.num_problems,
                r.eval_result.num_samples_per_task,
                ", ".join(f"pass@{k}={v:.4f}" for k, v in sorted(r.eval_result.pass_at_k.items())),
            ]
            for r in records
        ],
    )

    report.heading("4. Generation configuration")
    report.key_values(
        {
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
            "execution_timeout_seconds": config.scoring.execution_timeout_seconds,
            "run_id": run.run_id,
            "created_at": run.created_at,
        }
    )

    report.rule()
    report.paragraph(
        "Generated by `scripts/run_baseline_eval.py`. Real (non-mock) sanity check: "
        "`scripts/sanity_check_scoring.py`."
    )
    return report


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
