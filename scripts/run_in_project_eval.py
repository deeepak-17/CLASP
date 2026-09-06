#!/usr/bin/env python3
"""In-project completion metric — "in-project eval v1" (D5 primary metric).

Runs next-line completion over every federated client's held-out ``.py`` files
and reports edit similarity + exact match — the two
:class:`~interfaces.contracts.InProjectMetrics` fields P1's training loop does
not measure. Repeats each client's evaluation ``--repeats`` times to measure
``EvalResult.baseline_noise_band``: the smallest in-project gain D5 should
treat as signal rather than run-to-run jitter.

What it writes
--------------
* ``evaluation/results/in_project/<run_id>.json`` — per-client metrics, the
  noise band, and a schema-valid ``EvalResult`` per client (the shape P4's
  promotion rule reads).
* ``reports/in_project_eval_report.md``.

REAL vs DEMO/TEST
-----------------
With no GPU / merged model / trained adapter in this environment the backend
is :class:`~interfaces.edge_client.MockEdgeInferenceClient`, which never emits
real code — edit similarity is low and exact match ~0 by construction, and the
noise band is 0.0 because every repeat is byte-identical. Every artefact is
labelled ``DEMO_TEST``. Point ``--backend edge`` at P1's client (once
registered) for a real measurement.

Usage::

    python scripts/run_in_project_eval.py
    python scripts/run_in_project_eval.py --client client-flask --repeats 5
    python scripts/run_in_project_eval.py --limit 20 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from _cli import EXIT_FAILURE, EXIT_OK, base_parser, emit, report_line, run_cli, setup_logging
from evaluation.in_project import (
    InProjectConfig,
    evaluate_client_in_project,
    noise_band,
)
from evaluation.registry import build_inference_client
from evaluation.models import BackendConfig, EvaluationConfig
from interfaces.contracts import AdapterKind, AdapterRef, BenchmarkName, EvalResult
from interfaces.edge_client import MockEdgeInferenceClient
from interfaces.validation import jsonschema_available, validate_document
from partitions.partitioner import load_partition_manifest
from utils.errors import ClaspP5Error
from utils.io_utils import write_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.timing import file_timestamp, utc_timestamp

_LOG = get_logger(__name__)

_DEMO_NOTE = (
    "DEMO/TEST — P1's merged DeepSeek-Coder model is not available here (no GPU, no "
    "downloaded weights, no LoRA adapter). Completions come from MockEdgeInferenceClient, "
    "which emits a deterministic placeholder line, so edit_similarity is low, exact_match "
    "is ~0, and baseline_noise_band is 0.0 (every repeat is identical). These are the "
    "CORRECT outputs of a working metric given a generator that never produces real code, "
    "NOT a measurement of any model. Run with --backend edge against P1's client for a "
    "real number."
)


def _resolve_clients(manifest, requested: list[str] | None) -> list[str]:
    known = [shard.client_id for shard in manifest.shards]
    if not requested:
        return known
    unknown = [c for c in requested if c not in known]
    if unknown:
        raise ClaspP5Error(
            f"Unknown client(s): {', '.join(unknown)}. Known: {', '.join(known)}"
        )
    return requested


def _build_client(backend: str):
    if backend == "mock":
        return MockEdgeInferenceClient(model_id="mock/deepseek-coder-6.7b-base"), "DEMO_TEST"
    # backend == "edge": defer to the registry factory (P1 registers it).
    client = build_inference_client(
        EvaluationConfig(backend=BackendConfig(kind="edge", model_id="edge/merged"))
    )
    return client, "REAL"


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=None, help="Partition manifest. Defaults to datasets/partitions/manifest.json.")
    parser.add_argument("--client", action="append", default=None, help="Client id to evaluate (repeatable). Default: every client in the manifest.")
    parser.add_argument("--repeats", type=int, default=3, help="Evaluations per client, for the noise band.")
    parser.add_argument("--limit", type=int, default=None, help="Max completion examples per client. Overrides the default cap.")
    parser.add_argument("--seed", type=int, default=None, help="Held-out split + example-selection seed. Overrides the default.")
    parser.add_argument("--backend", choices=["mock", "edge"], default="mock", help="Inference backend.")
    parser.add_argument("--output", type=Path, default=None, help="Artefact path. Defaults to evaluation/results/in_project/<run_id>.json.")
    return parser


def main(args: argparse.Namespace) -> int:
    if args.repeats < 1:
        raise ClaspP5Error("--repeats must be >= 1")

    paths = project_paths()
    manifest_path = Path(args.manifest or paths.partitions / "manifest.json")
    if not manifest_path.is_file():
        raise ClaspP5Error(
            f"Partition manifest not found at {manifest_path}. Run "
            "`python scripts/build_partitions.py` first."
        )
    manifest = load_partition_manifest(manifest_path)

    config = InProjectConfig()
    if args.limit is not None:
        config = replace(config, max_examples_per_client=args.limit)
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    clients = _resolve_clients(manifest, args.client)
    client, provenance = _build_client(args.backend)
    run_id = f"p5-inproj-{file_timestamp()}"
    created_at = utc_timestamp()

    per_client: list[dict] = []
    schema_failures: list[str] = []
    warnings: list[str] = []

    for client_id in clients:
        repeats = [
            evaluate_client_in_project(manifest, client_id, client, config=config)
            for _ in range(args.repeats)
        ]
        primary = repeats[0]
        band = noise_band([r.edit_similarity for r in repeats])
        if primary.generation_errors:
            warnings.append(f"{client_id}: {primary.generation_errors} completion request(s) errored")

        metrics = primary.to_metrics()
        eval_result = EvalResult(
            adapter=AdapterRef(name=client_id, version=0, kind=AdapterKind.CLIENT),
            benchmark=BenchmarkName.HUMANEVAL,  # guard slot; the guard is scored separately
            pass_at_k={},
            num_tasks=primary.n_examples,
            num_samples_per_task=1,
            created_at=created_at,
            run_id=run_id,
            in_project=metrics,
            baseline_noise_band=band,
        )
        report = validate_document("eval_result", eval_result.to_dict(), document_name=f"{client_id} EvalResult")
        if not report.ok:
            schema_failures.append(f"{client_id}: " + "; ".join(report.errors))

        per_client.append(
            {
                "client_id": client_id,
                "cluster_id": primary.cluster_id,
                "repeats": args.repeats,
                "edit_similarity_repeats": [round(r.edit_similarity, 6) for r in repeats],
                "baseline_noise_band": round(band, 6),
                "in_project": primary.to_dict(include_examples=False),
                "metrics": metrics.to_dict(),
                "eval_result": eval_result.to_dict(),
            }
        )
        _LOG.info(
            "%s: edit_sim=%.4f exact=%.4f n=%d band=%.4f",
            client_id,
            primary.edit_similarity,
            primary.exact_match,
            primary.n_examples,
            band,
        )

    artefact = {
        "run_id": run_id,
        "created_at": created_at,
        "kind": "in_project_eval",
        "version": "1.0.0",
        "backend": args.backend,
        "provenance": provenance,
        "provenance_note": _DEMO_NOTE if provenance == "DEMO_TEST" else "",
        "manifest_path": paths.relative(manifest_path),
        "config": {
            "seed": config.seed,
            "held_out_fraction": config.held_out_fraction,
            "max_examples_per_client": config.max_examples_per_client,
            "min_prefix_lines": config.min_prefix_lines,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "repeats": args.repeats,
        },
        "jsonschema_validation": "enabled" if jsonschema_available() else "SKIPPED (jsonschema absent)",
        "clients": per_client,
    }

    output_path = Path(args.output or (paths.eval_results / "in_project" / f"{run_id}.json"))
    write_json(output_path, artefact)

    lines = [
        "",
        "CLASP-P5 · In-Project Completion Metric (in-project eval v1)",
        "=" * 68,
        f"  run id:    {run_id}",
        f"  backend:   {args.backend}  ({provenance})",
        f"  manifest:  {paths.relative(manifest_path)}",
        f"  repeats:   {args.repeats}   seed: {config.seed}",
        "",
        f"    {'client':<20} {'edit_sim':>9} {'exact':>8} {'n':>5} {'noise_band':>11}",
        f"    {'-' * 20} {'-' * 9} {'-' * 8} {'-' * 5} {'-' * 11}",
    ]
    for entry in per_client:
        m = entry["metrics"]
        lines.append(
            f"    {entry['client_id']:<20} {m['edit_similarity']:>9.4f} {m['exact_match']:>8.4f} "
            f"{m['n_examples']:>5} {entry['baseline_noise_band']:>11.4f}"
        )
    lines.append("")
    lines.append("  Schema checks:")
    if schema_failures:
        for failure in schema_failures:
            lines.append(f"    FAIL  {failure}")
    else:
        lines.append(f"    ok    {len(per_client)} EvalResult document(s) valid against eval_result.schema.json")
    for warning in warnings:
        lines.append(f"    warn  {warning}")
    lines.append("")
    lines.append(f"  artefact:  {paths.relative(output_path)}")
    if provenance == "DEMO_TEST":
        lines.append("")
        lines.append("  NOTE: low edit_sim / exact ~0 / band 0.0 is EXPECTED with the mock backend.")
    lines.append("")

    if not args.no_report:
        report = _render_report(artefact, per_client, schema_failures, warnings)
        path = report.write(paths.reports / "in_project_eval_report.md")
        lines.append(report_line(path))
        lines.append("")

    emit(lines)
    return EXIT_OK if not schema_failures else EXIT_FAILURE


def _render_report(artefact: dict, per_client: list[dict], schema_failures: list[str], warnings: list[str]) -> MarkdownReport:
    report = MarkdownReport(
        title="CLASP-P5 · In-Project Completion Metric",
        subtitle="in-project eval v1 — D5 primary metric (edit similarity + exact match on held-out client files)",
    )

    report.heading("1. Verdict")
    report.status_line(
        not schema_failures,
        f"{len(per_client)} client(s) evaluated, {len(schema_failures)} schema failure(s), {len(warnings)} warning(s)",
    )

    report.heading("2. REAL vs DEMO/TEST")
    report.paragraph(artefact["provenance_note"] or "REAL backend — numbers are a measurement.")

    report.heading("3. Per-client metrics")
    report.table(
        ["Client", "Cluster", "edit_similarity", "exact_match", "n_examples", "held-out files", "noise band"],
        [
            [
                e["client_id"],
                e["cluster_id"],
                f"{e['metrics']['edit_similarity']:.4f}",
                f"{e['metrics']['exact_match']:.4f}",
                e["metrics"]["n_examples"],
                e["in_project"]["n_held_out_files"],
                f"{e['baseline_noise_band']:.4f}",
            ]
            for e in per_client
        ],
    )

    report.heading("4. What each number is")
    report.bullets(
        [
            "**edit_similarity** — mean character-level `1 - lev(pred, target) / max(len)` over held-out "
            "next-line completions (CodeXGLUE convention). In `[0, 1]`; 1.0 is a perfect line.",
            "**exact_match** — fraction of held-out lines reproduced exactly (trailing whitespace ignored).",
            "**perplexity** — `null` here: P5 has no logits. P1 supplies its held-out perplexity at the "
            "integration seam and it is merged into `InProjectMetrics` there.",
            "**noise band** — population standard deviation of edit_similarity across "
            f"{artefact['config']['repeats']} repeated evaluations. D5 promotes only when the in-project gain "
            "exceeds this. 0.0 with a deterministic backend — it becomes a real gate once a stochastic "
            "backend makes the repeats differ.",
        ]
    )

    report.heading("5. Held-out split")
    report.paragraph(
        f"Seed `{artefact['config']['seed']}`, fraction `{artefact['config']['held_out_fraction']:.0%}`, "
        "via `partitions.materialize.split_held_out` — byte-identical to the slice materialized for P1, so "
        "no client is scored on a file it trained on."
    )

    if schema_failures:
        report.heading("6. Schema failures", level=3)
        report.bullets(schema_failures)
    if warnings:
        report.heading("Warnings", level=3)
        report.bullets(warnings)

    report.rule()
    report.paragraph("Generated by `scripts/run_in_project_eval.py`. See `docs/in_project_eval.md`.")
    return report


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    setup_logging(parsed)
    sys.exit(run_cli(main, parsed))
