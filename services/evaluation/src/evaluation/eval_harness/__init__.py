"""Evaluation harness for HumanEval and MBPP — the Week-1 deliverable.

Layout::

    eval_harness/
      models.py        normalised task/outcome types and the config schema
      base.py          BenchmarkAdapter base class
      registry.py      adapter registry + inference-backend factory
      dependencies.py  Week-1 Thursday dependency preflight
      harness.py       the runner
      humaneval/       HumanEval adapter + offline sample fixture
      mbpp/            MBPP adapter + offline sample fixture
      results/         run artefacts (git-ignored except .gitkeep)

Scope: Weeks 1-2 build, dry-run and validate the harness structure. Pass@k
scoring and wiring to P1's merged model are Week-3 deliverables and are marked
as seams in :mod:`eval_harness.harness`.
"""

from evaluation.eval_harness.base import BenchmarkAdapter
from evaluation.eval_harness.dependencies import (
    DependencyCheckResult,
    check_dependencies,
    render_dependency_report,
)
from evaluation.eval_harness.harness import (
    EvaluationHarness,
    HarnessRun,
    load_evaluation_config,
    render_dry_run_report,
)
from evaluation.eval_harness.in_project import (
    InProjectConfig,
    InProjectEvalResult,
    InProjectEvaluator,
    evaluate_client_in_project,
    noise_band,
)
from evaluation.eval_harness.models import BenchmarkRunSummary, EvalTask, EvaluationConfig, TaskOutcome
from evaluation.eval_harness.registry import build_adapter, build_adapters, build_inference_client

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BenchmarkAdapter",
    "BenchmarkRunSummary",
    "DependencyCheckResult",
    "EvalTask",
    "EvaluationConfig",
    "EvaluationHarness",
    "HarnessRun",
    "InProjectConfig",
    "InProjectEvalResult",
    "InProjectEvaluator",
    "TaskOutcome",
    "build_adapter",
    "build_adapters",
    "build_inference_client",
    "check_dependencies",
    "evaluate_client_in_project",
    "load_evaluation_config",
    "noise_band",
    "render_dependency_report",
    "render_dry_run_report",
]
