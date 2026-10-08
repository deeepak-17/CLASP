"""Seeded sampling — the same seed reproduces a run, independent of task order."""

from __future__ import annotations

import dataclasses

import pytest

from evaluation.eval_harness import EvaluationHarness
from evaluation.eval_harness.models import EvaluationConfig, GenerationConfig
from evaluation.interfaces.edge_client import GenerationRequest, MockEdgeInferenceClient, task_seed
from evaluation.utils.errors import ConfigError


class _Spy(MockEdgeInferenceClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest):
        self.requests.append(request)
        return super().generate(request)


def test_task_seed_is_stable_and_task_specific() -> None:
    assert task_seed(0, "HumanEval/0") == task_seed(0, "HumanEval/0")
    assert task_seed(0, "HumanEval/0") != task_seed(0, "HumanEval/1")
    assert task_seed(0, "HumanEval/0") != task_seed(1, "HumanEval/0")
    assert 0 <= task_seed(123, "x") < 2**31


def test_harness_passes_the_configured_seed(evaluation_config: EvaluationConfig) -> None:
    config = dataclasses.replace(
        evaluation_config, generation=dataclasses.replace(evaluation_config.generation, seed=42)
    )
    spy = _Spy()
    EvaluationHarness(config, client=spy).run(write_artifact=False)
    assert spy.requests and all(r.seed == 42 for r in spy.requests)


def test_seed_changes_sampled_output_and_repeats_reproduce_it() -> None:
    client = MockEdgeInferenceClient()
    make = lambda seed: GenerationRequest(task_id="t", prompt="def f():", temperature=0.2, seed=seed)  # noqa: E731
    assert client.generate(make(1)).completions == client.generate(make(1)).completions
    assert client.generate(make(1)).completions != client.generate(make(2)).completions


def test_negative_seed_rejected() -> None:
    with pytest.raises(ConfigError):
        GenerationConfig(seed=-1)
