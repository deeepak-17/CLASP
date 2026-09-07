"""Tests for the in-project completion metric (evaluation/in_project.py) and
the InProjectMetrics / EvalResult contract additions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from evaluation.in_project import (
    CompletionExample,
    InProjectConfig,
    InProjectEvaluator,
    build_completion_examples,
    char_edit_similarity,
    evaluate_client_in_project,
    levenshtein,
    line_exact_match,
    noise_band,
)
from interfaces.contracts import (
    AdapterKind,
    AdapterRef,
    BenchmarkName,
    EvalResult,
    InProjectMetrics,
)
from interfaces.edge_client import GenerationRequest, GenerationResult, MockEdgeInferenceClient
from interfaces.validation import validate_document
from partitions.partitioner import load_partition_manifest
from utils.errors import ContractViolationError, EvaluationError
from utils.paths import project_paths

from tests.conftest import make_record


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class ScriptedClient:
    """Returns a caller-provided completion per task id (default: empty)."""

    model_id = "test/scripted"

    def __init__(self, by_task_id: dict[str, str]) -> None:
        self._by_task_id = by_task_id

    def loaded_adapters(self):  # pragma: no cover - unused by the evaluator
        return ()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            task_id=request.task_id,
            completions=[self._by_task_id.get(request.task_id, "")],
            model_id=self.model_id,
            is_mock=True,
        )

    def generate_batch(self, requests):  # pragma: no cover - unused
        return [self.generate(r) for r in requests]


def _example(target: str, *, prefix: str = "import os\nx = 1\ny = 2\n", line: int = 4) -> CompletionExample:
    return CompletionExample(
        example_id=f"pkg:mod.py#L{line}",
        file_id="pkg:mod.py",
        relative_path="mod.py",
        line_number=line,
        prefix=prefix,
        target=target,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestLevenshtein:
    def test_identical_is_zero(self) -> None:
        assert levenshtein("return x", "return x") == 0

    def test_empty_operands(self) -> None:
        assert levenshtein("", "abc") == 3
        assert levenshtein("abc", "") == 3
        assert levenshtein("", "") == 0

    def test_known_distance(self) -> None:
        assert levenshtein("kitten", "sitting") == 3


class TestCharEditSimilarity:
    def test_identical_is_one(self) -> None:
        assert char_edit_similarity("    return None", "    return None") == 1.0

    def test_both_empty_is_one(self) -> None:
        assert char_edit_similarity("", "") == 1.0

    def test_one_empty_is_zero(self) -> None:
        assert char_edit_similarity("", "return None") == 0.0

    def test_disjoint_is_low(self) -> None:
        assert char_edit_similarity("aaaaaa", "bbbbbb") == pytest.approx(0.0)

    def test_partial_overlap_is_between(self) -> None:
        score = char_edit_similarity("return a + b", "return a - b")
        assert 0.5 < score < 1.0

    def test_always_in_unit_interval(self) -> None:
        for a, b in [("x", "yyyyy"), ("def f():", ""), ("", ""), ("abc", "abcdef")]:
            assert 0.0 <= char_edit_similarity(a, b) <= 1.0


class TestLineExactMatch:
    def test_trailing_whitespace_ignored(self) -> None:
        assert line_exact_match("    return x   ", "    return x")

    def test_leading_whitespace_matters(self) -> None:
        assert not line_exact_match("return x", "    return x")

    def test_different_content(self) -> None:
        assert not line_exact_match("return x", "return y")


class TestNoiseBand:
    def test_single_value_is_zero(self) -> None:
        assert noise_band([0.42]) == 0.0

    def test_empty_is_zero(self) -> None:
        assert noise_band([]) == 0.0

    def test_identical_repeats_is_zero(self) -> None:
        assert noise_band([0.31, 0.31, 0.31]) == 0.0

    def test_known_population_stddev(self) -> None:
        # values 0.0, 0.0, 0.6 -> mean 0.2, popvar = (0.04+0.04+0.16)/3 = 0.08
        assert noise_band([0.0, 0.0, 0.6]) == pytest.approx(0.08**0.5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestInProjectConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = InProjectConfig()
        assert 0.0 < cfg.held_out_fraction < 1.0

    def test_rejects_bad_fraction(self) -> None:
        with pytest.raises(EvaluationError):
            InProjectConfig(held_out_fraction=0.0)
        with pytest.raises(EvaluationError):
            InProjectConfig(held_out_fraction=1.0)

    def test_rejects_nonpositive_caps(self) -> None:
        with pytest.raises(EvaluationError):
            InProjectConfig(max_examples_per_client=0)
        with pytest.raises(EvaluationError):
            InProjectConfig(min_prefix_lines=0)
        with pytest.raises(EvaluationError):
            InProjectConfig(max_new_tokens=0)

    def test_rejects_negative_temperature(self) -> None:
        with pytest.raises(EvaluationError):
            InProjectConfig(temperature=-0.1)


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------
class TestBuildCompletionExamples:
    def _record(self, body: str):
        return make_record("alpha", 0, content=body)

    def test_skips_blank_comment_and_docstring_targets(self) -> None:
        body = (
            "import os\n"      # L1
            "import sys\n"     # L2
            "import re\n"      # L3
            "\n"              # L4 blank
            "# a comment\n"   # L5 comment
            '"""doc"""\n'     # L6 docstring delimiter
            "value = 1\n"     # L7 <- only scorable target (prefix >= 3 non-blank)
        )
        cfg = InProjectConfig(min_prefix_lines=3)
        examples = build_completion_examples([self._record(body)], config=cfg)
        assert [e.line_number for e in examples] == [7]
        assert examples[0].target == "value = 1"
        assert examples[0].prefix.endswith("\n")

    def test_min_prefix_lines_is_respected(self) -> None:
        body = "a = 1\nb = 2\nc = 3\nd = 4\n"
        cfg = InProjectConfig(min_prefix_lines=3)
        examples = build_completion_examples([self._record(body)], config=cfg)
        assert all(e.line_number >= 4 for e in examples)

    def test_cap_is_enforced(self) -> None:
        body = "".join(f"var_{i} = {i}\n" for i in range(50))
        cfg = InProjectConfig(min_prefix_lines=2, max_examples_per_client=5)
        examples = build_completion_examples([self._record(body)], config=cfg)
        assert len(examples) == 5

    def test_is_deterministic_for_a_seed(self) -> None:
        body = "".join(f"var_{i} = {i}\n" for i in range(40))
        cfg = InProjectConfig(min_prefix_lines=2, max_examples_per_client=8)
        first = build_completion_examples([self._record(body)], config=cfg)
        second = build_completion_examples([self._record(body)], config=cfg)
        assert [e.example_id for e in first] == [e.example_id for e in second]

    def test_seed_changes_selection(self) -> None:
        body = "".join(f"var_{i} = {i}\n" for i in range(40))
        base = InProjectConfig(min_prefix_lines=2, max_examples_per_client=8)
        a = build_completion_examples([self._record(body)], config=base)
        b = build_completion_examples([self._record(body)], config=replace(base, seed=999))
        assert [e.example_id for e in a] != [e.example_id for e in b]

    def test_output_is_sorted_by_file_then_line(self) -> None:
        body = "".join(f"var_{i} = {i}\n" for i in range(30))
        cfg = InProjectConfig(min_prefix_lines=2, max_examples_per_client=10)
        examples = build_completion_examples([self._record(body)], config=cfg)
        assert examples == sorted(examples, key=lambda e: (e.file_id, e.line_number))

    def test_empty_content_is_skipped(self) -> None:
        rec = make_record("alpha", 1, content="")
        assert build_completion_examples([rec], config=InProjectConfig()) == []


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class TestInProjectEvaluator:
    def test_perfect_predictions_score_one(self) -> None:
        examples = [_example("    return a + b", line=4), _example("x = compute()", line=5)]
        client = ScriptedClient({e.example_id: e.target for e in examples})
        result = InProjectEvaluator(client).evaluate(
            client_id="c", cluster_id="cl", examples=examples, n_held_out_files=1
        )
        assert result.edit_similarity == 1.0
        assert result.exact_match == 1.0
        assert result.n_examples == 2
        assert result.generation_errors == 0

    def test_constant_prediction_is_scored_in_range(self) -> None:
        examples = [_example("return 1"), _example("return 2"), _example("return 3")]
        client = ScriptedClient({e.example_id: "    pass" for e in examples})
        result = InProjectEvaluator(client).evaluate(
            client_id="c", cluster_id="cl", examples=examples, n_held_out_files=1
        )
        assert 0.0 <= result.edit_similarity <= 1.0
        assert result.exact_match == 0.0

    def test_only_first_predicted_line_is_scored(self) -> None:
        ex = _example("    return x")
        client = ScriptedClient({ex.example_id: "    return x\n    # trailing junk\n"})
        result = InProjectEvaluator(client).evaluate(
            client_id="c", cluster_id="cl", examples=[ex], n_held_out_files=1
        )
        assert result.exact_match == 1.0

    def test_client_error_is_recorded_not_raised(self) -> None:
        ex = _example("    return x")
        client = MockEdgeInferenceClient(fail_task_ids=[ex.example_id])
        result = InProjectEvaluator(client).evaluate(
            client_id="c", cluster_id="cl", examples=[ex], n_held_out_files=1
        )
        assert result.generation_errors == 1
        assert result.per_example[0].error is not None
        assert result.per_example[0].edit_similarity == 0.0

    def test_no_examples_raises(self) -> None:
        with pytest.raises(EvaluationError, match="no eligible completion examples"):
            InProjectEvaluator(ScriptedClient({})).evaluate(
                client_id="c", cluster_id="cl", examples=[], n_held_out_files=0
            )

    def test_to_metrics_yields_valid_contract_object(self) -> None:
        examples = [_example("    return a + b")]
        client = ScriptedClient({e.example_id: e.target for e in examples})
        result = InProjectEvaluator(client).evaluate(
            client_id="c", cluster_id="cl", examples=examples, n_held_out_files=1
        )
        metrics = result.to_metrics()
        assert isinstance(metrics, InProjectMetrics)
        assert metrics.perplexity is None
        assert metrics.n_examples == 1


# ---------------------------------------------------------------------------
# End-to-end against the committed partition manifest
# ---------------------------------------------------------------------------
class TestEvaluateClientInProject:
    @pytest.fixture
    def manifest(self):
        path = project_paths().partitions / "manifest.json"
        if not path.is_file():
            pytest.skip("datasets/partitions/manifest.json not present")
        return load_partition_manifest(path)

    def test_runs_for_a_real_client(self, manifest) -> None:
        client = MockEdgeInferenceClient()
        cfg = InProjectConfig(max_examples_per_client=15)
        result = evaluate_client_in_project(manifest, "client-flask", client, config=cfg)
        assert result.n_examples > 0
        assert result.n_held_out_files >= 1
        assert 0.0 <= result.edit_similarity <= 1.0
        validate_document("eval_result", _wrap(result).to_dict()).raise_for_status()

    def test_held_out_selection_is_deterministic(self, manifest) -> None:
        client = MockEdgeInferenceClient()
        cfg = InProjectConfig(max_examples_per_client=15)
        a = evaluate_client_in_project(manifest, "client-click", client, config=cfg)
        b = evaluate_client_in_project(manifest, "client-click", client, config=cfg)
        assert a.edit_similarity == b.edit_similarity
        assert [e.example_id for e in a.per_example] == [e.example_id for e in b.per_example]

    def test_unknown_client_raises(self, manifest) -> None:
        with pytest.raises(ContractViolationError):
            evaluate_client_in_project(manifest, "client-nope", MockEdgeInferenceClient())


def _wrap(result) -> EvalResult:
    return EvalResult(
        adapter=AdapterRef(name=result.client_id, version=0, kind=AdapterKind.CLIENT),
        benchmark=BenchmarkName.HUMANEVAL,
        in_project=result.to_metrics(),
        num_tasks=result.n_examples,
    )


# ---------------------------------------------------------------------------
# Contract: InProjectMetrics
# ---------------------------------------------------------------------------
class TestInProjectMetricsContract:
    def test_round_trips(self) -> None:
        m = InProjectMetrics(edit_similarity=0.71, exact_match=0.25, n_examples=40, perplexity=3.9)
        assert InProjectMetrics.from_dict(m.to_dict()) == m

    def test_perplexity_optional(self) -> None:
        m = InProjectMetrics(edit_similarity=0.5, exact_match=0.1, n_examples=10)
        assert m.perplexity is None
        assert InProjectMetrics.from_dict(m.to_dict()).perplexity is None

    def test_rejects_out_of_range_similarity(self) -> None:
        with pytest.raises(ContractViolationError):
            InProjectMetrics(edit_similarity=1.5, exact_match=0.1, n_examples=1)
        with pytest.raises(ContractViolationError):
            InProjectMetrics(edit_similarity=0.5, exact_match=-0.1, n_examples=1)

    def test_rejects_nonpositive_perplexity(self) -> None:
        with pytest.raises(ContractViolationError):
            InProjectMetrics(edit_similarity=0.5, exact_match=0.1, n_examples=1, perplexity=0.0)


# ---------------------------------------------------------------------------
# Contract: EvalResult.in_project / baseline_noise_band
# ---------------------------------------------------------------------------
class TestEvalResultInProject:
    def _result(self, **overrides) -> EvalResult:
        base = dict(
            adapter=AdapterRef(name="client-flask", version=1, kind=AdapterKind.CLIENT),
            benchmark=BenchmarkName.HUMANEVAL,
            in_project=InProjectMetrics(edit_similarity=0.8, exact_match=0.3, n_examples=50),
            baseline_noise_band=0.02,
        )
        base.update(overrides)
        return EvalResult(**base)

    def test_round_trips_with_in_project(self) -> None:
        result = self._result()
        restored = EvalResult.from_dict(result.to_dict())
        assert restored.in_project == result.in_project
        assert restored.baseline_noise_band == pytest.approx(0.02)

    def test_validates_against_schema(self) -> None:
        validate_document("eval_result", self._result().to_dict()).raise_for_status()

    def test_guard_only_document_still_valid(self) -> None:
        legacy = {
            "adapter": {"name": "baseline", "version": 0, "kind": "client"},
            "benchmark": "HumanEval",
            "pass_at_k": {"1": 0.0},
            "num_tasks": 5,
            "num_samples_per_task": 1,
            "contract_version": "0.2.0",
        }
        report = validate_document("eval_result", legacy)
        report.raise_for_status()
        assert EvalResult.from_dict(legacy).in_project is None

    def test_negative_noise_band_rejected(self) -> None:
        with pytest.raises(ContractViolationError):
            self._result(baseline_noise_band=-0.01)
