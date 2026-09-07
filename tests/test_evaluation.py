"""Tests for the Week-1/2 evaluation harness scaffold and Week-3 scoring."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.base import BenchmarkAdapter
from evaluation.dependencies import (
    DEPENDENCIES,
    DependencySpec,
    check_dependencies,
    check_dependency,
    render_dependency_report,
)
from evaluation.execution import execute_program
from evaluation.harness import (
    EvaluationHarness,
    compute_pass_at_k,
    load_evaluation_config,
    render_dry_run_report,
)
from evaluation.humaneval.adapter import HumanEvalAdapter
from evaluation.mbpp.adapter import MbppAdapter
from evaluation.models import (
    BenchmarkConfig,
    EvalTask,
    EvaluationConfig,
    RunConfig,
    ScoringConfig,
    TaskOutcome,
    parse_benchmark_name,
)
from evaluation.registry import build_adapter, build_adapters, build_inference_client
from evaluation.results_store import (
    GenerationSnapshot,
    ResultRecord,
    ResultsStoreError,
    append_results,
    load_results_index,
)
from evaluation.scoring import ScoringError, aggregate_pass_at_k, pass_at_k, task_pass_at_k
from interfaces.contracts import AdapterKind, AdapterRef, BenchmarkName, EvalResult
from interfaces.edge_client import MockEdgeInferenceClient
from utils.errors import ConfigError, EvaluationError


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestEvaluationConfig:
    def test_repository_config_loads(self) -> None:
        config = load_evaluation_config()
        assert config.run.mode == "dry_run"
        assert config.backend.kind == "mock"
        assert set(config.enabled_benchmarks()) == set(BenchmarkName)

    def test_rejects_unknown_benchmark_key(self) -> None:
        with pytest.raises(ConfigError, match="Unknown benchmark"):
            EvaluationConfig(benchmarks={"codecontests": BenchmarkConfig()})

    def test_rejects_bad_mode(self) -> None:
        with pytest.raises(ConfigError, match="run.mode"):
            RunConfig(mode="turbo")

    def test_rejects_zero_limit(self) -> None:
        with pytest.raises(ConfigError, match="run.limit"):
            RunConfig(limit=0)

    def test_rejects_empty_pass_at_k(self) -> None:
        with pytest.raises(ConfigError, match="pass_at_k"):
            ScoringConfig(pass_at_k=[])

    def test_parse_benchmark_name_is_case_insensitive(self) -> None:
        assert parse_benchmark_name("humaneval") is BenchmarkName.HUMANEVAL
        assert parse_benchmark_name("MBPP") is BenchmarkName.MBPP

    def test_parse_benchmark_name_rejects_unknown(self) -> None:
        with pytest.raises(EvaluationError, match="Unknown benchmark"):
            parse_benchmark_name("nope")

    def test_disabled_benchmark_is_excluded(self, evaluation_config: EvaluationConfig) -> None:
        config = replace(
            evaluation_config,
            benchmarks={
                **evaluation_config.benchmarks,
                "mbpp": replace(evaluation_config.benchmarks["mbpp"], enabled=False),
            },
        )
        assert config.enabled_benchmarks() == [BenchmarkName.HUMANEVAL]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class TestAdapters:
    @pytest.fixture
    def humaneval(self, evaluation_config: EvaluationConfig) -> HumanEvalAdapter:
        return build_adapter(BenchmarkName.HUMANEVAL, evaluation_config)

    @pytest.fixture
    def mbpp(self, evaluation_config: EvaluationConfig) -> MbppAdapter:
        return build_adapter(BenchmarkName.MBPP, evaluation_config)

    def test_registry_builds_the_right_types(
        self, humaneval: HumanEvalAdapter, mbpp: MbppAdapter
    ) -> None:
        assert isinstance(humaneval, HumanEvalAdapter)
        assert isinstance(mbpp, MbppAdapter)

    def test_bundled_fixtures_load(self, humaneval: HumanEvalAdapter, mbpp: MbppAdapter) -> None:
        assert len(humaneval.load_tasks()) == 5
        assert len(mbpp.load_tasks()) == 5

    def test_limit_is_honoured(self, humaneval: HumanEvalAdapter) -> None:
        assert len(humaneval.load_tasks(limit=2)) == 2

    def test_tasks_declare_the_right_benchmark(
        self, humaneval: HumanEvalAdapter, mbpp: MbppAdapter
    ) -> None:
        assert all(t.benchmark is BenchmarkName.HUMANEVAL for t in humaneval.load_tasks())
        assert all(t.benchmark is BenchmarkName.MBPP for t in mbpp.load_tasks())

    def test_humaneval_prompt_is_verbatim(self, humaneval: HumanEvalAdapter) -> None:
        task = humaneval.load_tasks(limit=1)[0]
        assert humaneval.build_prompt(task) == task.prompt

    def test_mbpp_prompt_pins_the_function_name(self, mbpp: MbppAdapter) -> None:
        task = mbpp.load_tasks(limit=1)[0]
        prompt = mbpp.build_prompt(task)
        assert f"def {task.entry_point}(" in prompt
        assert task.prompt.strip().splitlines()[0] in prompt

    def test_mbpp_prompt_includes_a_worked_example(self, mbpp: MbppAdapter) -> None:
        task = mbpp.load_tasks(limit=1)[0]
        assert "Example: assert" in mbpp.build_prompt(task)

    def test_mbpp_example_can_be_disabled(self, evaluation_config: EvaluationConfig) -> None:
        config = replace(
            evaluation_config,
            benchmarks={
                **evaluation_config.benchmarks,
                "mbpp": replace(
                    evaluation_config.benchmarks["mbpp"], include_test_example_in_prompt=False
                ),
            },
        )
        adapter = build_adapter(BenchmarkName.MBPP, config)
        assert "Example: assert" not in adapter.build_prompt(adapter.load_tasks(limit=1)[0])

    def test_assembled_programs_are_valid_python(
        self, humaneval: HumanEvalAdapter, mbpp: MbppAdapter
    ) -> None:
        """Assemble with each task's canonical solution and parse the result."""
        for adapter in (humaneval, mbpp):
            for task in adapter.load_tasks():
                assert task.canonical_solution is not None
                program = adapter.assemble_program(task, task.canonical_solution)
                ast.parse(program)  # raises SyntaxError on a malformed assembly

    def test_humaneval_assembly_invokes_check(self, humaneval: HumanEvalAdapter) -> None:
        task = humaneval.load_tasks(limit=1)[0]
        program = humaneval.assemble_program(task, "    return 0\n")
        assert program.rstrip().endswith(f"check({task.entry_point})")

    def test_mbpp_assembly_appends_asserts(self, mbpp: MbppAdapter) -> None:
        task = mbpp.load_tasks(limit=1)[0]
        assert "assert " in mbpp.assemble_program(task, "    return None\n")

    def test_truncation_cuts_at_first_stop(self) -> None:
        text = "    return 1\ndef other():\n    pass\n"
        assert BenchmarkAdapter.truncate_completion(text, ["\ndef "]) == "    return 1"

    def test_truncation_picks_the_earliest_stop(self) -> None:
        text = "    a\nclass X:\n    b\ndef y():\n"
        assert BenchmarkAdapter.truncate_completion(text, ["\ndef ", "\nclass "]) == "    a"

    def test_truncation_is_a_noop_without_stops(self) -> None:
        assert BenchmarkAdapter.truncate_completion("    return 1\n", []) == "    return 1\n"

    def test_missing_task_file_falls_back_to_the_fixture(
        self, evaluation_config: EvaluationConfig, tmp_path: Path
    ) -> None:
        config = replace(
            evaluation_config,
            benchmarks={
                **evaluation_config.benchmarks,
                "humaneval": replace(
                    evaluation_config.benchmarks["humaneval"], tasks_path=tmp_path / "absent.jsonl"
                ),
            },
        )
        assert len(build_adapter(BenchmarkName.HUMANEVAL, config).load_tasks()) == 5

    def test_missing_fixture_and_task_file_raises(self, tmp_path: Path) -> None:
        config = EvaluationConfig(
            benchmarks={
                "humaneval": BenchmarkConfig(
                    tasks_path=tmp_path / "a.jsonl", sample_tasks_path=tmp_path / "b.jsonl"
                )
            }
        )
        with pytest.raises(EvaluationError, match="neither tasks_path"):
            build_adapter(BenchmarkName.HUMANEVAL, config).load_tasks()

    def test_benchmark_mismatch_in_task_file_raises(
        self, evaluation_config: EvaluationConfig, tmp_path: Path
    ) -> None:
        from utils.io_utils import write_jsonl

        bad = tmp_path / "wrong.jsonl"
        write_jsonl(
            bad,
            [
                {
                    "task_id": "MBPP/1",
                    "benchmark": "MBPP",
                    "prompt": "do a thing",
                    "entry_point": "f",
                    "test_code": "assert True\n",
                }
            ],
        )
        config = replace(
            evaluation_config,
            benchmarks={
                **evaluation_config.benchmarks,
                "humaneval": replace(evaluation_config.benchmarks["humaneval"], tasks_path=bad),
            },
        )
        with pytest.raises(EvaluationError, match="was loaded by the HumanEval adapter"):
            build_adapter(BenchmarkName.HUMANEVAL, config).load_tasks()


class TestEvalTask:
    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(EvaluationError, match="empty prompt"):
            EvalTask(
                task_id="t", benchmark=BenchmarkName.MBPP, prompt="  ", test_code="", entry_point="f"
            )

    def test_rejects_missing_entry_point(self) -> None:
        with pytest.raises(EvaluationError, match="entry_point"):
            EvalTask(
                task_id="t", benchmark=BenchmarkName.MBPP, prompt="p", test_code="", entry_point=""
            )


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
class TestBackendFactory:
    def test_mock_backend_is_built(self, evaluation_config: EvaluationConfig) -> None:
        client = build_inference_client(evaluation_config)
        assert isinstance(client, MockEdgeInferenceClient)
        assert client.model_id == "mock/test-model"

    def test_edge_backend_is_not_available_before_week_three(
        self, evaluation_config: EvaluationConfig
    ) -> None:
        config = replace(evaluation_config, backend=replace(evaluation_config.backend, kind="edge"))
        with pytest.raises(EvaluationError, match="not integrated until"):
            build_inference_client(config)

    def test_no_enabled_benchmarks_raises(self, evaluation_config: EvaluationConfig) -> None:
        config = replace(
            evaluation_config,
            benchmarks={k: replace(v, enabled=False) for k, v in evaluation_config.benchmarks.items()},
        )
        with pytest.raises(EvaluationError, match="No benchmarks are enabled"):
            build_adapters(config)


# ---------------------------------------------------------------------------
# Harness (Week 1 · Friday)
# ---------------------------------------------------------------------------
class TestEvaluationHarness:
    def test_dry_run_completes(self, evaluation_config: EvaluationConfig) -> None:
        run = EvaluationHarness(evaluation_config).run()
        assert run.ok
        assert len(run.summaries) == 2
        assert run.total_tasks == 4  # limit=2 per benchmark
        assert run.total_completions == 4

    def test_run_is_unscored(self, evaluation_config: EvaluationConfig) -> None:
        run = EvaluationHarness(evaluation_config).run()
        assert run.scored is False
        assert all(summary.scored is False for summary in run.summaries)
        assert all(
            outcome.passed is None
            for outcomes in run.outcomes.values()
            for outcome in outcomes
        )

    def test_artifact_is_written(self, evaluation_config: EvaluationConfig) -> None:
        from utils.io_utils import read_json

        run = EvaluationHarness(evaluation_config).run()
        assert run.artifact_path is not None and run.artifact_path.is_file()
        payload = read_json(run.artifact_path)
        assert payload["scored"] is False
        assert payload["run_id"] == run.run_id
        assert payload["totals"]["tasks"] == 4

    def test_artifact_can_be_suppressed(self, evaluation_config: EvaluationConfig) -> None:
        assert EvaluationHarness(evaluation_config).run(write_artifact=False).artifact_path is None

    def test_limit_override(self, evaluation_config: EvaluationConfig) -> None:
        run = EvaluationHarness(evaluation_config).run(limit=1, write_artifact=False)
        assert run.total_tasks == 2

    def test_multiple_samples_per_task(self, evaluation_config: EvaluationConfig) -> None:
        config = replace(evaluation_config, run=replace(evaluation_config.run, num_samples_per_task=3))
        run = EvaluationHarness(config).run(write_artifact=False)
        assert run.total_completions == 12  # 2 benchmarks x 2 tasks x 3 samples

    def test_is_deterministic(self, evaluation_config: EvaluationConfig) -> None:
        first = EvaluationHarness(evaluation_config).run(write_artifact=False)
        second = EvaluationHarness(evaluation_config).run(write_artifact=False)
        assert first.outcomes["HumanEval"][0].completions == second.outcomes["HumanEval"][0].completions

    def test_task_failure_is_isolated(self, evaluation_config: EvaluationConfig) -> None:
        adapter = build_adapter(BenchmarkName.HUMANEVAL, evaluation_config)
        doomed = adapter.load_tasks(limit=1)[0].task_id
        client = MockEdgeInferenceClient(fail_task_ids=[doomed])

        run = EvaluationHarness(evaluation_config, client=client).run(write_artifact=False)
        assert not run.ok
        humaneval = next(s for s in run.summaries if s.benchmark is BenchmarkName.HUMANEVAL)
        assert humaneval.tasks_failed == 1
        assert humaneval.tasks_succeeded == 1  # the other task still ran

    def test_fail_fast_mode_aborts(self, evaluation_config: EvaluationConfig) -> None:
        adapter = build_adapter(BenchmarkName.HUMANEVAL, evaluation_config)
        doomed = adapter.load_tasks(limit=1)[0].task_id
        config = replace(evaluation_config, run=replace(evaluation_config.run, continue_on_task_error=False))

        with pytest.raises(EvaluationError, match="continue_on_task_error is false"):
            EvaluationHarness(config, client=MockEdgeInferenceClient(fail_task_ids=[doomed])).run(
                write_artifact=False
            )

    def test_privacy_label_defaults_to_dp_off(self, evaluation_config: EvaluationConfig) -> None:
        assert EvaluationHarness(evaluation_config).run(write_artifact=False).privacy_label == (
            "DP off (baseline)"
        )

    def test_report_renders_and_flags_scope(self, evaluation_config: EvaluationConfig) -> None:
        run = EvaluationHarness(evaluation_config).run(write_artifact=False)
        text = render_dry_run_report(run, evaluation_config).render()
        assert "Harness Dry Run" in text
        assert "Pass@k" in text
        assert "out of scope" in text


class TestWeekThreeSeams:
    """The Week-1/2 scaffold deliberately left `compute_pass_at_k` raising
    `NotImplementedError` as a placeholder for the Week-3 deliverable ("Implement
    Pass@k scoring function"). Week 3 replaces the placeholder with the real
    thing (see evaluation.scoring); these tests replace the old
    "is explicitly deferred" assertion with checks that it now works.
    """

    def test_pass_at_k_computes_real_values(self) -> None:
        assert compute_pass_at_k(5, 5, [1, 5]) == {1: 1.0, 5: 1.0}
        assert compute_pass_at_k(5, 0, [1]) == {1: 0.0}

    def test_pass_at_k_omits_k_exceeding_n(self) -> None:
        # Only 3 samples generated; pass@10 cannot be computed from them.
        assert compute_pass_at_k(3, 1, [1, 10]) == {1: pytest.approx(1 / 3)}


# ---------------------------------------------------------------------------
# Dependency preflight (Week 1 · Thursday)
# ---------------------------------------------------------------------------
class TestDependencyPreflight:
    def test_week_one_two_requirements_are_satisfied(self) -> None:
        result = check_dependencies()
        assert result.ok, [status.spec.package for status in result.blocking]

    def test_python_version_is_supported(self) -> None:
        assert check_dependencies().python_ok

    def test_every_spec_declares_a_purpose_and_week(self) -> None:
        for spec in DEPENDENCIES:
            assert spec.purpose
            assert spec.required_from_week >= 1
            assert spec.hint()

    def test_required_now_implies_week_one_or_two(self) -> None:
        for spec in DEPENDENCIES:
            if spec.required_now:
                assert spec.required_from_week <= 2

    def test_absent_package_is_reported_missing(self) -> None:
        spec = DependencySpec(
            package="clasp-not-a-real-package",
            import_name="clasp_not_a_real_package",
            purpose="test",
            required_from_week=1,
            required_now=True,
        )
        status = check_dependency(spec)
        assert not status.installed
        assert not status.ok
        assert status.state == "MISSING"

    def test_absent_but_deferred_package_is_not_blocking(self) -> None:
        spec = DependencySpec(
            package="clasp-not-a-real-package",
            import_name="clasp_not_a_real_package",
            purpose="test",
            required_from_week=9,
            required_now=False,
        )
        assert check_dependency(spec).ok

    def test_namespace_package_is_not_counted_as_installed(self) -> None:
        """The data-only `datasets/` directory must not read as HF `datasets`."""
        spec = DependencySpec(
            package="datasets",
            import_name="datasets",
            purpose="test",
            required_from_week=3,
            required_now=False,
        )
        status = check_dependency(spec)
        if status.installed:
            pytest.skip("the real `datasets` library is installed in this environment")
        assert not status.installed

    def test_outdated_version_is_flagged(self) -> None:
        spec = DependencySpec(
            package="PyYAML",
            import_name="yaml",
            purpose="test",
            required_from_week=1,
            required_now=True,
            min_version="9999.0",
        )
        status = check_dependency(spec)
        assert status.installed
        assert not status.version_ok
        assert "OUTDATED" in status.state

    def test_report_renders(self) -> None:
        text = render_dependency_report(check_dependencies()).render()
        assert "Dependency Report" in text
        assert "PyYAML" in text


# ---------------------------------------------------------------------------
# Pass@k (Week 3 · Wednesday)
# ---------------------------------------------------------------------------
class TestPassAtK:
    """Unbiased pass@k estimator: pass@k = 1 - C(n-c, k) / C(n, k)."""

    def test_pass_at_1_all_fail(self) -> None:
        assert pass_at_k(5, 0, 1) == 0.0

    def test_pass_at_1_all_pass(self) -> None:
        assert pass_at_k(5, 5, 1) == 1.0

    def test_pass_at_1_partial(self) -> None:
        assert pass_at_k(4, 1, 1) == pytest.approx(0.25)

    def test_arbitrary_valid_k(self) -> None:
        # Reference value from Chen et al. 2021's estimator, hand-computed:
        # C(200-50,10)/C(200,10) ~= 0.0521 -> pass@10 ~= 0.9479
        assert pass_at_k(200, 50, 10) == pytest.approx(0.9479, abs=1e-3)

    def test_k_equals_n_with_at_least_one_pass_is_one(self) -> None:
        assert pass_at_k(5, 1, 5) == 1.0

    def test_k_equals_n_with_zero_passes_is_zero(self) -> None:
        assert pass_at_k(5, 0, 5) == 0.0

    def test_c_equals_zero(self) -> None:
        assert pass_at_k(10, 0, 3) == 0.0

    def test_c_equals_n(self) -> None:
        assert pass_at_k(10, 10, 3) == 1.0

    def test_n_equals_zero_raises(self) -> None:
        with pytest.raises(ScoringError, match="undefined for n=0"):
            pass_at_k(0, 0, 1)

    def test_k_greater_than_n_raises(self) -> None:
        with pytest.raises(ScoringError, match="insufficient samples"):
            pass_at_k(5, 2, 10)

    def test_c_greater_than_n_raises(self) -> None:
        with pytest.raises(ScoringError, match="cannot exceed"):
            pass_at_k(5, 6, 1)

    def test_k_below_one_raises(self) -> None:
        with pytest.raises(ScoringError, match="k >= 1"):
            pass_at_k(5, 1, 0)

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ScoringError, match="non-negative"):
            pass_at_k(-1, 0, 1)

    def test_negative_c_raises(self) -> None:
        with pytest.raises(ScoringError, match="non-negative"):
            pass_at_k(5, -1, 1)

    def test_result_always_in_unit_interval(self) -> None:
        for n in range(1, 12):
            for c in range(0, n + 1):
                for k in range(1, n + 1):
                    assert 0.0 <= pass_at_k(n, c, k) <= 1.0

    def test_monotonic_in_k_for_fixed_n_c(self) -> None:
        # More attempts should never lower your chance of at least one pass.
        n, c = 10, 3
        values = [pass_at_k(n, c, k) for k in range(1, n + 1)]
        assert values == sorted(values)

    def test_monotonic_in_c_for_fixed_n_k(self) -> None:
        n, k = 10, 4
        values = [pass_at_k(n, c, k) for c in range(0, n + 1)]
        assert values == sorted(values)


class TestTaskPassAtK:
    def _outcome(self, passed: list[bool] | None) -> TaskOutcome:
        return TaskOutcome(
            task_id="t", benchmark=BenchmarkName.HUMANEVAL, completions=["x"] * (len(passed) if passed else 0), passed=passed
        )

    def test_computes_every_requested_k(self) -> None:
        result = task_pass_at_k(self._outcome([True, False, True, False]), [1, 4])
        assert result.n == 4 and result.c == 2
        assert set(result.computed) == {1, 4}

    def test_skips_k_exceeding_available_samples(self) -> None:
        result = task_pass_at_k(self._outcome([True, False]), [1, 10])
        assert 1 in result.computed
        assert 10 in result.skipped
        assert "insufficient" in result.skipped[10]

    def test_unexecuted_task_raises(self) -> None:
        with pytest.raises(ScoringError, match="not been executed"):
            task_pass_at_k(self._outcome(None), [1])


class TestAggregatePassAtK:
    def _outcome(self, task_id: str, passed: list[bool] | None) -> TaskOutcome:
        return TaskOutcome(
            task_id=task_id,
            benchmark=BenchmarkName.HUMANEVAL,
            completions=["x"] * (len(passed) if passed else 0),
            passed=passed,
        )

    def test_mean_across_tasks(self) -> None:
        outcomes = [self._outcome("a", [True]), self._outcome("b", [False])]
        result = aggregate_pass_at_k(outcomes, [1])
        assert result.pass_at_k[1] == pytest.approx(0.5)
        assert result.tasks_included[1] == 2

    def test_never_executed_task_is_excluded_and_reported(self) -> None:
        outcomes = [self._outcome("a", [True]), self._outcome("b", None)]
        result = aggregate_pass_at_k(outcomes, [1])
        assert result.tasks_included[1] == 1
        assert result.pass_at_k[1] == 1.0
        assert any("b" in reason for reason in result.tasks_skipped[1])

    def test_all_tasks_skipped_reports_zero_not_a_crash(self) -> None:
        outcomes = [self._outcome("a", None)]
        result = aggregate_pass_at_k(outcomes, [1])
        assert result.pass_at_k[1] == 0.0
        assert result.tasks_included[1] == 0

    def test_empty_outcomes(self) -> None:
        result = aggregate_pass_at_k([], [1, 10])
        assert result.pass_at_k == {1: 0.0, 10: 0.0}
        assert result.per_task == []

    def test_per_task_k_skip_does_not_affect_other_ks(self) -> None:
        outcomes = [self._outcome("a", [True, True, False])]  # n=3
        result = aggregate_pass_at_k(outcomes, [1, 10])
        assert result.tasks_included[1] == 1
        assert result.tasks_included[10] == 0
        assert result.tasks_skipped[10]

    def test_to_dict_is_json_shaped(self) -> None:
        outcomes = [self._outcome("a", [True])]
        payload = aggregate_pass_at_k(outcomes, [1]).to_dict()
        assert payload["pass_at_k"] == {"1": 1.0}
        assert payload["tasks_included"] == {"1": 1}


# ---------------------------------------------------------------------------
# Execution (Week 3 · Wednesday support module)
# ---------------------------------------------------------------------------
class TestExecuteProgram:
    def test_passing_program(self) -> None:
        outcome = execute_program("assert 1 + 1 == 2\n")
        assert outcome.passed
        assert outcome.exit_code == 0
        assert not outcome.timed_out

    def test_failing_assertion(self) -> None:
        outcome = execute_program("assert False\n")
        assert not outcome.passed
        assert outcome.exit_code != 0
        assert "AssertionError" in outcome.stderr_tail

    def test_syntax_error_fails_cleanly(self) -> None:
        outcome = execute_program("def broken(:\n")
        assert not outcome.passed
        assert not outcome.timed_out

    def test_timeout_is_reported_not_hung(self) -> None:
        outcome = execute_program("while True:\n    pass\n", timeout_seconds=0.5)
        assert not outcome.passed
        assert outcome.timed_out

    def test_runtime_exception_fails(self) -> None:
        outcome = execute_program("raise NotImplementedError('mock completion')\n")
        assert not outcome.passed
        assert "NotImplementedError" in outcome.stderr_tail

    def test_stdout_does_not_affect_verdict(self) -> None:
        outcome = execute_program("print('hello')\nassert True\n")
        assert outcome.passed

    def test_outcome_is_json_shaped(self) -> None:
        payload = execute_program("assert True\n").to_dict()
        assert payload["passed"] is True
        assert "duration_seconds" in payload

    def test_does_not_leak_into_the_caller_process(self) -> None:
        # The candidate runs in a subprocess; a name it defines must not leak
        # into this test process's globals.
        execute_program("x = 12345\n")
        assert "x" not in globals()


# ---------------------------------------------------------------------------
# Harness integration with real scoring (Week 3)
# ---------------------------------------------------------------------------
class TestHarnessScoring:
    def test_execution_disabled_leaves_passed_none(self, evaluation_config: EvaluationConfig) -> None:
        run = EvaluationHarness(evaluation_config).run(write_artifact=False)
        assert run.scored is False
        assert all(o.passed is None for outcomes in run.outcomes.values() for o in outcomes)

    def test_execution_enabled_scores_every_task(self, evaluation_config: EvaluationConfig) -> None:
        config = replace(evaluation_config, scoring=replace(evaluation_config.scoring, execution_enabled=True))
        run = EvaluationHarness(config).run(write_artifact=False)
        assert run.scored is True
        for outcomes in run.outcomes.values():
            for outcome in outcomes:
                assert outcome.passed is not None
                assert len(outcome.passed) == config.run.num_samples_per_task

    def test_mock_client_never_passes_its_own_tests(self, evaluation_config: EvaluationConfig) -> None:
        """MockEdgeInferenceClient's completions are deliberately incorrect
        (`raise NotImplementedError`) — pass@k against it must be exactly 0.0,
        not merely low. A non-zero value here would mean the executor is not
        actually running the assembled program.
        """
        config = replace(evaluation_config, scoring=replace(evaluation_config.scoring, execution_enabled=True))
        run = EvaluationHarness(config).run(write_artifact=False)
        for summary in run.summaries:
            assert summary.scored is True
            assert all(rate == 0.0 for rate in summary.pass_at_k.values())

    def test_canonical_solutions_pass_their_own_tests(self, evaluation_config: EvaluationConfig) -> None:
        """The inverse check: feeding each task's OWN reference solution back
        through the real execution path must pass. This is the same check
        scripts/sanity_check_scoring.py runs against the full real benchmark;
        here it runs against the bundled fixture so it stays fast and offline.
        """
        for name in BenchmarkName:
            adapter = build_adapter(name, evaluation_config)
            for task in adapter.load_tasks():
                assert task.canonical_solution is not None
                program = adapter.assemble_program(task, task.canonical_solution)
                outcome = execute_program(program, timeout_seconds=10)
                assert outcome.passed, f"{task.task_id} canonical solution failed: {outcome.stderr_tail}"

    def test_run_is_reproducible_when_scored(self, evaluation_config: EvaluationConfig) -> None:
        config = replace(evaluation_config, scoring=replace(evaluation_config.scoring, execution_enabled=True))
        first = EvaluationHarness(config).run(write_artifact=False)
        second = EvaluationHarness(config).run(write_artifact=False)
        assert first.summaries[0].pass_at_k == second.summaries[0].pass_at_k

    def test_insufficient_samples_is_reported_not_hidden(self, evaluation_config: EvaluationConfig) -> None:
        """num_samples_per_task=1 but pass@10 is requested: the summary must
        show 0 tasks included for k=10 rather than silently omitting it or
        crashing.
        """
        config = replace(
            evaluation_config,
            scoring=replace(evaluation_config.scoring, execution_enabled=True, pass_at_k=[1, 10]),
        )
        run = EvaluationHarness(config).run(write_artifact=False)
        for summary in run.summaries:
            assert summary.pass_at_k[10] == 0.0  # nothing could be scored at k=10


# ---------------------------------------------------------------------------
# Real HumanEval/MBPP task data (Week 3 · Monday/Tuesday) — conditional
# ---------------------------------------------------------------------------
class TestRealBenchmarkData:
    """Exercised only when `scripts/fetch_benchmark_data.py` has already been
    run in this checkout (network-dependent, so it is not a hard requirement
    of the offline-first test suite — see evaluation/base.py's fallback).
    """

    def _tasks_path(self, name: str) -> Path:
        from utils.paths import project_paths

        return project_paths().evaluation / name / "tasks.jsonl"

    def test_real_humaneval_loads_and_parses(self) -> None:
        path = self._tasks_path("humaneval")
        if not path.is_file():
            pytest.skip("scripts/fetch_benchmark_data.py has not been run")
        config = EvaluationConfig(benchmarks={"humaneval": BenchmarkConfig(tasks_path=path)})
        tasks = build_adapter(BenchmarkName.HUMANEVAL, config).load_tasks()
        assert len(tasks) == 164
        assert all(t.benchmark is BenchmarkName.HUMANEVAL for t in tasks)
        assert all(t.canonical_solution for t in tasks)

    def test_real_mbpp_loads_and_parses(self) -> None:
        path = self._tasks_path("mbpp")
        if not path.is_file():
            pytest.skip("scripts/fetch_benchmark_data.py has not been run")
        config = EvaluationConfig(benchmarks={"mbpp": BenchmarkConfig(tasks_path=path)})
        tasks = build_adapter(BenchmarkName.MBPP, config).load_tasks()
        assert len(tasks) > 0
        assert all(t.benchmark is BenchmarkName.MBPP for t in tasks)
        assert all(t.metadata.get("signature") for t in tasks)


# ---------------------------------------------------------------------------
# Results store / results.json (Week 3 · Friday)
# ---------------------------------------------------------------------------
class TestResultsStore:
    def _record(self, *, run_id: str = "run-1", benchmark: BenchmarkName = BenchmarkName.HUMANEVAL) -> ResultRecord:
        return ResultRecord(
            eval_result=EvalResult(
                adapter=AdapterRef(name="baseline", version=0, kind=AdapterKind.CLIENT),
                benchmark=benchmark,
                pass_at_k={1: 0.0},
                num_tasks=5,
                num_samples_per_task=1,
                created_at="2026-08-10T00:00:00Z",
                run_id=run_id,
            ),
            model_checkpoint="mock/deepseek-coder-6.7b-base",
            dataset_split="bundled sample fixture (5 tasks)",
            num_problems=5,
            seed=None,
            generation=GenerationSnapshot(max_new_tokens=384, temperature=0.2, stop_sequences=["\ndef "]),
            provenance="DEMO_TEST",
            provenance_note="test fixture",
        )

    def test_record_round_trips(self) -> None:
        record = self._record()
        assert ResultRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()

    def test_record_validates_against_eval_result_schema(self) -> None:
        self._record().validate()  # must not raise

    def test_rejects_bad_provenance(self) -> None:
        with pytest.raises(ResultsStoreError, match="REAL or DEMO_TEST"):
            record = self._record()
            ResultRecord(**{**record.__dict__, "provenance": "FAKE"})

    def test_load_missing_index_is_empty(self, tmp_path: Path) -> None:
        index = load_results_index(tmp_path / "absent.json")
        assert index.records == []

    def test_append_writes_and_is_loadable(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        append_results([self._record()], path)
        reloaded = load_results_index(path)
        assert len(reloaded.records) == 1
        assert reloaded.records[0].eval_result.run_id == "run-1"

    def test_append_accumulates_distinct_runs(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        append_results([self._record(run_id="run-1")], path)
        append_results([self._record(run_id="run-2")], path)
        assert len(load_results_index(path).records) == 2

    def test_append_updates_same_run_and_benchmark_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        append_results([self._record(run_id="run-1")], path)
        updated = self._record(run_id="run-1")
        updated = ResultRecord(**{**updated.__dict__, "num_problems": 999})
        append_results([updated], path)
        index = load_results_index(path)
        assert len(index.records) == 1
        assert index.records[0].num_problems == 999

    def test_distinct_benchmarks_same_run_id_both_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        append_results(
            [self._record(run_id="run-1", benchmark=BenchmarkName.HUMANEVAL),
             self._record(run_id="run-1", benchmark=BenchmarkName.MBPP)],
            path,
        )
        assert len(load_results_index(path).records) == 2

    def test_index_json_round_trips_through_disk(self, tmp_path: Path) -> None:
        from utils.io_utils import read_json

        path = tmp_path / "results.json"
        append_results([self._record()], path)
        raw = read_json(path)
        assert raw["results"][0]["eval_result"]["benchmark"] == "HumanEval"
        assert raw["results"][0]["run_metadata"]["provenance"] == "DEMO_TEST"

    def test_invalid_eval_result_is_rejected_before_writing(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        bad = self._record()
        object.__setattr__(bad, "eval_result", replace(bad.eval_result, num_tasks=-1))
        # num_tasks=-1 fails the eval_result JSON Schema (minimum: 0).
        with pytest.raises(Exception):
            append_results([bad], path)
        assert not path.exists()
