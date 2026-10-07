"""Tests for eval_harness/guard_anchor.py — scoring edge's HumanEval samples into a D5 guard anchor.

Uses the real HumanEval task file and the real executor; the samples are the
tasks' own canonical solutions (must pass) and deliberately wrong bodies (must
fail), so no model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.guard_anchor import assemble_program, build_anchor, load_samples, score_samples
from eval_harness.registry import build_adapter
from interfaces.contracts import BenchmarkName
from utils.errors import EvaluationError


@pytest.fixture(scope="module")
def tasks():
    from eval_harness.harness import load_evaluation_config

    adapter = build_adapter(BenchmarkName.HUMANEVAL, load_evaluation_config())
    loaded = {t.task_id: t for t in adapter.load_tasks()}
    if "HumanEval/0" not in loaded or loaded["HumanEval/0"].canonical_solution is None:
        pytest.skip("real HumanEval task file not present (run scripts/fetch_benchmark_data.py)")
    return loaded


def _good(task) -> dict:
    return {"task_id": task.task_id, "solution": task.prompt + task.canonical_solution}


def _bad(task) -> dict:
    return {"task_id": task.task_id, "solution": task.prompt + "    return None\n"}


def _write(tmp_path: Path, samples: list[dict]) -> Path:
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    return path


class TestLoadSamples:
    def test_reads_evalplus_lines(self, tmp_path) -> None:
        path = _write(tmp_path, [{"task_id": "HumanEval/0", "solution": "x"}, {"task_id": "HumanEval/1", "completion": "y"}])
        assert [s["task_id"] for s in load_samples(path)] == ["HumanEval/0", "HumanEval/1"]

    @pytest.mark.parametrize("line", ['{"solution": "x"}', '{"task_id": "HumanEval/0"}', "not json"])
    def test_rejects_malformed(self, tmp_path, line) -> None:
        path = tmp_path / "s.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(EvaluationError):
            load_samples(path)

    def test_rejects_empty(self, tmp_path) -> None:
        with pytest.raises(EvaluationError):
            load_samples(_write(tmp_path, []))


class TestScoring:
    def test_canonical_solutions_pass_wrong_bodies_fail(self, tasks) -> None:
        ids = ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
        outcomes = score_samples([_good(tasks[i]) for i in ids] + [_bad(tasks[i]) for i in ids], tasks)
        assert [o.passed for o in outcomes] == [True] * 3 + [False] * 3
        assert all(o.entry_point_defined for o in outcomes)

    def test_completion_form_is_appended_to_the_prompt(self, tasks) -> None:
        task = tasks["HumanEval/2"]
        sample = {"task_id": task.task_id, "completion": task.canonical_solution}
        assert assemble_program(task, sample).startswith(task.prompt)
        assert score_samples([sample], tasks)[0].passed

    def test_unknown_task_id_is_an_error(self, tasks) -> None:
        with pytest.raises(EvaluationError, match="not in the HumanEval task file"):
            score_samples([{"task_id": "HumanEval/9999", "solution": "pass"}], tasks)

    def test_infinite_loop_times_out(self, tasks) -> None:
        task = tasks["HumanEval/0"]
        sample = {"task_id": task.task_id, "solution": task.prompt + "    while True:\n        pass\n"}
        outcome = score_samples([sample], tasks, timeout_seconds=1.0)[0]
        assert not outcome.passed and outcome.timed_out


class TestBuildAnchor:
    def test_anchor_carries_what_resolve_guard_reads(self, tasks, tmp_path) -> None:
        samples = [_good(tasks["HumanEval/0"]), _bad(tasks["HumanEval/1"])]
        path = _write(tmp_path, samples)
        anchor = build_anchor(score_samples(samples, tasks), samples_path=path)
        assert anchor["base_pass_at_1"] == 0.5
        assert anchor["subset"] == {"n": 2, "task_ids": ["HumanEval/0", "HumanEval/1"]}
        assert anchor["plus_pass_at_1"] is None
        assert anchor["scorer"]["name"].startswith("clasp-p5")
        assert len(anchor["samples"]["sha256"]) == 64

    def test_generation_manifest_is_merged_and_subset_enforced(self, tasks, tmp_path) -> None:
        samples = [_good(tasks["HumanEval/0"]), _good(tasks["HumanEval/1"])]
        path = _write(tmp_path, samples)
        manifest = {"model_id": "deepseek-ai/deepseek-coder-1.3b-base", "subset": {"n": 2, "task_ids": ["HumanEval/0", "HumanEval/1"]}}
        anchor = build_anchor(score_samples(samples, tasks), samples_path=path, generation_manifest=manifest)
        assert anchor["model_id"] == manifest["model_id"] and anchor["base_pass_at_1"] == 1.0

        wrong = dict(manifest, subset={"n": 3, "task_ids": ["HumanEval/0", "HumanEval/1", "HumanEval/2"]})
        with pytest.raises(EvaluationError, match="frozen subset"):
            build_anchor(score_samples(samples, tasks), samples_path=path, generation_manifest=wrong)

    def test_several_samples_per_task_use_the_unbiased_estimator(self, tasks, tmp_path) -> None:
        task = tasks["HumanEval/0"]
        samples = [_good(task), _bad(task), _bad(task), _bad(task)]
        anchor = build_anchor(score_samples(samples, tasks), samples_path=_write(tmp_path, samples))
        assert anchor["base_pass_at_1"] == 0.25 and anchor["base_solved"] == 1

    def test_resolve_guard_accepts_the_anchor(self, tasks, tmp_path) -> None:
        promote = pytest.importorskip("edge.promote")
        samples = [_good(tasks["HumanEval/0"]), _bad(tasks["HumanEval/1"])]
        anchor_path = tmp_path / "anchor.json"
        anchor = build_anchor(score_samples(samples, tasks), samples_path=_write(tmp_path, samples))
        anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
        status = promote.resolve_guard(anchor_path, anchor_path)
        assert status.available
        assert status.candidate == {"benchmark": "HumanEval", "pass_at_k": {"1": 0.5}}
