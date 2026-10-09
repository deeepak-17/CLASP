"""Experiment machinery: run store (manifest v2), sweep expansion, resume, CI check.

Rules under test (D8/D9):
  * a config must state a seed and a per-run GPU-hours estimate before it runs
  * D8 caps (clients, rank, seq len, steps, rounds) are enforced unless the
    config overrides them WITH a written justification
  * the sweep's total estimate must fit its budget
  * completed runs are immutable; a re-run of a sweep skips them (resume)
  * every archived output is listed with its sha256
"""
from __future__ import annotations

import json
import shlex
import sys

import pytest
import yaml
from pathlib import Path
from registry.runs import (
    ConfigError,
    ResultStore,
    RunFrozen,
    check_configs,
    expand_sweep,
    load_config,
    run_id_for,
    run_sweep,
    validate_config,
)

BASE = {
    "name": "rank-sweep",
    "seed": 0,
    "gpu_hours_estimate": 0.5,
    "budget": {"gpu_hours_total": 10},
    "clients": 6, "rank": 16, "seq_len": 1024, "max_steps": 200, "rounds": 5,
    "sweep": {"rank": [4, 8, 16], "alpha": [0.5, 1.0]},
}


def _write(tmp_path, cfg, name="config.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_valid_config_has_no_problems():
    assert validate_config(BASE) == []


@pytest.mark.parametrize("missing", ["seed", "gpu_hours_estimate", "name"])
def test_required_fields(missing):
    cfg = {k: v for k, v in BASE.items() if k != missing}
    assert any(missing in p for p in validate_config(cfg))


@pytest.mark.parametrize("field,value", [
    ("clients", 7), ("rank", 32), ("seq_len", 2048), ("max_steps", 201), ("rounds", 6),
])
def test_d8_caps(field, value):
    problems = validate_config({**BASE, field: value, "sweep": {}})
    assert any(field in p and "D8" in p for p in problems)


def test_sweep_values_are_capped_too():
    problems = validate_config({**BASE, "sweep": {"rank": [16, 64]}})
    assert any("rank" in p for p in problems)


def test_caps_override_needs_a_justification():
    over = {**BASE, "sweep": {"rank": [16, 64]}, "caps_override": {"rank": 64}}
    assert any("justification" in p for p in validate_config(over))
    ok = {**over, "caps_override": {"rank": 64, "justification": "A100 available (D8 raise)"}}
    assert validate_config(ok) == []


def test_budget_must_cover_the_sweep():
    cfg = {**BASE, "budget": {"gpu_hours_total": 2}}  # 6 points x 0.5 = 3
    assert any("budget" in p for p in validate_config(cfg))


def test_load_config_raises_on_problems(tmp_path):
    with pytest.raises(ConfigError, match="seed"):
        load_config(_write(tmp_path, {k: v for k, v in BASE.items() if k != "seed"}))


# --------------------------------------------------------------------------- #
# sweep expansion
# --------------------------------------------------------------------------- #
def test_expand_is_cartesian_and_deterministic():
    points = expand_sweep(BASE)
    assert len(points) == 6
    assert points[0] == {"rank": 4, "alpha": 0.5}
    assert expand_sweep(BASE) == points


def test_expand_with_seeds_repeats_every_point():
    points = expand_sweep({**BASE, "seeds": [0, 1]})
    assert len(points) == 12 and {p["seed"] for p in points} == {0, 1}


def test_no_sweep_is_one_point():
    assert expand_sweep({**BASE, "sweep": {}}) == [{}]


def test_run_id_is_stable_and_point_specific():
    a = run_id_for(BASE, {"rank": 4, "alpha": 0.5})
    assert a == run_id_for(BASE, {"alpha": 0.5, "rank": 4})
    assert a != run_id_for(BASE, {"rank": 8, "alpha": 0.5})
    assert a.startswith("rank-sweep-")


# --------------------------------------------------------------------------- #
# running, archiving, resuming
# --------------------------------------------------------------------------- #
PY = shlex.quote(sys.executable)  # the venv path may contain spaces
WRITER = (
    f"{PY} -c \"import json,pathlib,sys; d=pathlib.Path(sys.argv[1]); "
    "d.mkdir(parents=True, exist_ok=True); "
    "(d/'metrics.json').write_text(json.dumps({'rank': int(sys.argv[2])}))\" "
    "{outputs_dir} {rank}"
)


def test_run_sweep_archives_every_point_with_manifest_v2(tmp_path):
    store = ResultStore(tmp_path / "store")
    cfg_path = _write(tmp_path, {**BASE, "sweep": {"rank": [4, 8]}})
    summary = run_sweep(cfg_path, WRITER, store)
    assert summary.completed == 2 and summary.skipped == 0 and summary.failed == 0

    records = store.list_runs("rank-sweep")
    assert len(records) == 2
    rec = records[0]
    assert rec["manifest_version"] == 2
    assert rec["status"] == "completed"
    assert rec["manifest"]["seed"] == 0
    assert rec["manifest"]["gpu_hours_estimate"] == 0.5
    assert rec["manifest"]["gpu_hours_actual"] is not None
    assert len(rec["manifest"]["config_hash"]) == 64
    assert rec["outputs"][0]["path"] == "metrics.json"
    assert len(rec["outputs"][0]["sha256"]) == 64
    run_dir = store.run_dir("rank-sweep", rec["run_id"])
    assert json.loads((run_dir / "outputs" / "metrics.json").read_text())["rank"] in (4, 8)
    assert (run_dir / "config.resolved.json").exists()


def test_rerunning_a_sweep_resumes_skipping_completed(tmp_path):
    store = ResultStore(tmp_path / "store")
    cfg_path = _write(tmp_path, {**BASE, "sweep": {"rank": [4, 8]}})
    run_sweep(cfg_path, WRITER, store)
    again = run_sweep(cfg_path, WRITER, store)
    assert again.completed == 0 and again.skipped == 2


def test_failed_point_is_recorded_and_retried_on_resume(tmp_path):
    store = ResultStore(tmp_path / "store")
    cfg_path = _write(tmp_path, {**BASE, "sweep": {"rank": [4, 8]}})
    flaky = f"{PY} -c \"import sys; sys.exit(1 if {{rank}} == 8 else 0)\""
    first = run_sweep(cfg_path, flaky, store)
    assert first.completed == 1 and first.failed == 1
    failed = [r for r in store.list_runs("rank-sweep") if r["status"] == "failed"]
    assert failed[0]["exit_code"] == 1

    second = run_sweep(cfg_path, WRITER, store)
    assert second.skipped == 1 and second.completed == 1
    assert {r["status"] for r in store.list_runs("rank-sweep")} == {"completed"}


def test_interrupted_running_record_is_retried(tmp_path):
    """A crash mid-run leaves status=running; resume must not treat it as done."""
    store = ResultStore(tmp_path / "store")
    cfg_path = _write(tmp_path, {**BASE, "sweep": {"rank": [4]}})
    cfg = load_config(cfg_path)
    rid = run_id_for(cfg, {"rank": 4})
    store.write_record("rank-sweep", rid, {"run_id": rid, "status": "running"})
    assert run_sweep(cfg_path, WRITER, store).completed == 1


def test_completed_runs_are_immutable(tmp_path):
    store = ResultStore(tmp_path / "store")
    run_sweep(_write(tmp_path, {**BASE, "sweep": {"rank": [4]}}), WRITER, store)
    rid = store.list_runs("rank-sweep")[0]["run_id"]
    with pytest.raises(RunFrozen):
        store.write_record("rank-sweep", rid, {"run_id": rid, "status": "running"})


def test_dry_run_plans_without_running(tmp_path):
    store = ResultStore(tmp_path / "store")
    summary = run_sweep(_write(tmp_path, BASE), WRITER, store, dry_run=True)
    assert summary.planned == 6 and summary.completed == 0
    assert summary.gpu_hours_estimate == 3.0
    assert store.list_runs("rank-sweep") == []


def test_index_lists_every_run(tmp_path):
    store = ResultStore(tmp_path / "store")
    run_sweep(_write(tmp_path, {**BASE, "sweep": {"rank": [4, 8]}}), WRITER, store)
    index = json.loads((tmp_path / "store" / "index.json").read_text())
    assert len(index["runs"]) == 2
    assert all(r["status"] == "completed" for r in index["runs"])


def test_freeze_marks_an_experiment_final(tmp_path):
    store = ResultStore(tmp_path / "store")
    run_sweep(_write(tmp_path, {**BASE, "sweep": {"rank": [4]}}), WRITER, store)
    store.freeze("rank-sweep", tag="paper-v1")
    assert store.is_frozen("rank-sweep")
    with pytest.raises(RunFrozen):
        run_sweep(_write(tmp_path, {**BASE, "sweep": {"rank": [4, 8]}}), WRITER, store)


# --------------------------------------------------------------------------- #
# CI completeness check over experiments/
# --------------------------------------------------------------------------- #
def test_check_configs_reports_each_bad_config(tmp_path):
    (tmp_path / "good").mkdir()
    _write(tmp_path / "good", BASE)
    (tmp_path / "bad").mkdir()
    _write(tmp_path / "bad", {"name": "bad"})
    (tmp_path / "_template").mkdir()
    _write(tmp_path / "_template", {"name": "template"})  # templates are skipped
    problems = check_configs(tmp_path)
    assert set(problems) == {str(tmp_path / "bad" / "config.yaml")}


def test_repo_experiment_configs_pass_the_check():
    from pathlib import Path

    experiments = Path(__file__).resolve().parents[3] / "experiments"
    assert check_configs(experiments) == {}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_plan_run_status_freeze(tmp_path, capsys):
    from registry.runs_cli import main

    cfg = _write(tmp_path, {**BASE, "sweep": {"rank": [4, 8]}})
    store = str(tmp_path / "store")
    assert main(["--store", store, "plan", str(cfg)]) == 0
    assert "2 run(s), 1 GPU-hours estimated" in capsys.readouterr().out
    assert main(["--store", store, "run", str(cfg), "--", *shlex.split(WRITER)]) == 0
    capsys.readouterr()
    assert main(["--store", store, "status", "rank-sweep"]) == 0
    assert capsys.readouterr().out.count("  completed  ") == 2
    assert main(["--store", store, "freeze", "rank-sweep", "--tag", "v1"]) == 0
    assert main(["--store", store, "run", str(cfg), "--", "true"]) == 2  # frozen


def test_cli_check_exit_codes(tmp_path):
    from registry.runs_cli import main

    (tmp_path / "bad").mkdir()
    _write(tmp_path / "bad", {"name": "bad"})
    assert main(["check", str(tmp_path)]) == 1
    assert main(["check", str(Path(__file__).resolve().parents[3] / "experiments")]) == 0


def test_cli_run_needs_a_command(tmp_path):
    from registry.runs_cli import main

    assert main(["--store", str(tmp_path / "s"), "run", str(_write(tmp_path, BASE))]) == 2
