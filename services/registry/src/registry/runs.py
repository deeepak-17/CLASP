"""Experiment machinery (D8/D9): validated configs, resumable sweeps, a result store.

    python -m registry.runs check  [experiments_dir]          # CI: every config valid
    python -m registry.runs plan   <config.yaml>               # points + GPU-hours, no run
    python -m registry.runs run    <config.yaml> -- <command>  # run / resume the sweep
    python -m registry.runs status [experiment]                # what is done
    python -m registry.runs freeze <experiment> --tag <tag>    # results are final

A config must state ``seed`` and a per-run ``gpu_hours_estimate`` before it
runs, stay inside the D8 caps (or override them with a written
``justification``), and fit its ``budget.gpu_hours_total``.

Result store layout (``CLASP_RESULT_STORE``, default ``.runtime/results``)::

    index.json                                 every run, rebuilt on each write
    runs/<experiment>/FROZEN                   present once the results are final
    runs/<experiment>/<run_id>/run.json        manifest v2 (below)
    runs/<experiment>/<run_id>/config.resolved.json
    runs/<experiment>/<run_id>/outputs/...     whatever the command wrote

Manifest v2 = the contracts' ``RunManifest`` (seed, config hash, adapter
versions, GPU-hours estimate AND actual) plus status, the sweep point, the
command, exit code, timestamps, git commit, environment and a sha256 for every
output. ``run_id`` is a hash of (config, point), so re-running a sweep resumes
it: completed points are skipped, failed or interrupted ones are re-run, and a
completed run can never be overwritten.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from contracts import RunManifest, utcnow_iso

from .manifest import config_hash

MANIFEST_VERSION = 2
DEFAULT_STORE = ".runtime/results"
STORE_ENV = "CLASP_RESULT_STORE"

#: D8 compute-conservative caps. Raise per config only with a justification.
D8_CAPS = {"clients": 6, "rank": 16, "seq_len": 1024, "max_steps": 200, "rounds": 5}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class ConfigError(ValueError):
    pass


class RunFrozen(RuntimeError):
    """Completed runs and frozen experiments are immutable."""


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cap_problems(config: dict) -> list[str]:
    override = config.get("caps_override") or {}
    problems = []
    if override and not str(override.get("justification", "")).strip():
        problems.append("caps_override needs a justification (D8: raise caps only with reason)")
    sweep = config.get("sweep") or {}
    for field, cap in D8_CAPS.items():
        limit = override.get(field, cap)
        values = [config[field]] if _number(config.get(field)) else []
        values += [v for v in sweep.get(field, []) if _number(v)]
        worst = max(values, default=None)
        if worst is not None and worst > limit:
            problems.append(f"{field}={worst} exceeds the D8 cap of {limit}")
    return problems


def expand_sweep(config: dict) -> list[dict]:
    """Cartesian product of ``sweep`` (key order kept), times ``seeds`` if given."""
    sweep = config.get("sweep") or {}
    keys = list(sweep)
    points = [dict(zip(keys, combo)) for combo in itertools.product(*(sweep[k] for k in keys))]
    seeds = config.get("seeds")
    if seeds:
        points = [{**p, "seed": s} for p in points for s in seeds]
    return points


def validate_config(config: dict) -> list[str]:
    """Every reason this config may not run yet; empty means it may."""
    problems = []
    if not isinstance(config.get("name"), str) or not config.get("name"):
        problems.append("name is required")
    if not isinstance(config.get("seed"), int) or isinstance(config.get("seed"), bool):
        problems.append("seed is required (an integer) — D9 reproducibility")
    estimate = config.get("gpu_hours_estimate")
    if not _number(estimate) or estimate < 0:
        problems.append("gpu_hours_estimate is required (per run, >= 0) — D8: estimate first")
    problems += _cap_problems(config)
    budget = (config.get("budget") or {}).get("gpu_hours_total")
    if _number(estimate) and _number(budget):
        total = estimate * len(expand_sweep(config))
        if total > budget:
            problems.append(f"sweep needs {total:g} GPU-hours, over its budget of {budget:g}")
    return problems


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: not a mapping")
    problems = validate_config(config)
    if problems:
        raise ConfigError(f"{path}: " + "; ".join(problems))
    return config


def check_configs(experiments_dir: str | Path) -> dict[str, list[str]]:
    """Problems for every ``*/config.yaml`` under ``experiments_dir`` (``_*`` skipped)."""
    found = {}
    for path in sorted(Path(experiments_dir).glob("*/config.yaml")):
        if path.parent.name.startswith("_"):
            continue
        config = yaml.safe_load(path.read_text())
        problems = validate_config(config if isinstance(config, dict) else {})
        if problems:
            found[str(path)] = problems
    return found


def run_id_for(config: dict, point: dict) -> str:
    digest = config_hash({"config": config, "point": dict(sorted(point.items()))})
    return f"{config['name']}-{digest[:10]}"


# --------------------------------------------------------------------------- #
# result store
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ResultStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get(STORE_ENV, DEFAULT_STORE))
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    def experiment_dir(self, experiment: str) -> Path:
        return self.root / "runs" / experiment

    def run_dir(self, experiment: str, run_id: str) -> Path:
        return self.experiment_dir(experiment) / run_id

    def is_frozen(self, experiment: str) -> bool:
        return (self.experiment_dir(experiment) / "FROZEN").exists()

    def read_record(self, experiment: str, run_id: str) -> dict | None:
        path = self.run_dir(experiment, run_id) / "run.json"
        return json.loads(path.read_text()) if path.exists() else None

    def write_record(self, experiment: str, run_id: str, record: dict) -> None:
        if self.is_frozen(experiment):
            raise RunFrozen(f"{experiment} is frozen")
        existing = self.read_record(experiment, run_id)
        if existing and existing.get("status") == "completed":
            raise RunFrozen(f"{experiment}/{run_id} is completed and immutable")
        run_dir = self.run_dir(experiment, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(run_dir / "run.json", json.dumps(record, indent=2, sort_keys=True))
        self._reindex()

    def list_runs(self, experiment: str | None = None) -> list[dict]:
        base = self.root / "runs"
        pattern = f"{experiment}/*/run.json" if experiment else "*/*/run.json"
        return [json.loads(p.read_text()) for p in sorted(base.glob(pattern))]

    def freeze(self, experiment: str, *, tag: str) -> None:
        unfinished = [r["run_id"] for r in self.list_runs(experiment) if r["status"] != "completed"]
        if unfinished:
            raise RunFrozen(f"cannot freeze {experiment}: unfinished runs {unfinished}")
        _atomic_write(self.experiment_dir(experiment) / "FROZEN",
                      json.dumps({"tag": tag, "frozen_at": utcnow_iso()}))
        self._reindex()

    def _reindex(self) -> None:
        runs = [{
            "experiment": r.get("experiment"), "run_id": r.get("run_id"),
            "status": r.get("status"), "point": r.get("point"),
            "gpu_hours_actual": (r.get("manifest") or {}).get("gpu_hours_actual"),
        } for r in self.list_runs()]
        _atomic_write(self.root / "index.json",
                      json.dumps({"updated_at": utcnow_iso(), "runs": runs}, indent=2))


# --------------------------------------------------------------------------- #
# running a sweep
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepSummary:
    planned: int
    completed: int
    skipped: int
    failed: int
    gpu_hours_estimate: float


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _render(command: str | list[str], values: dict[str, Any]) -> list[str]:
    """argv with ``{key}`` placeholders filled — never through a shell.

    A string is shlex-split first (substituted values are quoted so a path
    with spaces stays one argument); a list is taken as argv already.
    """
    if isinstance(command, str):
        def quoted(m: re.Match) -> str:
            key = m.group(1)
            return shlex.quote(str(values[key])) if key in values else m.group(0)
        return shlex.split(_PLACEHOLDER.sub(quoted, command))

    def plain(m: re.Match) -> str:
        return str(values[m.group(1)]) if m.group(1) in values else m.group(0)
    return [_PLACEHOLDER.sub(plain, token) for token in command]


def _archive(outputs_dir: Path) -> list[dict]:
    if not outputs_dir.exists():
        return []
    return [
        {"path": str(p.relative_to(outputs_dir)), "sha256": _sha256(p), "bytes": p.stat().st_size}
        for p in sorted(outputs_dir.rglob("*")) if p.is_file()
    ]


def _run_point(config: dict, point: dict, command: str | list[str], store: ResultStore,
               commit: str | None) -> bool:
    name, run_id = config["name"], run_id_for(config, point)
    run_dir = store.run_dir(name, run_id)
    resolved = {**config, **point}
    seed = point.get("seed", config["seed"])
    gpus = config.get("gpus", 1)
    manifest = RunManifest(
        run_id=run_id, seed=seed, config_hash=config_hash(resolved),
        adapter_versions=dict(config.get("adapter_versions") or {}),
        gpu_hours_estimate=float(config["gpu_hours_estimate"]),
        notes=str(config.get("notes", "")),
    )
    record = {
        "manifest_version": MANIFEST_VERSION, "experiment": name, "run_id": run_id,
        "status": "running", "point": point, "git_commit": commit,
        "command": command if isinstance(command, str) else shlex.join(command),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "started_at": utcnow_iso(), "manifest": manifest.to_json(),
    }
    store.write_record(name, run_id, record)
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))

    outputs = run_dir / "outputs"
    values = {**resolved, "seed": seed, "run_dir": run_dir, "outputs_dir": outputs,
              "run_id": run_id}
    start = time.perf_counter()
    try:
        exit_code = subprocess.run(_render(command, values), check=False).returncode
    except OSError as e:
        exit_code, record["error"] = 127, str(e)
    hours = (time.perf_counter() - start) / 3600 * gpus
    record.update({
        "status": "completed" if exit_code == 0 else "failed", "exit_code": exit_code,
        "finished_at": utcnow_iso(), "outputs": _archive(outputs),
        "manifest": {**manifest.to_json(), "gpu_hours_actual": round(hours, 6)},
    })
    store.write_record(name, run_id, record)
    return exit_code == 0


def run_sweep(config_path: str | Path, command: str | list[str], store: ResultStore, *,
              dry_run: bool = False) -> SweepSummary:
    """Run every not-yet-completed point of the sweep; safe to call again to resume."""
    config = load_config(config_path)
    if store.is_frozen(config["name"]):
        raise RunFrozen(f"{config['name']} is frozen; start a new experiment instead")
    points = expand_sweep(config) or [{}]
    estimate = float(config["gpu_hours_estimate"]) * len(points)
    if dry_run:
        return SweepSummary(len(points), 0, 0, 0, estimate)
    commit = _git_commit()
    completed = skipped = failed = 0
    for point in points:
        existing = store.read_record(config["name"], run_id_for(config, point))
        if existing and existing.get("status") == "completed":
            skipped += 1
            continue
        if _run_point(config, point, command, store, commit):
            completed += 1
        else:
            failed += 1
    return SweepSummary(len(points), completed, skipped, failed, estimate)


if __name__ == "__main__":
    from .runs_cli import main

    sys.exit(main())
