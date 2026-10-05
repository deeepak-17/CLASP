"""Command line for ``registry.runs`` — see that module's docstring for usage."""
from __future__ import annotations

import argparse
import json
import sys

from .runs import (
    ConfigError,
    ResultStore,
    RunFrozen,
    check_configs,
    expand_sweep,
    load_config,
    run_id_for,
    run_sweep,
)


def _check(args: argparse.Namespace) -> int:
    problems = check_configs(args.experiments_dir)
    for path, issues in problems.items():
        for issue in issues:
            print(f"{path}: {issue}")
    print(f"{len(problems)} config(s) with problems" if problems else "all experiment configs valid")
    return 1 if problems else 0


def _plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    points = expand_sweep(config) or [{}]
    for point in points:
        print(f"{run_id_for(config, point)}  {json.dumps(point)}")
    total = float(config["gpu_hours_estimate"]) * len(points)
    print(f"{len(points)} run(s), {total:g} GPU-hours estimated")
    return 0


def _run(args: argparse.Namespace) -> int:
    command = [token for token in args.command if token]
    if not command:
        print("give the command after --, e.g. -- python train.py --rank {rank}", file=sys.stderr)
        return 2
    summary = run_sweep(args.config, command, ResultStore(args.store), dry_run=args.dry_run)
    print(json.dumps(summary.__dict__))
    return 1 if summary.failed else 0


def _status(args: argparse.Namespace) -> int:
    for r in ResultStore(args.store).list_runs(args.experiment):
        hours = (r.get("manifest") or {}).get("gpu_hours_actual")
        print(f"{r['experiment']}/{r['run_id']}  {r['status']:<9}  gpu_h={hours}  {r.get('point')}")
    return 0


def _freeze(args: argparse.Namespace) -> int:
    ResultStore(args.store).freeze(args.experiment, tag=args.tag)
    print(f"{args.experiment} frozen as {args.tag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m registry.runs")
    ap.add_argument("--store", default=None, help="result store root (default: CLASP_RESULT_STORE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="validate every experiments/*/config.yaml (CI)")
    p.add_argument("experiments_dir", nargs="?", default="experiments")
    p.set_defaults(func=_check)
    p = sub.add_parser("plan", help="list the sweep's runs and GPU-hours")
    p.add_argument("config")
    p.set_defaults(func=_plan)
    p = sub.add_parser("run", help="run or resume a sweep")
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=_run)
    p = sub.add_parser("status", help="list archived runs")
    p.add_argument("experiment", nargs="?")
    p.set_defaults(func=_status)
    p = sub.add_parser("freeze", help="mark an experiment's results final")
    p.add_argument("experiment")
    p.add_argument("--tag", required=True)
    p.set_defaults(func=_freeze)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    try:
        return args.func(args)
    except (ConfigError, RunFrozen, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
