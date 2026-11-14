"""Feed a round's results into evaluation's React dashboard (demo plan v3 item 11).

    python scripts/feed_dashboard.py                    # newest rounds -> services/evaluation/results
    python scripts/feed_dashboard.py --build            # ... and rebuild the dashboard image too

Runs P5's own exporters, unchanged, on the round manifests the demo produced:

* ``export_round_feed.py`` on the four-seam round manifests
  (``scripts/demo_round.py`` -> ``experiments/w12-integration/results/round*_manifest.json``)
  -> ``results/rounds.json`` (Rounds page: versions, D5 decision, alerts);
* ``build_personalization_report.py`` and ``export_lineage.py`` on the edge
  round manifests (default: evaluation's vendored real rounds, plus any given
  with ``--edge-rounds``) -> ``results/personalization.json``, ``results/lineage.json``.

The dashboard bakes ``services/evaluation/results`` in at build time
(``npm run build`` / the compose ``dashboard`` image), so ``--build`` runs
``docker compose build dashboard`` afterwards; without docker, run
``npm run build`` in ``services/evaluation/dashboard``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "services" / "evaluation"
ROUND_MANIFESTS = REPO_ROOT / "experiments" / "w12-integration" / "results"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(EVAL / "scripts" / script), *args]
    print("$ " + " ".join([Path(cmd[1]).name, *args]), flush=True)
    # P5's reports print Greek letters; a cp1252 Windows console would crash on them
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run(cmd, cwd=str(EVAL), check=True, env=env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="feed round results into the evaluation dashboard")
    ap.add_argument("--rounds", type=Path, nargs="*", default=None,
                    help="four-seam round manifests (default: experiments/w12-integration/results/round*_manifest.json)")
    ap.add_argument("--edge-rounds", type=Path, nargs="*", default=None,
                    help="edge round_manifest.json files for personalization + lineage "
                         "(default: evaluation's vendored rounds)")
    ap.add_argument("--build", action="store_true", help="docker compose build dashboard afterwards")
    args = ap.parse_args(argv)

    rounds = args.rounds or sorted(ROUND_MANIFESTS.glob("round*_manifest.json"))
    if not rounds:
        raise SystemExit(f"no round manifests under {ROUND_MANIFESTS}: run scripts/demo_round.py first")
    run("export_round_feed.py", *map(str, (p.resolve() for p in rounds)))
    edge = [str(p.resolve()) for p in (args.edge_rounds or [])]
    run("build_personalization_report.py", *edge)
    run("export_lineage.py", *edge)
    if args.build:
        subprocess.run(["docker", "compose", "build", "dashboard"], cwd=str(REPO_ROOT), check=True)
        print("rebuilt: docker compose --profile demo up -d dashboard  -> http://<laptop A>:8005")
    else:
        print("next: docker compose build dashboard (or npm run build in services/evaluation/dashboard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
