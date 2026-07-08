"""Run-manifest utilities (P4, D9) — the reproducibility record for a run.

Every experiment records a RunManifest *before* it runs (GPU-hrs estimated
first, D8). The config hash pins the exact resolved config so a run is
reproducible; ``gpu_hours_actual`` is filled in on completion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from contracts import RunManifest


def config_hash(config: dict) -> str:
    """Deterministic sha256 over a resolved experiment config (order-insensitive)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    run_id: str,
    seed: int,
    config: dict,
    adapter_versions: dict[str, int],
    gpu_hours_estimate: float,
    notes: str = "",
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        seed=seed,
        config_hash=config_hash(config),
        adapter_versions=dict(adapter_versions),
        gpu_hours_estimate=gpu_hours_estimate,
        notes=notes,
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return path


def read_manifest(path: str | Path) -> RunManifest:
    d = json.loads(Path(path).read_text())
    return RunManifest(**d)
