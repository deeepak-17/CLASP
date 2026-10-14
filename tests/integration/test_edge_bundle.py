"""The edge laptop's setup steps (scripts/make_edge_bundle.py) say what actually works."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_edge_bundle.py"


def _bundle():
    spec = importlib.util.spec_from_file_location("make_edge_bundle", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_fine_tune_steps_install_dp_and_train_like_the_page():
    steps = _bundle().run_steps("client-numpy", "scientific", "http://10.0.0.2:8002",
                                ["round1"], certs=False, corpus=True)
    assert "-e security" in steps.split("## Optional")[1]  # --dp needs P3's library
    train = next(line for line in steps.splitlines() if "edge.train_client" in line)
    assert "--dp" in train and "--seq-len 512" in train
    assert "artifacts/live-" in train  # a set the page lists as trained here
