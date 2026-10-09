"""The compose demo-seed job, run against a live registry on a real socket."""
from __future__ import annotations

import json

from registry.demo import Registry, main, run_demo


def test_demo_walks_every_capability_against_a_live_registry(running):
    port = running({})
    # Small NFR adapter keeps CI fast; the full-size run is the compose job's.
    report = run_demo(Registry(f"http://127.0.0.1:{port}"), run_id="t1",
                      nfr_layers=4, nfr_hidden=256)
    steps = [s["step"] for s in report["steps"]]
    assert steps == ["healthz", "save", "compose", "promote", "rollback", "restore_nfr",
                     "lineage", "audit_trail", "gc_dry_run"]
    assert report["ok"] is True
    audit = next(s for s in report["steps"] if s["step"] == "audit_trail")
    assert audit["decisions"] == ["promote", "rollback"]


def test_demo_cli_writes_report_and_is_rerunnable(running, tmp_path):
    port = running({})
    out = tmp_path / "report.json"
    url = f"http://127.0.0.1:{port}"
    assert main(["--url", url, "--report", str(out), "--run-id", "a", "--wait", "5"]) == 0
    assert json.loads(out.read_text())["ok"] is True
    assert main(["--url", url, "--run-id", "b", "--wait", "5"]) == 0  # no collisions


def test_demo_cli_fails_cleanly_when_registry_is_down():
    from .conftest import free_port

    assert main(["--url", f"http://127.0.0.1:{free_port()}", "--wait", "0"]) == 1
