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
    assert steps == ["healthz", "save", "promote_v1", "compose", "promote_v2", "reject",
                     "restore_nfr", "lineage", "audit_trail", "gc_dry_run"]
    assert report["ok"] is True
    audit = next(s for s in report["steps"] if s["step"] == "audit_trail")
    assert audit["decisions"] == ["promote", "promote", "reject"]


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


def test_demo_signs_its_promotes_when_the_registry_checks_signatures(running, tmp_path,
                                                                     monkeypatch):
    """The compose stack gives the registry the evaluation-signing key, and the
    demo-seed job the same key: its promotes must carry valid signatures."""
    key = tmp_path / "eval_hmac.key"
    key.write_text("5e" * 32 + "\n", encoding="utf-8")
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET_FILE", str(key))   # the registry's and the job's
    port = running({})
    report = run_demo(Registry(f"http://127.0.0.1:{port}"), run_id="signed",
                      nfr_layers=4, nfr_hidden=256)
    assert report["ok"] is True
    audit = next(s for s in report["steps"] if s["step"] == "audit_trail")
    assert audit["decisions"] == ["promote", "promote", "reject"]


def test_without_the_key_the_demo_cannot_promote_on_a_signing_registry(running, tmp_path,
                                                                       monkeypatch):
    """The refusal the compose job hit: proof the registry really checks."""
    import pytest

    from registry.demo import DemoFailure

    key = tmp_path / "eval_hmac.key"
    key.write_text("5e" * 32 + "\n", encoding="utf-8")
    monkeypatch.setenv("CLASP_EVAL_HMAC_SECRET_FILE", str(key))
    port = running({})
    monkeypatch.setattr("registry.demo.eval_secret", lambda: None)   # the job has no key
    with pytest.raises(DemoFailure, match="401"):
        run_demo(Registry(f"http://127.0.0.1:{port}"), run_id="unsigned",
                 nfr_layers=4, nfr_hidden=256)
