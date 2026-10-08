"""The narrated demos must keep telling the truth: run them end to end.

Both demos assert their own claims, so completing without an exception is the
check; the output assertions only pin that the key results were printed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_demo_phase2_runs_and_its_claims_hold(capsys):
    from cluster import demo_phase2

    facts = demo_phase2.run(seed=42, verbose=True)
    out = capsys.readouterr().out
    assert facts["moved"] == {demo_phase2.WRONG: [demo_phase2.WEB, demo_phase2.SCI]}
    assert facts["isolation_violations"] == [] and facts["reproducible"] is True
    assert "static_fallback" in out and "PHASE II DEMO OK" in out
    assert "blocked by P3" in out  # the demo states what it does not cover


def test_demo_phase2_is_deterministic_across_runs():
    from cluster import demo_phase2

    a = demo_phase2.run(seed=5, verbose=False)
    b = demo_phase2.run(seed=5, verbose=False)
    assert a["digest"] == b["digest"]


def test_demo_all_weeks_runs_end_to_end(capsys):
    from cluster import demo_all_weeks

    demo_all_weeks.main()
    out = capsys.readouterr().out
    assert "Progress Demonstration" in out
    assert "Status: pipeline demonstrated; report/evidence deliverables present" in out
    assert "3-client federated aggregation round (cluster.demo)" in out
    assert "PHASE II DEMO OK" in out  # multi-cluster block ran and passed its own assertions
    assert "[MISSING]" not in out
    assert "blocked by other modules" in out  # G1 / DP are stated as not demonstrated
    assert "alpha=16" in out and "alpha=32" not in out  # Week 6: matches Edge's contract
    assert "line 51" not in out and "lines 43-45" not in out  # no stale line citations


def test_demo_terminal_output_names_no_week_numbers(capsys):
    """The narrated terminal output describes what runs, not a week of the plan.

    ``W0`` (the frozen base weight in the LoRA formula) is legitimate, so the
    pattern only matches week labels: "Week 3", "Wk 7-13", "W2 Tue", "(W4)".
    """
    import re

    from cluster import demo, demo_all_weeks, demo_phase2

    demo.run_demo(verbose=True)
    demo_phase2.run(seed=42, verbose=True)
    demo_all_weeks.main()
    out = capsys.readouterr().out
    week_label = re.compile(
        r"\bweeks?\s*\d|\bwk\s*\d|\bW\d+\s+(?:mon|tue|wed|thu|fri)\b|\(W\d", re.IGNORECASE
    )
    hits = [ln for ln in out.splitlines() if week_label.search(ln)]
    assert not hits, hits[:3]
