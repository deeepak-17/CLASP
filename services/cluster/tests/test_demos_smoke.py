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


def test_demo_terminal_output_names_no_week_numbers(capsys):
    """The narrated terminal output describes what runs, not a week of the plan.

    ``W0`` (the frozen base weight in the LoRA formula) is legitimate, so the
    pattern only matches week labels: "Week 3", "Wk 7-13", "W2 Tue", "(W4)".
    """
    import re

    from cluster import demo, demo_phase2

    demo.run_demo(verbose=True)
    demo_phase2.run(seed=42, verbose=True)
    out = capsys.readouterr().out
    week_label = re.compile(
        r"\bweeks?\s*\d|\bwk\s*\d|\bW\d+\s+(?:mon|tue|wed|thu|fri)\b|\(W\d", re.IGNORECASE
    )
    hits = [ln for ln in out.splitlines() if week_label.search(ln)]
    assert not hits, hits[:3]
