"""Tests for eval_harness/line_guard.py: the frozen 300-example line-completion guard set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.completion import examples_fingerprint
from evaluation.eval_harness import line_guard


def _fake_repo(root: Path, name: str, *, files: int, lines: int) -> Path:
    """``root/name/repo/pkg/mod<i>.py`` with ``lines`` usable lines each; returns ``root/name``."""
    repo = root / name / "repo" / "pkg"
    repo.mkdir(parents=True)
    for i in range(files):
        body = "\n".join(f"value_{i}_{n} = compute({n}, {i})" for n in range(lines))
        (repo / f"mod{i}.py").write_text(body + "\n", encoding="utf-8")
    return root / name


def _sources(tmp_path: Path):
    return [_fake_repo(tmp_path, n, files=4, lines=200) for n in line_guard.GUARD_SOURCES]


def test_build_is_deterministic_and_balanced(tmp_path):
    srcs = _sources(tmp_path)
    a = line_guard.build_guard_set(srcs, size=30, seed=0)
    assert a == line_guard.build_guard_set(srcs, size=30, seed=0)
    assert len(a) == 30
    assert Counter(e.file.split("/")[0] for e in a) == {s: 10 for s in line_guard.GUARD_SOURCES}
    assert a != line_guard.build_guard_set(srcs, size=30, seed=1)


def test_remainder_goes_to_the_earliest_sources(tmp_path):
    a = line_guard.build_guard_set(_sources(tmp_path), size=31, seed=0)
    counts = Counter(e.file.split("/")[0] for e in a)
    assert [counts[s] for s in line_guard.GUARD_SOURCES] == [11, 10, 10]


def test_short_source_is_topped_up_from_the_others(tmp_path):
    srcs = [_fake_repo(tmp_path, "client-click", files=1, lines=40),
            _fake_repo(tmp_path, "client-jinja", files=4, lines=200),
            _fake_repo(tmp_path, "client-colorama", files=4, lines=200)]
    # the first source yields only a handful (stride 7 over one 40-line file)
    a = line_guard.build_guard_set(srcs, size=30, seed=0)
    counts = Counter(e.file.split("/")[0] for e in a)
    assert len(a) == 30
    assert counts["client-click"] < 10
    assert counts["client-jinja"] + counts["client-colorama"] == 30 - counts["client-click"]
    assert abs(counts["client-jinja"] - counts["client-colorama"]) <= 1


def test_save_and_load_round_trip(tmp_path):
    a = line_guard.build_guard_set(_sources(tmp_path), size=30, seed=0)
    path = tmp_path / "g.jsonl"
    line_guard.save_guard_set(a, path)
    assert line_guard.load_guard_set(path) == a
    assert set(json.loads(path.read_text().splitlines()[0])) == {"file", "line_no", "prompt", "target"}


def test_committed_guard_set_matches_its_manifest():
    ex = line_guard.load_guard_set()
    man = json.loads(line_guard.GUARD_FILE.with_suffix(".manifest.json").read_text())
    assert len(ex) == 300 and examples_fingerprint(ex) == man["fingerprint"]
    assert {e.file.split("/")[0] for e in ex} == set(line_guard.GUARD_SOURCES)
    assert {s["license"] for s in man["sources"].values()} == {"BSD-3-Clause"}


def test_guard_metrics_shape():
    m = line_guard.guard_metrics({"edit_similarity": 0.5, "exact_match": 0.25, "n_examples": 300}, 300)
    assert m == {"benchmark": "LineCompletion", "edit_similarity": 0.5, "exact_match": 0.25,
                 "n_examples": 300}


def test_paired_noise_is_zero_for_identical_rows_and_positive_otherwise():
    assert line_guard.paired_noise([0.5] * 50, [0.5] * 50) == 0.0
    assert line_guard.paired_noise([0.0, 1.0] * 25, [1.0, 0.0] * 25) > 0.1
    a, b = [0.1, 0.9, 0.4, 0.7] * 10, [0.2, 0.5, 0.4, 0.9] * 10
    assert line_guard.paired_noise(a, b) == line_guard.paired_noise(a, b)
