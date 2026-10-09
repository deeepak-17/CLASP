"""Tests for eval_harness/archive.py — checksummed results inventory."""

from __future__ import annotations

from pathlib import Path

from evaluation.eval_harness.archive import MANIFEST_NAME, build_manifest, verify_manifest


def _tree(root: Path) -> None:
    (root / "results").mkdir()
    (root / "results" / "a.json").write_text("{}")
    (root / "results" / MANIFEST_NAME).write_text("ignored")
    (root / "reports").mkdir()
    (root / "reports" / "r.md").write_text("# r")


def test_manifest_lists_files_but_not_itself(tmp_path: Path) -> None:
    _tree(tmp_path)
    manifest = build_manifest(tmp_path, ("results", "reports", "absent"))
    assert set(manifest["files"]) == {"results/a.json", "reports/r.md"}
    assert manifest["n_files"] == 2


def test_verify_reports_changed_missing_and_added(tmp_path: Path) -> None:
    _tree(tmp_path)
    manifest = build_manifest(tmp_path, ("results", "reports"))
    assert verify_manifest(manifest, tmp_path) == {"changed": [], "missing": [], "added": []}
    (tmp_path / "results" / "a.json").write_text('{"x": 1}')
    (tmp_path / "reports" / "r.md").unlink()
    (tmp_path / "results" / "b.json").write_text("{}")
    assert verify_manifest(manifest, tmp_path) == {
        "changed": ["results/a.json"],
        "missing": ["reports/r.md"],
        "added": ["results/b.json"],
    }
