"""Tests for the shared utility layer."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from utils.config import (
    LoggingConfig,
    build_dataclass,
    load_config,
    load_logging_config,
    load_yaml,
    resolve_path,
)
from utils.errors import ClaspP5Error, ConfigError
from utils.io_utils import (
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from utils.logging_utils import configure_logging, get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.text_utils import count_code_lines, count_lines, normalise_newlines, truncate
from utils.timing import Stopwatch, file_timestamp, utc_timestamp


class TestPaths:
    def test_root_contains_expected_directories(self) -> None:
        paths = project_paths()
        assert paths.configs.is_dir()
        assert paths.schemas.is_dir()

    def test_relative_renders_inside_root(self) -> None:
        paths = project_paths()
        assert paths.relative(paths.configs / "logging.yaml") == "configs/logging.yaml"

    def test_relative_passes_through_outside_paths(self, tmp_path: Path) -> None:
        # Compare against the resolved form: relative() resolves, and on macOS
        # /tmp is a symlink to /private/tmp.
        outside = tmp_path / "elsewhere"
        assert project_paths().relative(outside) == str(outside.resolve())

    def test_is_cached(self) -> None:
        assert project_paths() is project_paths()


class TestIoUtils:
    def test_atomic_write_creates_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "file.txt"
        atomic_write_text(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_atomic_write_leaves_no_temp_files(self, tmp_path: Path) -> None:
        atomic_write_text(tmp_path / "a.txt", "x")
        assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]

    def test_json_round_trip(self, tmp_path: Path) -> None:
        payload = {"b": 2, "a": [1, 2, 3], "nested": {"k": None}}
        path = write_json(tmp_path / "x.json", payload)
        assert read_json(path) == payload

    def test_write_json_serialises_paths(self, tmp_path: Path) -> None:
        path = write_json(tmp_path / "p.json", {"where": Path("/tmp/x")})
        assert read_json(path)["where"] == "/tmp/x"

    def test_read_json_missing_raises_clasp_error(self, tmp_path: Path) -> None:
        with pytest.raises(ClaspP5Error, match="not found"):
            read_json(tmp_path / "absent.json")

    def test_read_json_malformed_raises_with_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ClaspP5Error, match="Malformed JSON"):
            read_json(bad)

    def test_jsonl_round_trip(self, tmp_path: Path) -> None:
        rows = [{"i": i} for i in range(5)]
        path = tmp_path / "x.jsonl"
        assert write_jsonl(path, rows) == 5
        assert list(read_jsonl(path)) == rows

    def test_read_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_text('{"a": 1}\n\n\n{"a": 2}\n', encoding="utf-8")
        assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]

    def test_read_jsonl_reports_bad_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_text('{"a": 1}\nnot-json\n', encoding="utf-8")
        with pytest.raises(ClaspP5Error, match=r":2:"):
            list(read_jsonl(path))

    def test_sha256_text_is_stable(self) -> None:
        assert sha256_text("abc") == sha256_text("abc")
        assert sha256_text("abc") != sha256_text("abd")

    def test_sha256_file_matches_text(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("payload", encoding="utf-8")
        assert sha256_file(path) == sha256_text("payload")


class TestTextUtils:
    def test_normalise_newlines_handles_crlf_and_cr(self) -> None:
        assert normalise_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_count_lines(self) -> None:
        assert count_lines("") == 0
        assert count_lines("a\n") == 1
        assert count_lines("a\nb") == 2

    def test_count_code_lines_ignores_blanks_and_comments(self) -> None:
        source = "import os\n\n# a comment\n\ndef f():\n    return 1\n"
        assert count_code_lines(source) == 3

    def test_truncate_collapses_whitespace(self) -> None:
        assert truncate("a   b\n c", limit=100) == "a b c"

    def test_truncate_respects_limit(self) -> None:
        assert len(truncate("x" * 500, limit=20)) == 20


class TestTiming:
    def test_timestamps_have_expected_shape(self) -> None:
        assert utc_timestamp().endswith("Z")
        assert len(utc_timestamp()) == 20
        assert len(file_timestamp()) == 16

    def test_stopwatch_measures_a_block(self) -> None:
        with Stopwatch("x") as watch:
            pass
        assert watch.elapsed_seconds >= 0.0

    def test_stopwatch_is_live_during_the_block(self) -> None:
        with Stopwatch() as watch:
            assert watch.elapsed_seconds >= 0.0


@dataclass(frozen=True)
class _Inner:
    value: int = 1


@dataclass(frozen=True)
class _Outer:
    name: str
    where: Path
    inner: _Inner = field(default_factory=_Inner)
    items: list[str] = field(default_factory=list)
    optional: str | None = None


class TestConfig:
    def test_build_dataclass_hydrates_nested_types(self) -> None:
        built = build_dataclass(
            _Outer,
            {"name": "n", "where": "configs", "inner": {"value": 7}, "items": ["a", "b"]},
        )
        assert built.inner.value == 7
        assert built.items == ["a", "b"]
        assert built.where.is_absolute()

    def test_build_dataclass_rejects_unknown_keys(self) -> None:
        with pytest.raises(ConfigError, match="Unknown key"):
            build_dataclass(_Outer, {"name": "n", "where": ".", "typo": 1})

    def test_build_dataclass_accepts_none_for_optional(self) -> None:
        built = build_dataclass(_Outer, {"name": "n", "where": ".", "optional": None})
        assert built.optional is None

    def test_build_dataclass_rejects_wrong_scalar_type(self) -> None:
        with pytest.raises(ConfigError, match="must be int"):
            build_dataclass(_Outer, {"name": "n", "where": ".", "inner": {"value": "abc"}})

    def test_load_yaml_expands_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLASP_TEST_VAR", "expanded")
        path = tmp_path / "c.yaml"
        path.write_text("key: ${CLASP_TEST_VAR}\n", encoding="utf-8")
        assert load_yaml(path)["key"] == "expanded"

    def test_load_yaml_uses_env_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLASP_UNSET_VAR", raising=False)
        path = tmp_path / "c.yaml"
        path.write_text("key: ${CLASP_UNSET_VAR:fallback}\n", encoding="utf-8")
        assert load_yaml(path)["key"] == "fallback"

    def test_load_yaml_raises_on_unset_env_without_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLASP_UNSET_VAR", raising=False)
        path = tmp_path / "c.yaml"
        path.write_text("key: ${CLASP_UNSET_VAR}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not set"):
            load_yaml(path)

    def test_load_yaml_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(tmp_path / "absent.yaml")

    def test_load_config_missing_section(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("other: {}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Section 'wanted' missing"):
            load_config(_Inner, path, section="wanted")

    def test_resolve_path_makes_relative_absolute(self) -> None:
        assert resolve_path("configs") == project_paths().root / "configs"

    def test_resolve_path_preserves_absolute(self) -> None:
        assert resolve_path("/tmp/x") == Path("/tmp/x")

    def test_real_logging_config_loads(self) -> None:
        assert isinstance(load_logging_config(), LoggingConfig)


class TestLogging:
    def test_get_logger_namespaces(self) -> None:
        assert get_logger("corpus.x").name == "clasp_p5.corpus.x"

    def test_get_logger_does_not_double_prefix(self) -> None:
        assert get_logger("clasp_p5.a").name == "clasp_p5.a"

    def test_configure_writes_to_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "run.log"
        configure_logging(level="INFO", log_file=log_file, force=True)
        get_logger("test.sink").info("marker-line")
        logging.getLogger("clasp_p5").handlers[-1].flush()
        assert "marker-line" in log_file.read_text(encoding="utf-8")
        configure_logging(level="CRITICAL", log_file=None, force=True)

    def test_configure_replaces_handlers(self, tmp_path: Path) -> None:
        configure_logging(level="INFO", log_file=tmp_path / "a.log", force=True)
        first = len(logging.getLogger("clasp_p5").handlers)
        configure_logging(level="INFO", log_file=tmp_path / "b.log", force=True)
        assert len(logging.getLogger("clasp_p5").handlers) == first
        configure_logging(level="CRITICAL", log_file=None, force=True)


class TestReporting:
    def test_report_renders_all_blocks(self) -> None:
        text = (
            MarkdownReport("Title", subtitle="sub")
            .heading("Section")
            .paragraph("body")
            .bullets(["one", "two"])
            .key_values({"k": 1, "flag": True, "none": None})
            .table(["A", "B"], [[1, 2]])
            .code("x = 1", "python")
            .status_line(True, "all good")
            .rule()
            .render()
        )
        assert text.startswith("# Title")
        assert "| A | B |" in text
        assert "```python" in text
        assert "**PASS**" in text
        assert "- **flag:** yes" in text

    def test_table_escapes_pipes(self) -> None:
        text = MarkdownReport("T").table(["A"], [["a|b"]]).render()
        assert "a\\|b" in text

    def test_empty_table_renders_placeholder(self) -> None:
        assert "_No rows._" in MarkdownReport("T").table(["A"], []).render()

    def test_write_creates_file(self, tmp_path: Path) -> None:
        path = MarkdownReport("T").paragraph("x").write(tmp_path / "out" / "r.md")
        assert path.read_text(encoding="utf-8").startswith("# T")


class TestJsonSerialisation:
    def test_unserialisable_type_raises_type_error(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="not JSON serialisable"):
            write_json(tmp_path / "x.json", {"bad": object()})

    def test_sets_serialise_as_lists(self, tmp_path: Path) -> None:
        path = write_json(tmp_path / "x.json", {"s": {1, 2}})
        assert sorted(json.loads(path.read_text(encoding="utf-8"))["s"]) == [1, 2]
