"""Tests for the Week-1 corpus survey, filtering and collection pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from corpus.acquisition import LocalAcquirer, SyntheticAcquirer, build_acquirer
from corpus.collector import CorpusCollector, load_corpus_config, load_corpus_records
from corpus.filters import FileSelector, SkipReason, matches_any
from corpus.models import (
    AcquisitionConfig,
    CorpusConfig,
    FilterConfig,
    SourceConfig,
    SyntheticConfig,
)
from corpus.survey import load_survey_config, render_survey_report, score_survey
from utils.errors import ConfigError, CorpusError


# ---------------------------------------------------------------------------
# Survey (Week 1 · Monday)
# ---------------------------------------------------------------------------
class TestSurvey:
    def test_repository_survey_config_is_valid(self) -> None:
        config = load_survey_config()
        assert config.candidates
        assert abs(sum(c.weight for c in config.criteria) - 1.0) < 1e-9

    def test_scoring_ranks_candidates(self) -> None:
        outcome = score_survey(load_survey_config())
        scores = [item.weighted_score for item in outcome.ranking]
        assert scores == sorted(scores, reverse=True)
        assert outcome.ranking[0].rank == 1

    def test_selected_candidate_is_flagged(self) -> None:
        outcome = score_survey(load_survey_config())
        assert outcome.selected.is_selected
        assert sum(1 for item in outcome.ranking if item.is_selected) == 1

    def test_repository_selection_is_the_top_scorer(self) -> None:
        outcome = score_survey(load_survey_config())
        assert outcome.selection_is_top_scorer
        assert not outcome.warnings

    def test_override_produces_a_warning(self) -> None:
        config = load_survey_config()
        runner_up = score_survey(config).ranking[1].key
        outcome = score_survey(replace(config, selected=runner_up))
        assert not outcome.selection_is_top_scorer
        assert outcome.warnings

    def test_weights_must_sum_to_one(self) -> None:
        config = load_survey_config()
        broken = [replace(config.criteria[0], weight=0.99), *config.criteria[1:]]
        with pytest.raises(ConfigError, match="sum to 1.0"):
            replace(config, criteria=broken)

    def test_selected_must_be_a_declared_candidate(self) -> None:
        with pytest.raises(ConfigError, match="not among the declared candidates"):
            replace(load_survey_config(), selected="does-not-exist")

    def test_report_names_the_selection(self) -> None:
        outcome = score_survey(load_survey_config())
        text = render_survey_report(outcome).render()
        assert outcome.selected.candidate.name in text
        assert "Known limitations" in text


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
class TestGlobMatching:
    @pytest.mark.parametrize(
        ("path", "pattern", "expected"),
        [
            ("src/pkg/tests/test_a.py", "**/tests/**", True),
            ("src/pkg/mod.py", "**/tests/**", False),
            ("setup.py", "**/setup.py", True),
            ("src/a/setup.py", "**/setup.py", True),
            ("src/a/mod_test.py", "**/*_test.py", True),
            ("src/a/mod.py", "**/*_test.py", False),
        ],
    )
    def test_matches_any(self, path: str, pattern: str, expected: bool) -> None:
        assert matches_any(path, [pattern]) is expected

    def test_empty_pattern_list_never_matches(self) -> None:
        assert matches_any("a/b.py", []) is False


class TestFileSelector:
    @pytest.fixture
    def selector(self) -> FileSelector:
        return FileSelector(
            FilterConfig(
                include_globs=["**/*.py"],
                exclude_globs=["**/tests/**"],
                min_code_lines=3,
                max_file_bytes=1000,
                drop_duplicate_content=True,
            )
        )

    @staticmethod
    def _write(tmp_path: Path, name: str, body: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def test_accepts_a_normal_file(self, selector: FileSelector, tmp_path: Path) -> None:
        path = self._write(tmp_path, "a.py", "x = 1\ny = 2\nz = 3\n")
        outcome = selector.evaluate(path, "a.py")
        assert not isinstance(outcome, SkipReason)
        assert outcome.num_code_lines == 3

    def test_rejects_excluded_glob(self, selector: FileSelector, tmp_path: Path) -> None:
        path = self._write(tmp_path, "tests/t.py", "x = 1\ny = 2\nz = 3\n")
        assert selector.evaluate(path, "tests/t.py") is SkipReason.EXCLUDED_GLOB

    def test_rejects_empty_file(self, selector: FileSelector, tmp_path: Path) -> None:
        path = self._write(tmp_path, "e.py", "")
        assert selector.evaluate(path, "e.py") is SkipReason.EMPTY

    def test_rejects_oversized_file(self, selector: FileSelector, tmp_path: Path) -> None:
        path = self._write(tmp_path, "big.py", "x = 1\n" * 500)
        assert selector.evaluate(path, "big.py") is SkipReason.TOO_LARGE

    def test_rejects_thin_file(self, selector: FileSelector, tmp_path: Path) -> None:
        path = self._write(tmp_path, "thin.py", "# only a comment\n\nx = 1\n")
        assert selector.evaluate(path, "thin.py") is SkipReason.TOO_FEW_CODE_LINES

    def test_rejects_undecodable_file(self, selector: FileSelector, tmp_path: Path) -> None:
        path = tmp_path / "bin.py"
        path.write_bytes(b"\x00\x01\x02binary")
        assert selector.evaluate(path, "bin.py") is SkipReason.UNDECODABLE

    def test_deduplicates_identical_content(self, selector: FileSelector, tmp_path: Path) -> None:
        body = "a = 1\nb = 2\nc = 3\n"
        first = self._write(tmp_path, "one.py", body)
        second = self._write(tmp_path, "two.py", body)
        assert not isinstance(selector.evaluate(first, "one.py"), SkipReason)
        assert selector.evaluate(second, "two.py") is SkipReason.DUPLICATE_CONTENT
        assert selector.duplicates_dropped == 1

    def test_crlf_and_lf_variants_deduplicate(self, tmp_path: Path) -> None:
        selector = FileSelector(FilterConfig(min_code_lines=1, drop_duplicate_content=True))
        (tmp_path / "lf.py").write_bytes(b"a = 1\nb = 2\n")
        (tmp_path / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
        assert not isinstance(selector.evaluate(tmp_path / "lf.py", "lf.py"), SkipReason)
        assert selector.evaluate(tmp_path / "crlf.py", "crlf.py") is SkipReason.DUPLICATE_CONTENT

    def test_reset_clears_per_source_but_not_totals(
        self, selector: FileSelector, tmp_path: Path
    ) -> None:
        selector.evaluate(self._write(tmp_path, "e.py", ""), "e.py")
        assert selector.skip_counts()
        selector.reset_skip_counts()
        assert not selector.skip_counts()
        assert selector.total_skip_counts()

    def test_discover_is_sorted_and_recursive(self, selector: FileSelector, tmp_path: Path) -> None:
        for name in ("z.py", "a.py", "sub/m.py"):
            self._write(tmp_path, name, "x = 1\n")
        found = selector.discover(tmp_path)
        assert [p.name for p in found] == sorted(p.name for p in found)
        assert len(found) == 3

    def test_discover_honours_subpaths(self, selector: FileSelector, tmp_path: Path) -> None:
        self._write(tmp_path, "src/in.py", "x = 1\n")
        self._write(tmp_path, "other/out.py", "x = 1\n")
        found = selector.discover(tmp_path, ["src"])
        assert [p.name for p in found] == ["in.py"]

    def test_missing_subpath_falls_back_to_root(self, selector: FileSelector, tmp_path: Path) -> None:
        self._write(tmp_path, "a.py", "x = 1\n")
        assert len(selector.discover(tmp_path, ["does/not/exist"])) == 1


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
class TestAcquisition:
    def test_synthetic_generates_deterministic_files(self, tmp_path: Path) -> None:
        acquirer = SyntheticAcquirer(SyntheticConfig(files_per_source=4, seed=7))
        source = SourceConfig(cluster_id="alpha", project_label="Alpha", license="MIT")

        first = acquirer.acquire(source, tmp_path / "a")
        second = acquirer.acquire(source, tmp_path / "b")

        files_a = sorted(p.read_text(encoding="utf-8") for p in first.root.rglob("*.py"))
        files_b = sorted(p.read_text(encoding="utf-8") for p in second.root.rglob("*.py"))
        assert len(files_a) == 4
        assert files_a == files_b
        assert first.synthetic is True

    def test_synthetic_differs_between_clusters(self, tmp_path: Path) -> None:
        acquirer = SyntheticAcquirer(SyntheticConfig(files_per_source=2, seed=7))
        a = acquirer.acquire(SourceConfig(cluster_id="alpha", project_label="A", license="MIT"), tmp_path)
        b = acquirer.acquire(SourceConfig(cluster_id="beta", project_label="B", license="MIT"), tmp_path)
        text_a = (a.root / "alpha" / "module_00.py").read_text(encoding="utf-8")
        text_b = (b.root / "beta" / "module_00.py").read_text(encoding="utf-8")
        assert text_a != text_b

    def test_local_requires_an_existing_checkout(self, tmp_path: Path) -> None:
        source = SourceConfig(cluster_id="alpha", project_label="A", license="MIT")
        with pytest.raises(CorpusError, match="no checkout found"):
            LocalAcquirer().acquire(source, tmp_path)

    def test_local_accepts_an_existing_checkout(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        source = SourceConfig(cluster_id="alpha", project_label="A", license="MIT")
        assert LocalAcquirer().acquire(source, tmp_path).root == tmp_path / "alpha"

    def test_factory_returns_matching_strategy(self) -> None:
        acquirer = build_acquirer(AcquisitionConfig(mode="synthetic"), SyntheticConfig())
        assert isinstance(acquirer, SyntheticAcquirer)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestCorpusConfig:
    def test_repository_config_loads(self) -> None:
        config = load_corpus_config()
        assert config.name == "D1"
        assert len(config.sources) == 6
        assert all(source.license for source in config.sources)

    def test_rejects_bad_acquisition_mode(self) -> None:
        with pytest.raises(ConfigError, match="acquisition.mode"):
            AcquisitionConfig(mode="carrier-pigeon")

    def test_rejects_uppercase_cluster_id(self) -> None:
        with pytest.raises(ConfigError, match="lowercase"):
            SourceConfig(cluster_id="Alpha", project_label="A", license="MIT")

    def test_rejects_missing_license(self) -> None:
        with pytest.raises(ConfigError, match="must declare a license"):
            SourceConfig(cluster_id="alpha", project_label="A", license="")

    def test_rejects_duplicate_cluster_ids(self, synthetic_corpus_config: CorpusConfig) -> None:
        duplicated = [synthetic_corpus_config.sources[0], synthetic_corpus_config.sources[0]]
        with pytest.raises(ConfigError, match="Duplicate cluster_id"):
            replace(synthetic_corpus_config, sources=duplicated)

    def test_git_mode_requires_repo_urls(self, synthetic_corpus_config: CorpusConfig) -> None:
        with pytest.raises(ConfigError, match="no repo_url"):
            replace(
                synthetic_corpus_config,
                acquisition=AcquisitionConfig(mode="git"),
            )

    def test_rejects_empty_include_globs(self) -> None:
        with pytest.raises(ConfigError, match="include_globs"):
            FilterConfig(include_globs=[])


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class TestCollector:
    def test_collects_a_synthetic_corpus(self, synthetic_corpus_config: CorpusConfig) -> None:
        result = CorpusCollector(synthetic_corpus_config).collect()
        manifest = result.manifest

        assert manifest.total_files == 18  # 3 sources x 6 files
        assert manifest.num_clusters == 3
        assert manifest.synthetic is True
        assert result.corpus_path.is_file()
        assert result.manifest_path.is_file()

    def test_manifest_totals_match_records(self, synthetic_corpus_config: CorpusConfig) -> None:
        result = CorpusCollector(synthetic_corpus_config).collect()
        records = load_corpus_records(result.corpus_path)
        assert len(records) == result.manifest.total_files
        assert sum(r.num_code_lines for r in records) == result.manifest.total_code_lines

    def test_is_reproducible(self, synthetic_corpus_config: CorpusConfig, tmp_path: Path) -> None:
        first = CorpusCollector(synthetic_corpus_config).collect()
        digest_one = first.manifest.corpus_sha256

        second_config = replace(
            synthetic_corpus_config,
            output=replace(
                synthetic_corpus_config.output,
                raw_dir=tmp_path / "raw2",
                corpus_path=tmp_path / "processed2" / "corpus.jsonl",
                manifest_path=tmp_path / "metadata2" / "m.json",
            ),
        )
        second = CorpusCollector(second_config).collect()

        # Absolute paths differ between runs, so compare record content rather
        # than the file digest.
        one = [r.content_sha256 for r in load_corpus_records(first.corpus_path)]
        two = [r.content_sha256 for r in load_corpus_records(second.corpus_path)]
        assert one == two
        assert len(digest_one) == 64

    def test_dry_run_writes_nothing(self, synthetic_corpus_config: CorpusConfig) -> None:
        result = CorpusCollector(synthetic_corpus_config).collect(dry_run=True)
        assert result.manifest.total_files == 18
        assert not result.corpus_path.exists()
        assert not result.manifest_path.exists()

    def test_index_only_mode_omits_content(self, synthetic_corpus_config: CorpusConfig) -> None:
        config = replace(
            synthetic_corpus_config,
            output=replace(synthetic_corpus_config.output, include_content=False),
        )
        result = CorpusCollector(config).collect()
        assert all(r.content is None for r in load_corpus_records(result.corpus_path))

    def test_records_carry_cluster_and_licence(self, synthetic_corpus_config: CorpusConfig) -> None:
        result = CorpusCollector(synthetic_corpus_config).collect()
        records = load_corpus_records(result.corpus_path)
        assert {r.cluster_id for r in records} == {"alpha", "beta", "gamma"}
        assert all(r.license for r in records)
        assert all(r.file_id.startswith(r.cluster_id + ":") for r in records)

    def test_rejects_a_filter_that_drops_everything(
        self, synthetic_corpus_config: CorpusConfig
    ) -> None:
        config = replace(
            synthetic_corpus_config,
            filters=replace(synthetic_corpus_config.filters, min_code_lines=100_000),
        )
        with pytest.raises(CorpusError, match="zero records|no files"):
            CorpusCollector(config).collect()

    def test_per_source_stats_are_recorded(self, synthetic_corpus_config: CorpusConfig) -> None:
        manifest = CorpusCollector(synthetic_corpus_config).collect().manifest
        assert len(manifest.sources) == 3
        assert all(source.files_kept > 0 for source in manifest.sources)
        assert sum(source.files_kept for source in manifest.sources) == manifest.total_files

    def test_load_corpus_records_rejects_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(CorpusError, match="no records"):
            load_corpus_records(empty)
