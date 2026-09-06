"""Corpus collection — Week 1, Tuesday.

Pipeline
--------
::

    configs/dataset.yaml
        -> acquisition (git | local | synthetic)
        -> per-source file discovery
        -> filtering (globs, size, decodability, SLOC floor, dedup)
        -> normalisation into CorpusRecord
        -> datasets/processed/corpus.jsonl   (the corpus itself)
        -> datasets/metadata/corpus_manifest.json  (provenance + statistics)

The manifest is the important half. It pins the corpus SHA-256 that the
Week-2 partitioner records in its own manifest, giving an unbroken chain from
"these exact source bytes" to "this federated client topology".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from corpus.acquisition import AcquiredSource, SourceAcquirer, build_acquirer
from corpus.filters import AcceptedFile, FileSelector, SkipReason
from corpus.models import (
    LANGUAGE_PYTHON,
    CorpusConfig,
    CorpusManifest,
    CorpusRecord,
    SourceConfig,
    SourceStats,
)
from utils.config import load_config
from utils.errors import CorpusError
from utils.io_utils import ensure_dir, sha256_file, write_json, write_jsonl
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.timing import Stopwatch, utc_timestamp

_LOG = get_logger(__name__)


def load_corpus_config(path: Path | str | None = None) -> CorpusConfig:
    """Load ``configs/dataset.yaml`` into a :class:`CorpusConfig`."""
    target = path or (project_paths().configs / "dataset.yaml")
    return load_config(CorpusConfig, target, section="corpus")


@dataclass(frozen=True)
class CollectionResult:
    """What a collection run produced."""

    manifest: CorpusManifest
    corpus_path: Path
    manifest_path: Path
    elapsed_seconds: float

    @property
    def total_files(self) -> int:
        return self.manifest.total_files


class CorpusCollector:
    """Builds the D1 corpus from a :class:`CorpusConfig`.

    Args:
        config: Parsed ``corpus`` section of ``configs/dataset.yaml``.
        acquirer: Override the acquisition strategy. Injected by the test
            suite; production callers let the factory pick from config.
    """

    def __init__(self, config: CorpusConfig, acquirer: SourceAcquirer | None = None) -> None:
        self._config = config
        self._acquirer = acquirer or build_acquirer(config.acquisition, config.synthetic)
        self._selector = FileSelector(config.filters)

    # --- public API --------------------------------------------------------
    def collect(self, *, dry_run: bool = False) -> CollectionResult:
        """Run the full collection pipeline.

        Args:
            dry_run: Discover and filter, but write nothing to disk. Used to
                validate configuration changes before spending clone time.

        Raises:
            CorpusError: If no source yields any usable file.
        """
        paths = project_paths()
        raw_dir = ensure_dir(self._config.output.raw_dir)

        records: list[CorpusRecord] = []
        stats: list[SourceStats] = []

        with Stopwatch("collect") as watch:
            for source in self._config.sources:
                source_records, source_stats = self._collect_source(source, raw_dir)
                records.extend(source_records)
                stats.append(source_stats)

            if not records:
                raise CorpusError(
                    "Collection produced zero records. Check filters.exclude_globs and "
                    "filters.min_code_lines in configs/dataset.yaml."
                )

            empty_sources = [s.cluster_id for s in stats if s.files_kept == 0]
            if empty_sources:
                raise CorpusError(
                    "Source(s) yielded no files, which would create an empty federated client: "
                    f"{', '.join(empty_sources)}"
                )

            corpus_path = Path(self._config.output.corpus_path)
            manifest_path = Path(self._config.output.manifest_path)

            if dry_run:
                _LOG.info("Dry run: %d record(s) collected, nothing written", len(records))
                manifest = self._build_manifest(records, stats, corpus_path, corpus_sha256="0" * 64)
                return CollectionResult(
                    manifest=manifest,
                    corpus_path=corpus_path,
                    manifest_path=manifest_path,
                    elapsed_seconds=watch.elapsed_seconds,
                )

            ensure_dir(corpus_path.parent)
            written = write_jsonl(
                corpus_path,
                (r.to_dict(include_content=self._config.output.include_content) for r in records),
            )
            corpus_sha256 = sha256_file(corpus_path)
            _LOG.info("Wrote %d record(s) -> %s", written, paths.relative(corpus_path))

            manifest = self._build_manifest(records, stats, corpus_path, corpus_sha256)
            write_json(manifest_path, manifest.to_dict())
            _LOG.info("Wrote manifest -> %s", paths.relative(manifest_path))

        return CollectionResult(
            manifest=manifest,
            corpus_path=corpus_path,
            manifest_path=manifest_path,
            elapsed_seconds=watch.elapsed_seconds,
        )

    # --- internals ---------------------------------------------------------
    def _collect_source(
        self, source: SourceConfig, raw_dir: Path
    ) -> tuple[list[CorpusRecord], SourceStats]:
        """Acquire, discover, filter and normalise one source repository."""
        _LOG.info("Collecting cluster '%s' (%s)", source.cluster_id, source.project_label)
        acquired: AcquiredSource = self._acquirer.acquire(source, raw_dir)

        candidates = self._selector.discover(acquired.root, source.subpaths)
        _LOG.info("  discovered %d candidate file(s)", len(candidates))

        records: list[CorpusRecord] = []
        for path in candidates:
            relative_path = self._relative_path(path, acquired.root)
            outcome = self._selector.evaluate(path, relative_path)
            if isinstance(outcome, SkipReason):
                continue
            records.append(self._to_record(source, acquired, outcome, relative_path))

        skips = self._selector.reset_skip_counts()
        stats = SourceStats(
            cluster_id=source.cluster_id,
            project_label=source.project_label,
            license=source.license,
            source_url=source.repo_url,
            resolved_ref=acquired.resolved_ref,
            files_discovered=len(candidates),
            files_kept=len(records),
            files_skipped=skips,
            num_lines=sum(r.num_lines for r in records),
            num_code_lines=sum(r.num_code_lines for r in records),
            num_bytes=sum(r.num_bytes for r in records),
        )
        _LOG.info(
            "  kept %d/%d file(s), %d code line(s)",
            stats.files_kept,
            stats.files_discovered,
            stats.num_code_lines,
        )
        return records, stats

    @staticmethod
    def _relative_path(path: Path, root: Path) -> str:
        """POSIX-style path of ``path`` relative to its checkout root."""
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:  # pragma: no cover - defensive
            return path.name

    def _to_record(
        self,
        source: SourceConfig,
        acquired: AcquiredSource,
        accepted: AcceptedFile,
        relative_path: str,
    ) -> CorpusRecord:
        return CorpusRecord(
            file_id=f"{source.cluster_id}:{relative_path}",
            cluster_id=source.cluster_id,
            project_label=source.project_label,
            relative_path=relative_path,
            language=LANGUAGE_PYTHON,
            num_lines=accepted.num_lines,
            num_code_lines=accepted.num_code_lines,
            num_bytes=accepted.num_bytes,
            content_sha256=accepted.content_sha256,
            license=source.license,
            source_url=source.repo_url,
            source_ref=acquired.resolved_ref,
            content=accepted.content if self._config.output.include_content else None,
        )

    def _build_manifest(
        self,
        records: list[CorpusRecord],
        stats: list[SourceStats],
        corpus_path: Path,
        corpus_sha256: str,
    ) -> CorpusManifest:
        paths = project_paths()
        filters = self._config.filters
        return CorpusManifest(
            corpus_name=self._config.name,
            corpus_version=self._config.version,
            created_at=utc_timestamp(),
            acquisition_mode=self._config.acquisition.mode,
            synthetic=self._config.acquisition.mode == "synthetic",
            corpus_path=paths.relative(corpus_path),
            corpus_sha256=corpus_sha256,
            total_files=len(records),
            total_lines=sum(r.num_lines for r in records),
            total_code_lines=sum(r.num_code_lines for r in records),
            total_bytes=sum(r.num_bytes for r in records),
            num_clusters=len({r.cluster_id for r in records}),
            include_content=self._config.output.include_content,
            sources=stats,
            duplicates_dropped=self._selector.duplicates_dropped,
            filter_summary={
                "include_globs": list(filters.include_globs),
                "exclude_glob_count": len(filters.exclude_globs),
                "min_code_lines": filters.min_code_lines,
                "max_file_bytes": filters.max_file_bytes,
                "drop_duplicate_content": filters.drop_duplicate_content,
                "skipped_by_reason": self._selector.total_skip_counts(),
            },
        )


def load_corpus_records(path: Path | str) -> list[CorpusRecord]:
    """Read ``corpus.jsonl`` back into :class:`CorpusRecord` objects."""
    from utils.io_utils import read_jsonl  # local import keeps module import light

    records = [CorpusRecord.from_dict(row) for row in read_jsonl(path)]
    if not records:
        raise CorpusError(f"Corpus file {path} contains no records")
    return records
