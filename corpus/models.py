"""Domain objects and configuration schema for the D1 code corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from utils.errors import ConfigError, CorpusError

#: Value written to ``CorpusRecord.language``. Phase II is Python-only.
LANGUAGE_PYTHON = "python"


# ---------------------------------------------------------------------------
# Configuration schema (mirrors configs/dataset.yaml)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutputConfig:
    """Where the collector writes its artefacts."""

    raw_dir: Path
    corpus_path: Path
    manifest_path: Path
    include_content: bool = True


@dataclass(frozen=True)
class AcquisitionConfig:
    """How source material is obtained."""

    mode: str = "git"
    git_depth: int = 1
    timeout_seconds: int = 600
    fail_fast: bool = True

    VALID_MODES = ("git", "local", "synthetic")

    def __post_init__(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise ConfigError(
                f"acquisition.mode must be one of {self.VALID_MODES}, got '{self.mode}'"
            )
        if self.git_depth < 0:
            raise ConfigError("acquisition.git_depth must be >= 0 (0 means full history)")
        if self.timeout_seconds <= 0:
            raise ConfigError("acquisition.timeout_seconds must be positive")


@dataclass(frozen=True)
class FilterConfig:
    """File selection rules applied to every source repository."""

    include_globs: list[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_globs: list[str] = field(default_factory=list)
    min_code_lines: int = 10
    max_file_bytes: int = 262_144
    drop_duplicate_content: bool = True

    def __post_init__(self) -> None:
        if not self.include_globs:
            raise ConfigError("filters.include_globs must not be empty")
        if self.min_code_lines < 0:
            raise ConfigError("filters.min_code_lines must be >= 0")
        if self.max_file_bytes <= 0:
            raise ConfigError("filters.max_file_bytes must be positive")


@dataclass(frozen=True)
class SourceConfig:
    """One source repository == one CLASP cluster."""

    cluster_id: str
    project_label: str
    license: str
    repo_url: str | None = None
    ref: str | None = None
    subpaths: list[str] = field(default_factory=list)
    rationale: str = ""
    local_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.cluster_id or not self.cluster_id.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(
                f"cluster_id '{self.cluster_id}' must be alphanumeric with - or _ separators"
            )
        if self.cluster_id != self.cluster_id.lower():
            raise ConfigError(f"cluster_id '{self.cluster_id}' must be lowercase")
        if not self.license:
            raise ConfigError(f"Source '{self.cluster_id}' must declare a license")

    def checkout_dir(self, raw_dir: Path) -> Path:
        """Directory this source is (or will be) checked out into."""
        return self.local_path if self.local_path else raw_dir / self.cluster_id


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters for the deterministic offline stand-in corpus."""

    files_per_source: int = 12
    seed: int = 20260616

    def __post_init__(self) -> None:
        if self.files_per_source < 1:
            raise ConfigError("synthetic.files_per_source must be >= 1")


@dataclass(frozen=True)
class CorpusConfig:
    """Root of ``configs/dataset.yaml`` under the ``corpus`` key."""

    name: str
    version: str
    output: OutputConfig
    acquisition: AcquisitionConfig
    filters: FilterConfig
    sources: list[SourceConfig]
    description: str = ""
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ConfigError("corpus.sources must declare at least one source")
        ids = [source.cluster_id for source in self.sources]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ConfigError(f"Duplicate cluster_id(s) in corpus.sources: {sorted(duplicates)}")
        if self.acquisition.mode == "git":
            missing = [s.cluster_id for s in self.sources if not s.repo_url]
            if missing:
                raise ConfigError(f"acquisition.mode=git but no repo_url for: {sorted(missing)}")

    def source_for(self, cluster_id: str) -> SourceConfig:
        for source in self.sources:
            if source.cluster_id == cluster_id:
                return source
        raise CorpusError(f"No source configured for cluster_id '{cluster_id}'")


# ---------------------------------------------------------------------------
# Corpus records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusRecord:
    """One collected source file. Serialises to a line of ``corpus.jsonl``.

    Validated by ``interfaces/schemas/corpus_record.schema.json``.
    """

    file_id: str
    cluster_id: str
    project_label: str
    relative_path: str
    language: str
    num_lines: int
    num_code_lines: int
    num_bytes: int
    content_sha256: str
    license: str
    source_url: str | None = None
    source_ref: str | None = None
    content: str | None = None

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_content:
            payload["content"] = None
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusRecord":
        try:
            return cls(
                file_id=str(data["file_id"]),
                cluster_id=str(data["cluster_id"]),
                project_label=str(data["project_label"]),
                relative_path=str(data["relative_path"]),
                language=str(data["language"]),
                num_lines=int(data["num_lines"]),
                num_code_lines=int(data["num_code_lines"]),
                num_bytes=int(data["num_bytes"]),
                content_sha256=str(data["content_sha256"]),
                license=str(data["license"]),
                source_url=data.get("source_url"),
                source_ref=data.get("source_ref"),
                content=data.get("content"),
            )
        except (KeyError, ValueError) as exc:
            raise CorpusError(f"Invalid corpus record: {exc}") from exc


@dataclass(frozen=True)
class SourceStats:
    """Per-source outcome of a collection run, for the manifest and reports."""

    cluster_id: str
    project_label: str
    license: str
    source_url: str | None
    resolved_ref: str | None
    files_discovered: int
    files_kept: int
    files_skipped: dict[str, int]
    num_lines: int
    num_code_lines: int
    num_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusManifest:
    """Provenance record for one collection run.

    Written to ``datasets/metadata/corpus_manifest.json``. The partitioner
    reads ``corpus_sha256`` from here to pin the exact input it consumed.
    """

    corpus_name: str
    corpus_version: str
    created_at: str
    acquisition_mode: str
    synthetic: bool
    corpus_path: str
    corpus_sha256: str
    total_files: int
    total_lines: int
    total_code_lines: int
    total_bytes: int
    num_clusters: int
    include_content: bool
    sources: list[SourceStats] = field(default_factory=list)
    duplicates_dropped: int = 0
    filter_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [s.to_dict() for s in self.sources]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusManifest":
        try:
            return cls(
                corpus_name=str(data["corpus_name"]),
                corpus_version=str(data["corpus_version"]),
                created_at=str(data["created_at"]),
                acquisition_mode=str(data["acquisition_mode"]),
                synthetic=bool(data["synthetic"]),
                corpus_path=str(data["corpus_path"]),
                corpus_sha256=str(data["corpus_sha256"]),
                total_files=int(data["total_files"]),
                total_lines=int(data["total_lines"]),
                total_code_lines=int(data["total_code_lines"]),
                total_bytes=int(data["total_bytes"]),
                num_clusters=int(data["num_clusters"]),
                include_content=bool(data.get("include_content", True)),
                sources=[SourceStats(**s) for s in data.get("sources", [])],
                duplicates_dropped=int(data.get("duplicates_dropped", 0)),
                filter_summary=dict(data.get("filter_summary", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusError(f"Invalid corpus manifest: {exc}") from exc
