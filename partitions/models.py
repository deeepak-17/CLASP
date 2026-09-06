"""Configuration schema for the partitioning stage (mirrors ``configs/partition.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from interfaces.contracts import PartitionStrategyName
from utils.errors import ConfigError


@dataclass(frozen=True)
class PartitionInputConfig:
    """Where the partitioner reads from."""

    corpus_path: Path
    corpus_manifest_path: Path | None = None


@dataclass(frozen=True)
class PartitionOutputConfig:
    """Where the partitioner writes shards and the manifest."""

    partitions_dir: Path
    manifest_path: Path
    shard_filename_template: str = "{client_id}.jsonl"
    include_content: bool = True

    def __post_init__(self) -> None:
        if "{client_id}" not in self.shard_filename_template:
            raise ConfigError("output.shard_filename_template must contain '{client_id}'")

    def shard_path(self, client_id: str) -> Path:
        return self.partitions_dir / self.shard_filename_template.format(client_id=client_id)

    def report_suffix(self) -> str:
        """Disambiguator for shared-script report filenames.

        ``scripts/build_partitions.py`` and ``scripts/validate_partitions.py``
        are reused as-is for both the Week-2 project-level config and the
        Week-4 per-developer config (``configs/partition_per_developer.yaml``)
        — same code, different ``--config``. Without this, both runs would
        write ``reports/partition_metadata.md`` /
        ``reports/partition_validation_report.md``, and the second run would
        silently clobber the first's report.

        Empty for the canonical ``datasets/partitions`` location, so Week-2's
        report filenames are completely unchanged; otherwise the output
        directory's own name (``per_developer`` -> ``_per_developer``).
        """
        from utils.paths import project_paths

        if self.partitions_dir == project_paths().partitions:
            return ""
        return f"_{self.partitions_dir.name}"


@dataclass(frozen=True)
class MaterializeConfig:
    """Config for ``scripts/materialize_client_repo.py`` (mirrors ``configs/materialize_client.yaml``).

    Turns one client's partition shard into a real on-disk file tree at the
    files' original relative paths — what P1's task E3.1 ("Train client LoRA
    on a real partition") needs as input — while deterministically holding
    out a fraction of the files into a separate directory so P1 never trains
    on the files used to score the result. Purely a downstream reader of
    ``datasets/partitions/``; never writes there.
    """

    manifest_path: Path
    output_root: Path
    repo_dirname: str = "repo"
    held_out_dirname: str = "held_out"
    held_out_fraction: float = 0.10
    #: Independent of ``PartitionConfig.seed`` so a client can be
    #: re-materialized with a different held-out cut without touching the
    #: partition itself.
    seed: int = 20260616

    def __post_init__(self) -> None:
        if not 0.0 < self.held_out_fraction < 1.0:
            raise ConfigError("materialize.held_out_fraction must lie strictly between 0 and 1")
        if self.seed < 0:
            raise ConfigError("materialize.seed must be non-negative")
        if self.repo_dirname == self.held_out_dirname:
            raise ConfigError("materialize.repo_dirname and materialize.held_out_dirname must differ")

    def client_dir(self, client_id: str) -> Path:
        return self.output_root / client_id

    def repo_dir(self, client_id: str) -> Path:
        return self.client_dir(client_id) / self.repo_dirname

    def held_out_dir(self, client_id: str) -> Path:
        return self.client_dir(client_id) / self.held_out_dirname


@dataclass(frozen=True)
class ClientNamingConfig:
    """How cluster ids map to Flower client ids."""

    template: str = "client-{cluster_id}"

    def __post_init__(self) -> None:
        if "{cluster_id}" not in self.template:
            raise ConfigError("client_naming.template must contain '{cluster_id}'")

    def client_id(self, cluster_id: str) -> str:
        return self.template.format(cluster_id=cluster_id)


@dataclass(frozen=True)
class ValidationConfig:
    """Week-2 Wednesday acceptance thresholds.

    Hard constraints protect correctness and CLASP's isolation claim; the
    balance thresholds are diagnostics that surface as warnings unless
    ``fail_on_warning`` is set.
    """

    require_disjoint_files: bool = True
    require_full_coverage: bool = True
    require_project_purity: bool = True
    require_content_disjoint: bool = True
    min_files_per_shard: int = 3
    max_imbalance_ratio: float = 20.0
    max_gini: float = 0.50
    max_coefficient_of_variation: float = 1.0
    fail_on_warning: bool = False

    def __post_init__(self) -> None:
        if self.min_files_per_shard < 1:
            raise ConfigError("validation.min_files_per_shard must be >= 1")
        if self.max_imbalance_ratio < 1.0:
            raise ConfigError("validation.max_imbalance_ratio must be >= 1.0")
        if not 0.0 <= self.max_gini <= 1.0:
            raise ConfigError("validation.max_gini must lie in [0, 1]")
        if self.max_coefficient_of_variation < 0:
            raise ConfigError("validation.max_coefficient_of_variation must be >= 0")


@dataclass(frozen=True)
class PerDeveloperConfig:
    """Week-4 Monday: parameters for the per-developer (individual-style) split.

    See ``partitions/strategies.py::PerDeveloperStrategy`` for the full
    rationale. In short: this corpus's git history is a depth-1 shallow
    clone (``configs/dataset.yaml``'s ``acquisition.git_depth: 1``), so every
    file in a project resolves to the same single commit/author — there is
    no real per-file authorship signal to split on. Two knobs control the
    defensible proxy used instead:

    * where a project's files form genuine subpackages (``src/x/routing/``,
      ``src/x/debug/``, ...), each subpackage becomes one developer's slice
      (``basis: module_path``);
    * where a project is a flat file list (no subpackages), files are
      deterministically hash-shuffled and cut into equal-ish contiguous
      chunks (``basis: deterministic_hash_chunk``) — reproducible from the
      seed, but explicitly *not* claiming any real identity.
    """

    #: Target files per developer shard when falling back to hash-chunking.
    #: Module-path grouping is unaffected by this value; it only sizes the
    #: fallback for flat (no-subpackage) projects.
    target_files_per_developer: int = 6

    #: A candidate developer grouping (by either basis) is accepted only if
    #: every resulting shard has at least this many files; otherwise the
    #: whole cluster falls back to a single developer (== the project-level
    #: shard), which is the honest answer when a project is too small to
    #: split further (e.g. Colorama's 5 files).
    min_files_per_developer: int = 2

    #: Naming template for per-developer client ids. Distinct from
    #: ``ClientNamingConfig.template`` (project-level) because it needs a
    #: second placeholder.
    client_id_template: str = "client-{cluster_id}-dev{dev_index}"

    def __post_init__(self) -> None:
        if self.target_files_per_developer < 1:
            raise ConfigError("per_developer.target_files_per_developer must be >= 1")
        if self.min_files_per_developer < 1:
            raise ConfigError("per_developer.min_files_per_developer must be >= 1")
        if "{cluster_id}" not in self.client_id_template or "{dev_index}" not in self.client_id_template:
            raise ConfigError(
                "per_developer.client_id_template must contain both {cluster_id} and {dev_index}"
            )

    def client_id(self, cluster_id: str, dev_index: int) -> str:
        return self.client_id_template.format(cluster_id=cluster_id, dev_index=dev_index)


@dataclass(frozen=True)
class IntegrationConfig:
    """Expectations held by P2's Cluster Layer, checked on Week-2 Friday."""

    expected_client_count: int = 3
    max_client_count: int = 16

    def __post_init__(self) -> None:
        if self.expected_client_count < 1:
            raise ConfigError("integration.expected_client_count must be >= 1")
        if self.max_client_count < self.expected_client_count:
            raise ConfigError(
                "integration.max_client_count must be >= integration.expected_client_count"
            )


@dataclass(frozen=True)
class PartitionConfig:
    """Root of ``configs/partition.yaml`` under the ``partition`` key."""

    strategy: str
    seed: int
    input: PartitionInputConfig
    output: PartitionOutputConfig
    manifest_version: str = "1.0.0"
    client_naming: ClientNamingConfig = field(default_factory=ClientNamingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    per_developer: PerDeveloperConfig = field(default_factory=PerDeveloperConfig)

    def __post_init__(self) -> None:
        try:
            PartitionStrategyName(self.strategy)
        except ValueError as exc:
            valid = ", ".join(s.value for s in PartitionStrategyName)
            raise ConfigError(f"Unknown partition.strategy '{self.strategy}'. Valid: {valid}") from exc
        if self.seed < 0:
            raise ConfigError("partition.seed must be non-negative")

    @property
    def strategy_name(self) -> PartitionStrategyName:
        return PartitionStrategyName(self.strategy)
