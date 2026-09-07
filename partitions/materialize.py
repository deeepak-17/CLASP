"""Materialize one client's partition shard as a real on-disk file tree.

Supports P1's task E3.1 ("Train client LoRA on a real partition"): P1 needs
one project's ``.py`` files as an actual directory of files at their real
relative paths, not a JSONL shard — with a deterministic slice of files held
out into a separate directory so P1 never trains on the files used to score
the result.

Pipeline
--------
::

    datasets/partitions/<client_id>.jsonl        (read-only)
        -> hash-ordered deterministic split       (held_out_fraction)
        -> datasets/materialized/<client_id>/repo/**/*.py
        -> datasets/materialized/<client_id>/held_out/**/*.py
        -> datasets/materialized/<client_id>/materialize_manifest.json

Reads only from ``datasets/partitions/``, through the existing manifest/shard
loaders in :mod:`partitions.partitioner`; never writes there. The held-out
split reuses the same ``sha256(seed:client_id:file_id)`` hash-ordering
:class:`partitions.strategies.PerDeveloperStrategy` uses for its
deterministic-hash-chunk basis, so it is deterministic, seeded and
reproducible without depending on Python's global RNG state or file write
order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from corpus.models import CorpusRecord
from interfaces.contracts import PartitionManifest
from partitions.models import MaterializeConfig
from partitions.partitioner import load_shard_records
from utils.config import load_config
from utils.errors import PartitionError
from utils.io_utils import ensure_dir, write_json
from utils.logging_utils import get_logger
from utils.paths import project_paths
from utils.reporting import MarkdownReport
from utils.timing import utc_timestamp

_LOG = get_logger(__name__)


def load_materialize_config(path: Path | str | None = None) -> MaterializeConfig:
    """Load ``configs/materialize_client.yaml`` into a :class:`MaterializeConfig`."""
    target = path or (project_paths().configs / "materialize_client.yaml")
    return load_config(MaterializeConfig, target, section="materialize")


@dataclass(frozen=True)
class MaterializeResult:
    """What one materialization run produced."""

    client_id: str
    cluster_id: str
    repo_dir: Path
    held_out_dir: Path
    manifest_path: Path
    kept_files: tuple[str, ...]
    held_out_files: tuple[str, ...]
    seed: int
    held_out_fraction: float

    @property
    def num_kept(self) -> int:
        return len(self.kept_files)

    @property
    def num_held_out(self) -> int:
        return len(self.held_out_files)


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------
def _holdout_sort_key(seed: int, client_id: str, file_id: str) -> str:
    """``sha256(seed:client_id:file_id)`` — same construction as
    ``PerDeveloperStrategy._hash_chunks`` (partitions/strategies.py), reused
    here for the same reason: hashing before splitting avoids the artefact of
    grouping by path/name prefix, which is not a meaningful boundary either.
    """
    digest = hashlib.sha256(f"{seed}:{client_id}:{file_id}".encode("utf-8"))
    return digest.hexdigest()


def split_held_out(
    records: list[CorpusRecord], *, client_id: str, seed: int, fraction: float
) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
    """Deterministically partition a shard's records into ``(kept, held_out)``.

    Held-out count is ``round(n * fraction)``, clamped to ``[1, n - 1]`` so a
    non-zero fraction always yields both a genuine training set and a genuine
    held-out set, even for D1's smallest shard (Colorama: 5 files).

    Raises:
        PartitionError: If there are fewer than 2 records, so no split can
            leave both sides non-empty.
    """
    n = len(records)
    if n < 2:
        raise PartitionError(
            f"Client '{client_id}' has only {n} file(s); cannot hold out {fraction:.0%} "
            "and still leave a non-empty training set"
        )
    held_out_count = round(n * fraction)
    held_out_count = max(1, min(held_out_count, n - 1))

    ordered = sorted(records, key=lambda r: _holdout_sort_key(seed, client_id, r.file_id))
    held_out, kept = ordered[:held_out_count], ordered[held_out_count:]

    # Restore canonical (file_id) order within each group — the hash sort is
    # only for group membership, matching PerDeveloperStrategy's convention.
    return sorted(kept, key=lambda r: r.file_id), sorted(held_out, key=lambda r: r.file_id)


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------
def _safe_write(root: Path, relative_path: str, content: str) -> Path:
    """Write ``content`` at ``root / relative_path``, refusing to escape ``root``."""
    root_resolved = root.resolve()
    target = (root / relative_path).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise PartitionError(f"Refusing to write outside {root}: relative_path={relative_path!r}")
    ensure_dir(target.parent)
    target.write_text(content, encoding="utf-8")
    return target


def materialize_client_repo(
    manifest: PartitionManifest,
    client_id: str,
    config: MaterializeConfig,
    *,
    dry_run: bool = False,
) -> MaterializeResult:
    """Write one client's shard out as a real file tree, holding out a slice.

    Args:
        manifest: The already-loaded partition manifest (P5's contract
            artefact — see :func:`partitions.partitioner.load_partition_manifest`).
        client_id: e.g. ``"client-flask"``. Must name a shard in ``manifest``.
        config: Where to write, and the held-out fraction/seed.
        dry_run: Compute the split without writing anything.

    Raises:
        ContractViolationError: If ``client_id`` has no shard in ``manifest``.
        PartitionError: If the shard was written index-only (no content), or
            is too small to hold out a non-empty slice from.
    """
    shard = manifest.shard_for(client_id)  # raises ContractViolationError if unknown
    records = load_shard_records(manifest, client_id)

    missing_content = [r.file_id for r in records if r.content is None]
    if missing_content:
        raise PartitionError(
            f"Shard '{client_id}' has {len(missing_content)} record(s) with no content "
            "(shard was written with output.include_content: false); cannot materialize files"
        )

    kept, held_out = split_held_out(
        records, client_id=client_id, seed=config.seed, fraction=config.held_out_fraction
    )

    repo_dir = config.repo_dir(client_id)
    held_out_dir = config.held_out_dir(client_id)
    manifest_path = config.client_dir(client_id) / "materialize_manifest.json"

    if not dry_run:
        ensure_dir(repo_dir)
        ensure_dir(held_out_dir)
        for record in kept:
            _safe_write(repo_dir, record.relative_path, record.content)
        for record in held_out:
            _safe_write(held_out_dir, record.relative_path, record.content)

        write_json(manifest_path, _manifest_payload(shard.cluster_id, shard.project_label, client_id, config, kept, held_out))
        _LOG.info(
            "Materialized %s: %d kept -> %s, %d held out -> %s",
            client_id,
            len(kept),
            project_paths().relative(repo_dir),
            len(held_out),
            project_paths().relative(held_out_dir),
        )
    else:
        _LOG.info(
            "Dry run: %s would write %d kept, %d held out (nothing written)",
            client_id,
            len(kept),
            len(held_out),
        )

    return MaterializeResult(
        client_id=client_id,
        cluster_id=shard.cluster_id,
        repo_dir=repo_dir,
        held_out_dir=held_out_dir,
        manifest_path=manifest_path,
        kept_files=tuple(r.relative_path for r in kept),
        held_out_files=tuple(r.relative_path for r in held_out),
        seed=config.seed,
        held_out_fraction=config.held_out_fraction,
    )


def _manifest_payload(
    cluster_id: str,
    project_label: str,
    client_id: str,
    config: MaterializeConfig,
    kept: list[CorpusRecord],
    held_out: list[CorpusRecord],
) -> dict:
    paths = project_paths()
    return {
        "client_id": client_id,
        "cluster_id": cluster_id,
        "project_label": project_label,
        "source_manifest_path": paths.relative(Path(config.manifest_path)),
        "seed": config.seed,
        "held_out_fraction": config.held_out_fraction,
        "created_at": utc_timestamp(),
        "repo_dir": paths.relative(config.repo_dir(client_id)),
        "held_out_dir": paths.relative(config.held_out_dir(client_id)),
        "num_kept": len(kept),
        "num_held_out": len(held_out),
        "kept_files": [
            {"file_id": r.file_id, "relative_path": r.relative_path, "content_sha256": r.content_sha256}
            for r in kept
        ],
        "held_out_files": [
            {"file_id": r.file_id, "relative_path": r.relative_path, "content_sha256": r.content_sha256}
            for r in held_out
        ],
    }


def render_materialize_report(result: MaterializeResult) -> MarkdownReport:
    """Render a materialization result as a human-readable Markdown summary."""
    paths = project_paths()
    report = MarkdownReport(
        title="CLASP-P5 · Materialize Client Repo",
        subtitle="E3.1 support — real file tree + deterministic held-out split for P1",
    )
    report.heading("1. Summary")
    report.key_values(
        {
            "Client": result.client_id,
            "Cluster (project)": result.cluster_id,
            "Seed": result.seed,
            "Held-out fraction": f"{result.held_out_fraction:.2%}",
            "Kept files": result.num_kept,
            "Held-out files": result.num_held_out,
            "Repo dir": paths.relative(result.repo_dir),
            "Held-out dir": paths.relative(result.held_out_dir),
            "Manifest": paths.relative(result.manifest_path),
        }
    )
    report.heading("2. Held-out files")
    report.bullets(sorted(result.held_out_files))
    return report
