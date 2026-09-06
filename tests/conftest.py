"""Shared pytest fixtures.

Every fixture builds its subject in a ``tmp_path``; no test reads or writes
the repository's real ``datasets/`` tree, so the suite is safe to run at any
point in the pipeline and in any order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.models import (
    AcquisitionConfig,
    CorpusConfig,
    CorpusRecord,
    FilterConfig,
    OutputConfig,
    SourceConfig,
    SyntheticConfig,
)
from evaluation.models import (
    BackendConfig,
    BenchmarkConfig,
    EvaluationConfig,
    GenerationConfig,
    RunConfig,
    ScoringConfig,
)
from interfaces.contracts import BenchmarkName
from partitions.models import (
    ClientNamingConfig,
    IntegrationConfig,
    PartitionConfig,
    PartitionInputConfig,
    PartitionOutputConfig,
    ValidationConfig,
)
from utils.logging_utils import configure_logging


@pytest.fixture(autouse=True, scope="session")
def _quiet_logging() -> None:
    """Keep test output readable; failures still surface via assertions."""
    configure_logging(level="CRITICAL", log_file=None, force=True)


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------
def make_record(
    cluster_id: str,
    index: int,
    *,
    code_lines: int = 20,
    content: str | None = None,
    project_label: str | None = None,
) -> CorpusRecord:
    """Build a deterministic corpus record for tests."""
    body = content if content is not None else f"# {cluster_id} module {index}\n" * code_lines
    from utils.io_utils import sha256_text

    relative_path = f"src/{cluster_id}/module_{index:02d}.py"
    return CorpusRecord(
        file_id=f"{cluster_id}:{relative_path}",
        cluster_id=cluster_id,
        project_label=project_label or cluster_id.title(),
        relative_path=relative_path,
        language="python",
        num_lines=body.count("\n"),
        num_code_lines=code_lines,
        num_bytes=len(body.encode("utf-8")),
        content_sha256=sha256_text(body),
        license="BSD-3-Clause",
        source_url=f"https://example.invalid/{cluster_id}.git",
        source_ref="deadbeef",
        content=body,
    )


@pytest.fixture
def sample_records() -> list[CorpusRecord]:
    """Three clusters of deliberately unequal size (6, 4 and 3 files)."""
    records: list[CorpusRecord] = []
    for index in range(6):
        records.append(make_record("alpha", index, code_lines=30))
    for index in range(4):
        records.append(make_record("beta", index, code_lines=20))
    for index in range(3):
        records.append(make_record("gamma", index, code_lines=12))
    return records


@pytest.fixture
def corpus_file(tmp_path: Path, sample_records: list[CorpusRecord]) -> Path:
    """A ``corpus.jsonl`` on disk containing :func:`sample_records`."""
    from utils.io_utils import write_jsonl

    path = tmp_path / "processed" / "corpus.jsonl"
    write_jsonl(path, (record.to_dict() for record in sample_records))
    return path


@pytest.fixture
def synthetic_corpus_config(tmp_path: Path) -> CorpusConfig:
    """A collector config that generates a synthetic corpus under ``tmp_path``."""
    return CorpusConfig(
        name="TEST",
        version="0.1.0",
        description="synthetic test corpus",
        output=OutputConfig(
            raw_dir=tmp_path / "raw",
            corpus_path=tmp_path / "processed" / "corpus.jsonl",
            manifest_path=tmp_path / "metadata" / "corpus_manifest.json",
            include_content=True,
        ),
        acquisition=AcquisitionConfig(mode="synthetic"),
        filters=FilterConfig(
            include_globs=["**/*.py"],
            exclude_globs=["**/tests/**"],
            min_code_lines=5,
            max_file_bytes=262_144,
            drop_duplicate_content=True,
        ),
        sources=[
            SourceConfig(cluster_id="alpha", project_label="Alpha", license="BSD-3-Clause"),
            SourceConfig(cluster_id="beta", project_label="Beta", license="Apache-2.0"),
            SourceConfig(cluster_id="gamma", project_label="Gamma", license="MIT"),
        ],
        synthetic=SyntheticConfig(files_per_source=6, seed=1234),
    )


# ---------------------------------------------------------------------------
# Partition fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def partition_config(tmp_path: Path, corpus_file: Path) -> PartitionConfig:
    """A partition config wired to the temporary corpus."""
    return PartitionConfig(
        strategy="project_level",
        seed=99,
        manifest_version="1.0.0",
        input=PartitionInputConfig(corpus_path=corpus_file, corpus_manifest_path=None),
        output=PartitionOutputConfig(
            partitions_dir=tmp_path / "partitions",
            manifest_path=tmp_path / "partitions" / "manifest.json",
            shard_filename_template="{client_id}.jsonl",
            include_content=True,
        ),
        client_naming=ClientNamingConfig(template="client-{cluster_id}"),
        validation=ValidationConfig(min_files_per_shard=2, max_imbalance_ratio=5.0),
        integration=IntegrationConfig(expected_client_count=3, max_client_count=16),
    )


# ---------------------------------------------------------------------------
# Evaluation fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def evaluation_config(tmp_path: Path) -> EvaluationConfig:
    """An evaluation config using the bundled fixtures and a temp results dir."""
    from utils.paths import project_paths

    root = project_paths().evaluation
    return EvaluationConfig(
        run=RunConfig(
            mode="dry_run",
            run_id_prefix="test",
            limit=2,
            num_samples_per_task=1,
            continue_on_task_error=True,
            results_dir=tmp_path / "results",
        ),
        generation=GenerationConfig(
            max_new_tokens=64, temperature=0.0, stop_sequences=["\ndef ", "\nclass "]
        ),
        backend=BackendConfig(kind="mock", model_id="mock/test-model"),
        benchmarks={
            "humaneval": BenchmarkConfig(
                enabled=True,
                tasks_path=root / "humaneval" / "sample_tasks.jsonl",
                sample_tasks_path=root / "humaneval" / "sample_tasks.jsonl",
            ),
            "mbpp": BenchmarkConfig(
                enabled=True,
                tasks_path=root / "mbpp" / "sample_tasks.jsonl",
                sample_tasks_path=root / "mbpp" / "sample_tasks.jsonl",
            ),
        },
        scoring=ScoringConfig(pass_at_k=[1], execution_enabled=False),
    )


@pytest.fixture
def benchmark_names() -> list[BenchmarkName]:
    return list(BenchmarkName)
