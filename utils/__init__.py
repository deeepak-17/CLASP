"""Reusable, dependency-light utilities shared by every CLASP-P5 module.

Nothing in this package may import from ``corpus``, ``partitions``,
``evaluation`` or ``interfaces`` — the dependency arrow points one way only,
which keeps the utility layer trivially testable in isolation.
"""

from utils.errors import (
    ClaspP5Error,
    ConfigError,
    ContractViolationError,
    CorpusError,
    EvaluationError,
    PartitionError,
)
from utils.logging_utils import configure_logging, get_logger
from utils.paths import ProjectPaths, project_paths

__all__ = [
    "ClaspP5Error",
    "ConfigError",
    "ContractViolationError",
    "CorpusError",
    "EvaluationError",
    "PartitionError",
    "ProjectPaths",
    "configure_logging",
    "get_logger",
    "project_paths",
]
