"""Exception hierarchy for CLASP-P5.

Every failure mode this repository raises deliberately derives from
:class:`ClaspP5Error`, so callers (and the CLI entry points in ``scripts/``)
can distinguish "our code said no" from "the interpreter blew up".
"""

from __future__ import annotations


class ClaspP5Error(Exception):
    """Base class for every error raised deliberately by CLASP-P5 code."""


class ConfigError(ClaspP5Error):
    """A configuration file is missing, malformed, or semantically invalid."""


class CorpusError(ClaspP5Error):
    """Dataset survey / sourcing / collection failed."""


class PartitionError(ClaspP5Error):
    """Partition construction or validation failed."""


class EvaluationError(ClaspP5Error):
    """The evaluation harness could not be built or executed."""


class ContractViolationError(ClaspP5Error):
    """An artefact does not satisfy the agreed cross-module interface contract.

    Raised by :mod:`interfaces.validation` when a manifest, snapshot or result
    document fails schema or semantic validation. This is the error that
    guards the P4 (State Registry) integration seam.
    """


class DependencyError(ClaspP5Error):
    """A required third-party dependency is absent or the wrong version."""
