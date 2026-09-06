"""Contract validation — the executable half of the interface agreement.

A contract that is only written down drifts. This module makes the contracts in
:mod:`interfaces.contracts` and ``interfaces/schemas/*.json`` *checkable*, in
two independent layers:

1. **Structural** — JSON Schema (draft 2020-12) via ``jsonschema``. Catches
   missing fields, wrong types, bad patterns.
2. **Semantic** — round-tripping the document through the dataclass, whose
   ``__post_init__`` enforces cross-field invariants a schema cannot express
   (``num_clients == len(shards)``, ``total_files == sum(shard.num_files)``).

``jsonschema`` is an optional dependency. If it is absent, layer 1 degrades to
a reported *skip* rather than a silent pass — a check that quietly stops
checking is worse than no check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from interfaces.contracts import EvalResult, PartitionManifest, SnapshotMetadata
from utils.errors import ClaspP5Error, ContractViolationError
from utils.io_utils import read_json
from utils.logging_utils import get_logger
from utils.paths import project_paths

_LOG = get_logger(__name__)

try:  # pragma: no cover - exercised implicitly by whichever branch applies
    import jsonschema

    _JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
    _JSONSCHEMA_AVAILABLE = False


#: Logical contract name -> (schema filename, dataclass implementing it).
SCHEMA_REGISTRY: Final[dict[str, tuple[str, type | None]]] = {
    "partition_manifest": ("partition_manifest.schema.json", PartitionManifest),
    "eval_result": ("eval_result.schema.json", EvalResult),
    "snapshot_metadata": ("snapshot_metadata.schema.json", SnapshotMetadata),
    "corpus_record": ("corpus_record.schema.json", None),
}


@dataclass
class ValidationReport:
    """Accumulated outcome of validating one document against one contract."""

    contract: str
    document: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_checks: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no errors were recorded."""
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def skip(self, message: str) -> None:
        self.skipped_checks.append(message)

    def passed(self, message: str) -> None:
        self.checks_run.append(message)

    def raise_for_status(self) -> None:
        """Raise :class:`ContractViolationError` if any error was recorded."""
        if not self.ok:
            joined = "; ".join(self.errors)
            raise ContractViolationError(f"{self.contract} validation failed for {self.document}: {joined}")

    def summary(self) -> str:
        return (
            f"[{'PASS' if self.ok else 'FAIL'}] {self.contract} <- {self.document} "
            f"({len(self.checks_run)} passed, {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), {len(self.skipped_checks)} skipped)"
        )


def schema_path(contract: str) -> Path:
    """Return the on-disk path of a registered JSON Schema."""
    if contract not in SCHEMA_REGISTRY:
        raise ClaspP5Error(
            f"Unknown contract '{contract}'. Known: {', '.join(sorted(SCHEMA_REGISTRY))}"
        )
    filename, _ = SCHEMA_REGISTRY[contract]
    path = project_paths().schemas / filename
    if not path.is_file():
        raise ClaspP5Error(f"Schema file missing: {path}")
    return path


def load_schema(contract: str) -> dict[str, Any]:
    """Load a registered JSON Schema document."""
    return read_json(schema_path(contract))


def jsonschema_available() -> bool:
    """Whether structural validation can run in this environment."""
    return _JSONSCHEMA_AVAILABLE


def validate_document(
    contract: str,
    payload: Mapping[str, Any],
    *,
    document_name: str = "<in-memory>",
) -> ValidationReport:
    """Validate ``payload`` structurally and semantically against ``contract``.

    Args:
        contract: Key from :data:`SCHEMA_REGISTRY`.
        payload: Parsed JSON document.
        document_name: Label used in the report (usually a file path).

    Returns:
        A :class:`ValidationReport`; inspect :attr:`~ValidationReport.ok` or
        call :meth:`~ValidationReport.raise_for_status`.
    """
    report = ValidationReport(contract=contract, document=document_name)
    _, dataclass_type = SCHEMA_REGISTRY[contract]

    # --- layer 1: structural ------------------------------------------
    if _JSONSCHEMA_AVAILABLE:
        schema = load_schema(contract)
        validator_cls = jsonschema.validators.validator_for(schema)  # type: ignore[union-attr]
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        structural_errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
        for err in structural_errors:
            location = "/".join(str(part) for part in err.path) or "<root>"
            report.error(f"schema[{location}]: {err.message}")
        if not structural_errors:
            report.passed("json-schema structural validation")
    else:
        report.skip("json-schema structural validation (install `jsonschema` to enable)")

    # --- layer 2: semantic --------------------------------------------
    if dataclass_type is None:
        report.skip(f"semantic round-trip (no dataclass bound to '{contract}')")
        return report

    try:
        instance = dataclass_type.from_dict(payload)  # type: ignore[attr-defined]
    except ContractViolationError as exc:
        report.error(f"semantic: {exc}")
        return report
    except Exception as exc:  # defensive: a malformed doc must not crash the CLI
        report.error(f"semantic: unexpected {type(exc).__name__}: {exc}")
        return report

    report.passed("dataclass semantic round-trip")

    # --- layer 3: re-serialisation stability ----------------------------
    # If to_dict(from_dict(x)) != x the producer and the contract disagree
    # about representation even though both "validate" -- exactly the class of
    # bug that surfaces as a mystery KeyError in someone else's module.
    reserialised = instance.to_dict()  # type: ignore[attr-defined]
    drift = _diff_keys(dict(payload), reserialised)
    if drift:
        report.warn(f"re-serialisation drift on field(s): {', '.join(sorted(drift))}")
    else:
        report.passed("re-serialisation stability")

    return report


def validate_file(contract: str, path: Path | str) -> ValidationReport:
    """Load a JSON file and validate it against ``contract``."""
    target = Path(path).expanduser()
    paths = project_paths()
    payload = read_json(target)
    if not isinstance(payload, Mapping):
        report = ValidationReport(contract=contract, document=paths.relative(target))
        report.error(f"document root must be a JSON object, got {type(payload).__name__}")
        return report
    return validate_document(contract, payload, document_name=paths.relative(target))


def _diff_keys(original: Mapping[str, Any], reserialised: Mapping[str, Any]) -> set[str]:
    """Return top-level keys whose values differ between the two mappings.

    Keys absent from ``original`` but present with a ``None``/default value in
    ``reserialised`` are ignored — optional fields legitimately materialise.
    """
    differing: set[str] = set()
    for key, value in reserialised.items():
        if key not in original:
            if value not in (None, [], {}, 0, ""):
                differing.add(key)
            continue
        if original[key] != value:
            differing.add(key)
    for key in original:
        if key not in reserialised:
            differing.add(key)
    return differing
