"""Typed configuration loading for CLASP-P5.

Design notes
------------
* YAML is the on-disk format (``configs/*.yaml``); PyYAML's ``safe_load`` is
  used so config files can never execute code.
* Every config section is materialised into a frozen dataclass so downstream
  code gets attribute access, type hints and validation instead of dictionary
  spelunking.
* ``${ENV_VAR}`` and ``${ENV_VAR:default}`` placeholders in string values are
  substituted at load time. This is how container/CI runs override paths
  without editing tracked files.
* Relative paths resolve against the repository root, never the CWD.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from utils.errors import ConfigError
from utils.paths import project_paths

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Raw YAML loading
# ---------------------------------------------------------------------------
def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML document into a dict, expanding ``${ENV}`` placeholders.

    Raises:
        ConfigError: If the file is missing, unparseable, or not a mapping.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise ConfigError(f"Configuration file not found: {target}")

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {target}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {target} must be a mapping, got {type(raw).__name__}")
    return _expand_env(raw)


def _expand_env(node: Any) -> Any:
    """Recursively substitute ``${VAR}`` / ``${VAR:default}`` in string leaves."""
    if isinstance(node, dict):
        return {key: _expand_env(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_env(item) for item in node]
    if isinstance(node, str):
        return _ENV_PATTERN.sub(_substitute, node)
    return node


def _substitute(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    value = os.environ.get(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ConfigError(f"Environment variable ${{{name}}} is referenced by a config file but not set")


def resolve_path(value: Path | str) -> Path:
    """Resolve a configured path against the repository root when relative."""
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate
    return (project_paths().root / candidate).resolve()


# ---------------------------------------------------------------------------
# Generic dataclass hydration
# ---------------------------------------------------------------------------
def build_dataclass(cls: type[T], data: Mapping[str, Any], *, context: str = "config") -> T:
    """Instantiate a (possibly nested) dataclass from a mapping.

    Unknown keys are rejected rather than silently dropped: a typo in a config
    file must fail loudly, not change behaviour invisibly.

    Args:
        cls: Target dataclass type.
        data: Mapping of field name to raw value.
        context: Dotted path used in error messages.

    Raises:
        ConfigError: On unknown keys, missing required keys, or type mismatch.
    """
    if not is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in {context}: {', '.join(sorted(unknown))}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # dataclass default (or a missing-arg TypeError below) applies
        kwargs[f.name] = _coerce(hints[f.name], data[f.name], context=f"{context}.{f.name}")

    try:
        return cls(**kwargs)  # type: ignore[return-value]
    except TypeError as exc:
        raise ConfigError(f"Invalid {context}: {exc}") from exc


def _coerce(annotation: Any, value: Any, *, context: str) -> Any:
    """Best-effort conversion of a raw YAML value to its annotated type."""
    origin = get_origin(annotation)

    # Optional[X] / X | None: null passes through, otherwise coerce as X.
    if origin in (Union, UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if value is None:
            return None
        if len(members) == 1:
            return _coerce(members[0], value, context=context)
        return value  # genuine multi-type union: accept as parsed

    if annotation is Path:
        return resolve_path(value)

    if origin in (list, Sequence, tuple):
        if not isinstance(value, list):
            raise ConfigError(f"{context} must be a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation) or (Any,)
        return [_coerce(item_type, item, context=f"{context}[{i}]") for i, item in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}")
        args = get_args(annotation)
        val_type = args[1] if len(args) == 2 else Any
        return {k: _coerce(val_type, v, context=f"{context}.{k}") for k, v in value.items()}

    if is_dataclass(annotation) and isinstance(value, Mapping):
        return build_dataclass(annotation, value, context=context)

    if annotation in (int, float, str, bool) and value is not None:
        if annotation is bool and not isinstance(value, bool):
            raise ConfigError(f"{context} must be a boolean, got {value!r}")
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{context} must be {annotation.__name__}, got {value!r}") from exc

    return value


def load_config(cls: type[T], path: Path | str, *, section: str | None = None) -> T:
    """Load a YAML file (optionally one section of it) into a dataclass.

    Args:
        cls: Dataclass describing the expected shape.
        path: YAML file path; relative paths resolve against the repo root.
        section: Top-level key to descend into before hydrating ``cls``.
    """
    resolved = resolve_path(path)
    raw = load_yaml(resolved)
    if section is not None:
        if section not in raw:
            raise ConfigError(f"Section '{section}' missing from {resolved}")
        payload = raw[section]
        if not isinstance(payload, Mapping):
            raise ConfigError(f"Section '{section}' in {resolved} must be a mapping")
    else:
        payload = raw
    return build_dataclass(cls, payload, context=f"{resolved.name}" + (f":{section}" if section else ""))


# ---------------------------------------------------------------------------
# Shared config sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoggingConfig:
    """Mirror of ``configs/logging.yaml``."""

    level: str = "INFO"
    log_file: Path | None = None
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%dT%H:%M:%S"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3

    def as_mapping(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "log_file": str(self.log_file) if self.log_file else None,
            "format": self.format,
            "date_format": self.date_format,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
        }


def load_logging_config(path: Path | str | None = None) -> LoggingConfig:
    """Load ``configs/logging.yaml``, falling back to defaults if absent."""
    target = resolve_path(path) if path else project_paths().configs / "logging.yaml"
    if not target.is_file():
        return LoggingConfig()
    raw = load_yaml(target)
    payload = raw.get("logging", raw)
    if not isinstance(payload, Mapping):
        raise ConfigError(f"'logging' section in {target} must be a mapping")
    data = dict(payload)
    if data.get("log_file") in (None, "", "null"):
        data.pop("log_file", None)
    return build_dataclass(LoggingConfig, data, context=f"{target.name}:logging")


@dataclass(frozen=True)
class SeedConfig:
    """Determinism controls shared by the corpus and partition pipelines."""

    seed: int = 20260616
    deterministic_ordering: bool = True
    tags: list[str] = field(default_factory=list)
