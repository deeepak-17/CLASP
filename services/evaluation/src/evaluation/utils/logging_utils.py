"""Centralised logging configuration.

Every CLI entry point calls :func:`configure_logging` exactly once; library
modules only ever call :func:`get_logger`. Library code never configures
handlers, which keeps CLASP-P5 importable from P1–P4 test suites without
hijacking their logging setup.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Final, Mapping

_LOGGER_NAMESPACE: Final[str] = "clasp_p5"
_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"
_CONFIGURED = False


class _RelativePathFilter(logging.Filter):
    """Shorten logger names so console output stays readable."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        if record.name.startswith(f"{_LOGGER_NAMESPACE}."):
            record.name = record.name[len(_LOGGER_NAMESPACE) + 1 :]
        return True


def configure_logging(
    level: str | int = "INFO",
    *,
    log_file: Path | str | None = None,
    fmt: str = _DEFAULT_FORMAT,
    date_fmt: str = _DEFAULT_DATE_FORMAT,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``clasp_p5`` logger tree.

    Args:
        level: Threshold as a name (``"DEBUG"``) or numeric level.
        log_file: Optional path for a size-rotating file handler. Parent
            directories are created if absent.
        fmt: ``logging`` format string.
        date_fmt: ``strftime`` pattern for ``%(asctime)s``.
        max_bytes: Rotation threshold for the file handler.
        backup_count: Number of rotated files to retain.
        force: Reconfigure even if configuration already happened. Used by
            tests; production entry points should leave this ``False``.

    Returns:
        The configured root logger for the ``clasp_p5`` namespace.
    """
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAMESPACE)
    if _CONFIGURED and not force:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    numeric_level = _coerce_level(level)
    logger.setLevel(numeric_level)
    # Root logging is owned by the application, not by us.
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    console.addFilter(_RelativePathFilter())
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def configure_from_mapping(config: Mapping[str, Any], *, force: bool = False) -> logging.Logger:
    """Configure logging from a parsed ``configs/logging.yaml`` mapping."""
    return configure_logging(
        level=config.get("level", "INFO"),
        log_file=config.get("log_file"),
        fmt=config.get("format", _DEFAULT_FORMAT),
        date_fmt=config.get("date_format", _DEFAULT_DATE_FORMAT),
        max_bytes=int(config.get("max_bytes", 5 * 1024 * 1024)),
        backup_count=int(config.get("backup_count", 3)),
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger.

    Args:
        name: Usually ``__name__``. A leading ``clasp_p5.`` is added when
            absent so all project logs share one configurable root.
    """
    if name == _LOGGER_NAMESPACE or name.startswith(f"{_LOGGER_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    if isinstance(resolved, int):
        return resolved
    return logging.INFO
