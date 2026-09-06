"""Shared CLI plumbing for the ``scripts/`` entry points.

Centralises the three things every entry point needs and none should
reimplement: a common argument set, logging configuration, and a top-level
error boundary that turns a :class:`~utils.errors.ClaspP5Error` into a clean
message plus a non-zero exit code instead of a traceback.

Exit codes
----------
``0`` success · ``1`` expected failure (bad config, failed validation) ·
``2`` argparse usage error · ``130`` interrupted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

import _bootstrap  # noqa: F401  (path side effect must happen before project imports)

from utils.config import load_logging_config
from utils.errors import ClaspP5Error
from utils.logging_utils import configure_from_mapping, get_logger

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130

_LOG = get_logger(__name__)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser preloaded with the arguments every CLI accepts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the level from configs/logging.yaml.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Override the rotating log file path.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the Markdown report to reports/.",
    )
    return parser


def setup_logging(args: argparse.Namespace) -> None:
    """Configure logging from config, applying any CLI overrides."""
    config = load_logging_config()
    mapping = config.as_mapping()
    if getattr(args, "log_level", None):
        mapping["level"] = args.log_level
    if getattr(args, "log_file", None):
        mapping["log_file"] = str(args.log_file)
    configure_from_mapping(mapping, force=True)


def run_cli(
    main: Callable[[argparse.Namespace], int],
    args: argparse.Namespace,
) -> int:
    """Invoke ``main`` inside the standard error boundary."""
    try:
        return main(args)
    except ClaspP5Error as exc:
        _LOG.error("%s: %s", type(exc).__name__, exc)
        return EXIT_FAILURE
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        _LOG.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except Exception:  # pragma: no cover - genuinely unexpected
        _LOG.exception("Unhandled error")
        return EXIT_FAILURE


def emit(lines: Sequence[str]) -> None:
    """Print a human-facing summary block to stdout.

    Logs go to stderr; the summary a person reads goes to stdout, so a CLI can
    be piped without its log stream contaminating the output.
    """
    for line in lines:
        print(line, file=sys.stdout)
    sys.stdout.flush()


def report_line(path: Path) -> str:
    """Format a 'report written' line relative to the repo root."""
    from utils.paths import project_paths

    return f"  report:   {project_paths().relative(path)}"
