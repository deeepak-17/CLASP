"""Registry storage exceptions (shared by storage and the metadata repository)."""

from __future__ import annotations


class StorageError(Exception):
    """Base class for registry storage errors."""


class AdapterNotFound(StorageError):
    pass


class KindMismatch(StorageError):
    """A name's versions all share one AdapterKind; a save tried to change it."""


class InvalidAdapterName(StorageError):
    """Name fails the path-safe pattern — a client error, never a 500."""


class VersionExists(StorageError):
    """Refused to overwrite an existing immutable version (D9)."""
