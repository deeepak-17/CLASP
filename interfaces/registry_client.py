"""P4 (State Registry) boundary — snapshot-read protocol plus a local mock.

**P5 does not implement the Registry.** P4 (Deepak) owns the FastAPI service,
``safetensors`` versioning and rollback. P5 is a *reader*: from Week 7 the
eval harness auto-triggers on "the latest Registry snapshot".

In Weeks 1-2 the Registry does not exist yet, so this module supplies:

* :class:`RegistryReadClient` — the read-side ``Protocol`` P5 depends on. P5
  deliberately does **not** declare ``save``/``rollback``; narrowing the
  interface to what P5 actually calls keeps the coupling minimal.
* :class:`MockRegistryClient` — an in-memory / on-disk fake, so the Week-2
  contract sanity-check can be run today against realistic snapshot metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from interfaces.contracts import AdapterKind, AdapterRef, SnapshotMetadata
from utils.errors import ClaspP5Error, ContractViolationError
from utils.io_utils import read_json
from utils.logging_utils import get_logger
from utils.timing import utc_timestamp

_LOG = get_logger(__name__)


class SnapshotNotFoundError(ClaspP5Error):
    """No snapshot matches the requested adapter/version."""


@runtime_checkable
class RegistryReadClient(Protocol):
    """The read-only slice of P4's Registry API that P5 consumes.

    Maps onto P4's Week-3 endpoints: ``list`` and ``load``.
    """

    def list_versions(self, name: str, kind: AdapterKind) -> Sequence[SnapshotMetadata]:
        """Return every snapshot for an adapter, oldest first."""
        ...

    def latest(self, name: str, kind: AdapterKind) -> SnapshotMetadata:
        """Return the newest snapshot for an adapter.

        Raises:
            SnapshotNotFoundError: If the adapter has no snapshots.
        """
        ...

    def load_metadata(self, ref: AdapterRef) -> SnapshotMetadata:
        """Return the metadata sidecar for one exact adapter version."""
        ...


class MockRegistryClient:
    """Offline stand-in for P4's Registry.

    Two ways to populate it:

    * programmatically, via :meth:`add_snapshot` (used by the test suite);
    * from disk, via :meth:`from_directory`, which reads ``*.json`` sidecars
      laid out the way P4's design note describes. That path is what proves
      P5 can parse P4's real output the day it lands.
    """

    def __init__(self, snapshots: Sequence[SnapshotMetadata] = ()) -> None:
        self._snapshots: list[SnapshotMetadata] = list(snapshots)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_directory(cls, directory: Path | str) -> "MockRegistryClient":
        """Build a client from a directory of ``metadata.json`` sidecars."""
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise ClaspP5Error(f"Mock registry directory not found: {root}")

        snapshots: list[SnapshotMetadata] = []
        for path in sorted(root.rglob("*.json")):
            payload = read_json(path)
            try:
                snapshots.append(SnapshotMetadata.from_dict(payload))
            except ContractViolationError as exc:
                _LOG.warning("Skipping non-snapshot JSON %s: %s", path.name, exc)
        _LOG.info("MockRegistryClient loaded %d snapshot(s) from %s", len(snapshots), root)
        return cls(snapshots)

    def add_snapshot(
        self,
        ref: AdapterRef,
        *,
        round_number: int,
        artifact_path: str,
        artifact_sha256: str | None = None,
        training_loss: float | None = None,
        eval_score: float | None = None,
        created_at: str | None = None,
    ) -> SnapshotMetadata:
        """Register a synthetic snapshot and return it."""
        snapshot = SnapshotMetadata(
            adapter=ref,
            round_number=round_number,
            created_at=created_at or utc_timestamp(),
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            training_loss=training_loss,
            eval_score=eval_score,
        )
        self._snapshots.append(snapshot)
        return snapshot

    # --- RegistryReadClient ------------------------------------------------
    def list_versions(self, name: str, kind: AdapterKind) -> Sequence[SnapshotMetadata]:
        matches = [s for s in self._snapshots if s.adapter.name == name and s.adapter.kind is kind]
        return sorted(matches, key=lambda s: s.adapter.version)

    def latest(self, name: str, kind: AdapterKind) -> SnapshotMetadata:
        versions = self.list_versions(name, kind)
        if not versions:
            raise SnapshotNotFoundError(f"No snapshots for {kind.value}/{name}")
        return versions[-1]

    def load_metadata(self, ref: AdapterRef) -> SnapshotMetadata:
        for snapshot in self._snapshots:
            if snapshot.adapter == ref:
                return snapshot
        raise SnapshotNotFoundError(f"No snapshot for {ref.uri}")

    def __len__(self) -> int:
        return len(self._snapshots)
