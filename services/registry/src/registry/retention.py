"""Retention / GC policy: keep the newest N versions plus everything ever live.

A version survives GC if any of these hold (all are reported, so an operator
can see why each survivor survived):

    active     the active version right now
    recent     among the newest ``keep_last`` versions
    promoted   a D5 decision promoted it (it served traffic)
    restored   a rollback/restore made it active (it served traffic)
    composite  a stored composite is built from it— deleting it would orphan
               the composite's lineage and make it unreproducible (D9)

Everything else is deleted, oldest first. GC is opt-in and defaults to a dry
run. The newest version is always protected, so version numbers stay
monotonic: a deleted number is never handed out again.

``sweep_orphans`` removes files no database row accounts for: a ``vN``
directory without its ``versions`` row (a write whose transaction never
committed, or a delete that dropped the row and then crashed) and staging or
trash directories left by an interrupted save or delete. It runs once when
the app creates its store and at the start of every non-dry-run GC call; a
dry run changes nothing, orphans included.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from contracts import PromotionAction, PromotionDecision, utcnow_iso

from .storage import RegistryStore

#: Default for ``keep_last`` when neither the request nor the env sets it.
DEFAULT_KEEP_LAST = 5
KEEP_LAST_ENV = "CLASP_REGISTRY_KEEP_LAST"

#: A staging or trash directory younger than this may belong to a save or
#: delete in progress, so the sweep leaves it alone.
SCRATCH_MIN_AGE_SECONDS = 60
_SCRATCH_PREFIXES = (".tmp-v", ".trash-")
_VERSION_DIR = re.compile(r"v([1-9][0-9]*)")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionPlan:
    keep: dict[int, tuple[str, ...]]  # version -> why it survives
    delete: tuple[int, ...]


def default_keep_last() -> int:
    raw = os.environ.get(KEEP_LAST_ENV)
    if raw is None:
        return DEFAULT_KEEP_LAST
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{KEEP_LAST_ENV}={raw!r} is not an integer") from e
    if value < 1:
        raise ValueError(f"{KEEP_LAST_ENV} must be >= 1, got {value}")
    return value


def plan_retention(
    versions: Iterable[int],
    *,
    active: int | None,
    decisions: Iterable[PromotionDecision],
    referenced: set[int],
    keep_last: int,
) -> RetentionPlan:
    """Pure policy: which versions to keep (and why) and which to delete."""
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    ordered = sorted(versions)
    recent = set(ordered[-keep_last:])
    # active_version_after is the version that went live: for a D5 PROMOTE it is
    # the candidate itself, for an operator restore forward it is the target.
    promoted = {d.active_version_after for d in decisions
                if d.action is PromotionAction.PROMOTE}
    restored = {d.active_version_after for d in decisions
                if d.action is PromotionAction.ROLLBACK}

    keep: dict[int, tuple[str, ...]] = {}
    delete: list[int] = []
    for v in ordered:
        reasons = tuple(label for label, hit in (
            ("active", v == active),
            ("recent", v in recent),
            ("promoted", v in promoted),
            ("restored", v in restored),
            ("composite", v in referenced),
        ) if hit)
        if reasons:
            keep[v] = reasons
        else:
            delete.append(v)
    return RetentionPlan(keep=keep, delete=tuple(delete))


def collect_referenced(store: RegistryStore) -> set[tuple[str, int]]:
    """Every stored (adapter, version) that some composite is built from."""
    return {(name, v) for name in store.list_adapters()
            for v in store.repo.referenced_versions(name)}


def _is_orphan(entry: Path, registered: set[int], now: float) -> bool:
    m = _VERSION_DIR.fullmatch(entry.name)
    if m:  # a published directory without a row is never valid, whatever its age
        return int(m.group(1)) not in registered
    return (entry.name.startswith(_SCRATCH_PREFIXES)
            and now - entry.stat().st_mtime > SCRATCH_MIN_AGE_SECONDS)


def sweep_orphans(store: RegistryStore) -> list[str]:
    """Remove every file tree no database row accounts for; return the removed paths.

    Never touches a registered version or a staging/trash directory younger
    than ``SCRATCH_MIN_AGE_SECONDS``. A ``vN`` directory is first renamed to a
    trash name (atomic), so its number is free at once even if the removal
    is interrupted; the trash is then swept on a later run.
    """
    now = time.time()
    removed: list[str] = []
    for adapter_dir in sorted(store.adapters_dir.iterdir()):
        if not adapter_dir.is_dir() or adapter_dir.name.startswith("."):
            continue
        registered = set(store.repo.list_versions(adapter_dir.name))
        for entry in sorted(adapter_dir.iterdir()):
            if not entry.is_dir() or not _is_orphan(entry, registered, now):
                continue
            target = entry
            if _VERSION_DIR.fullmatch(entry.name):
                target = Path(tempfile.mkdtemp(prefix=f".trash-{entry.name}-", dir=adapter_dir))
                os.replace(entry, target / entry.name)
            shutil.rmtree(target, ignore_errors=True)
            if not entry.exists():  # a scratch dir that resisted removal is retried next run
                _log.warning("swept orphaned %s", entry)
                removed.append(str(entry))
    return removed


def gc_adapter(
    store: RegistryStore,
    name: str,
    *,
    keep_last: int,
    dry_run: bool = True,
    referenced: set[tuple[str, int]] | None = None,
) -> RetentionPlan:
    """Plan (and unless ``dry_run``, apply) retention for one adapter."""
    versions = store.list_versions(name)  # AdapterNotFound for an unknown name
    plan = plan_retention(
        versions,
        active=store.get_active(name),
        decisions=store.list_promotions(name),
        referenced=(store.repo.referenced_versions(name) if referenced is None
                    else {v for n, v in referenced if n == name}),
        keep_last=keep_last,
    )
    if dry_run or not plan.delete:
        return plan
    for v in plan.delete:
        store.delete_version(name, v)
    store.append_log(name, "gc.jsonl", json.dumps({
        "timestamp": utcnow_iso(),
        "keep_last": keep_last,
        "deleted": list(plan.delete),
        "kept": {str(v): list(r) for v, r in plan.keep.items()},
    }))
    return plan
