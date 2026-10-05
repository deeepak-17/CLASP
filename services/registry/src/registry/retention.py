"""Retention / GC policy: keep the newest N versions plus everything ever live.

A version survives GC if any of these hold (all are reported, so an operator
can see why each survivor survived):

    active     the active version right now
    recent     among the newest ``keep_last`` versions
    promoted   a D5 decision promoted it (it served traffic)
    restored   a rollback/restore made it active (it served traffic)
    composite  a composite's provenance names it — deleting it would orphan
               the composite's lineage and make it unreproducible (D9)

Everything else is deleted, oldest first. GC is opt-in and defaults to a dry
run. The newest version is always protected, so version numbers stay
monotonic: a deleted number is never handed out again.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

from contracts import AdapterKind, PromotionAction, PromotionDecision, utcnow_iso

from .storage import RegistryStore

#: Default for ``keep_last`` when neither the request nor the env sets it.
DEFAULT_KEEP_LAST = 5
KEEP_LAST_ENV = "CLASP_REGISTRY_KEEP_LAST"


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
    """Every (adapter, version) named by any stored composite's provenance."""
    refs: set[tuple[str, int]] = set()
    for name in store.list_adapters():
        for v in store.list_versions(name):
            meta = store.get_metadata(name, v)
            prov = meta.composed_from
            if meta.ref.kind is AdapterKind.COMPOSITE and prov is not None:
                refs.add((prov.cluster_name, prov.cluster_version))
                refs.add((prov.client_name, prov.client_version))
    return refs


def gc_adapter(
    store: RegistryStore,
    name: str,
    *,
    keep_last: int,
    dry_run: bool = True,
    referenced: set[tuple[str, int]] | None = None,
) -> RetentionPlan:
    """Plan (and unless ``dry_run``, apply) retention for one adapter."""
    if referenced is None:
        referenced = collect_referenced(store)
    plan = plan_retention(
        store.list_versions(name),
        active=store.get_active(name),
        decisions=store.list_promotions(name),
        referenced={v for n, v in referenced if n == name},
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
