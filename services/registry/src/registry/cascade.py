"""Carry a part's rollback or restore through to the composites the edge serves.

Under D6 the edge serves a pre-merged composite, not the client or cluster
adapter. Moving a part's ``active`` pointer alone would leave every composite
built from the old version live, so a D5 rollback or an operator restore would
report success while the edge kept serving the bad weights.

When part ``P`` moves from ``v_from`` to ``v_to``, every composite whose
*active* version was composed from ``P@v_from`` is repointed at a composite of
``P@v_to`` with the same other part, alpha, beta and base model:

    reuse     an existing composite version with exactly that provenance
    rebuild   otherwise, build one now (a new immutable version)

Composites pinned to other versions of ``P`` are left alone. ``plan_cascade``
writes nothing and fails before any write if a rebuild cannot be built, so the
part and its composites always move together.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from contracts import (
    AdapterKind,
    AdapterMetadata,
    CompositeProvenance,
    PromotionAction,
    PromotionDecision,
)

from .composition import ComposeRequest, PlannedComposite, plan_composite, store_composite
from .storage import RegistryStore


@dataclass(frozen=True)
class CascadeStep:
    composite: str
    from_version: int
    provenance: CompositeProvenance
    reuse_version: int | None = None  # set -> repoint; None -> store ``plan``
    plan: PlannedComposite | None = None


def _swap(prov: CompositeProvenance, part: str, from_v: int,
          to_v: int) -> CompositeProvenance | None:
    if (prov.cluster_name, prov.cluster_version) == (part, from_v):
        return replace(prov, cluster_version=to_v)
    if (prov.client_name, prov.client_version) == (part, from_v):
        return replace(prov, client_version=to_v)
    return None


def _active_composite(store: RegistryStore, name: str) -> AdapterMetadata | None:
    active = store.get_active(name)
    if active is None:
        return None
    meta = store.get_metadata(name, active)
    if meta.ref.kind is not AdapterKind.COMPOSITE or meta.composed_from is None:
        return None
    return meta


def _find(store: RegistryStore, name: str, prov: CompositeProvenance) -> int | None:
    for v in reversed(store.list_versions(name)):
        if store.get_metadata(name, v).composed_from == prov:
            return v
    return None


def _request(prov: CompositeProvenance) -> ComposeRequest:
    return ComposeRequest(
        cluster_name=prov.cluster_name, client_name=prov.client_name,
        alpha=prov.alpha, beta=prov.beta, cluster_version=prov.cluster_version,
        client_version=prov.client_version, base_model=prov.base_model,
    )


def plan_cascade(store: RegistryStore, part: str, from_version: int,
                 to_version: int) -> list[CascadeStep]:
    """Every composite move needed when ``part`` goes ``from_version -> to_version``.

    Writes nothing. Raises CompositionError if a needed rebuild cannot be built.
    """
    steps = []
    for name in store.list_adapters():
        meta = _active_composite(store, name)
        if meta is None:
            continue
        target = _swap(meta.composed_from, part, from_version, to_version)
        if target is None:
            continue
        reuse = _find(store, name, target)
        plan = None if reuse is not None else plan_composite(store, _request(target), name)
        steps.append(CascadeStep(name, meta.ref.version, target, reuse, plan))
    return steps


def apply_cascade(store: RegistryStore, steps: list[CascadeStep], *,
                  action: PromotionAction, reason: str) -> list[dict]:
    """Repoint (or build) each composite and record the move in its audit trail."""
    moved = []
    for step in steps:
        if step.reuse_version is not None:
            store.set_active(step.composite, step.reuse_version)
            version, rebuilt = step.reuse_version, False
        else:
            meta, rebuilt = store_composite(store, step.composite, step.plan)
            version = meta.ref.version
        candidate = store.get_metadata(step.composite, step.from_version).ref
        store.record_promotion(step.composite, PromotionDecision(
            adapter=candidate, action=action, active_version_after=version,
            reason=f"cascade v{step.from_version} -> v{version}: {reason}",
        ))
        moved.append({"name": step.composite, "from_version": step.from_version,
                      "to_version": version, "rebuilt": rebuilt,
                      "composed_from": step.provenance.to_json()})
    return moved
