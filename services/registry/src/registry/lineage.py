"""Lineage view: every version of an adapter with where it came from (P4).

A version's *parents* are read from its immutable metadata, never inferred:

    cluster    -> the clients that were aggregated into it  ("client:<id>")
    composite  -> the exact cluster and client versions merged ("<name>@v<n>")
    client     -> none recorded (trained locally from the base)

Each version also carries the promotion/rollback decisions that targeted it,
so one GET answers "where did this come from and what happened to it".
"""
from __future__ import annotations

from contracts import AdapterKind, AdapterMetadata

from .storage import RegistryStore


def parents_of(meta: AdapterMetadata) -> list[str]:
    if meta.ref.kind is AdapterKind.COMPOSITE and meta.composed_from is not None:
        prov = meta.composed_from
        return [
            f"{prov.cluster_name}@v{prov.cluster_version}",
            f"{prov.client_name}@v{prov.client_version}",
        ]
    if meta.ref.kind is AdapterKind.CLUSTER:
        return [f"client:{c}" for c in meta.source_clients]
    return []


def build_lineage(store: RegistryStore, name: str) -> dict:
    """The lineage document for ``name``. Raises AdapterNotFound if unknown."""
    versions = store.list_versions(name)
    active = store.get_active(name)
    decisions = store.list_promotions(name)

    entries = []
    for v in versions:
        meta = store.get_metadata(name, v)
        entries.append({
            "version": v,
            "kind": meta.ref.kind.value,
            "is_active": v == active,
            "created_at": meta.created_at,
            "round": meta.round,
            "sha256": meta.sha256,
            "num_bytes": meta.num_bytes,
            "aggregation": meta.aggregation.value if meta.aggregation else None,
            "epsilon": meta.privacy.epsilon,
            "parents": parents_of(meta),
            "composed_from": meta.composed_from.to_json() if meta.composed_from else None,
            "decisions": [d.to_json() for d in decisions if d.adapter.version == v],
        })
    return {"name": name, "active": active, "versions": entries}
