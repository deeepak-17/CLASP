"""Composite storage service (D6): resolve parts, merge, store with provenance.

Split in two so the promote path can be all-or-nothing:

    plan_composite   pure — reads the parts, builds the payload in memory
    store_composite  writes — a new immutable COMPOSITE version (or reuses an
                     identical one), and points `active` at it

Privacy of a composite (D7): it releases both parts, so its epsilon is the
basic-composition bound eps_cluster + eps_client (delta likewise summed). If
either part was trained without DP its epsilon is None, and so is the
composite's — None means "no DP guarantee", never "zero".
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from contracts import (
    DEFAULT_BASE_MODEL,
    AdapterKind,
    AdapterMetadata,
    CompositeProvenance,
    PrivacySpec,
)

from .composite import PartSpec, build_composite, embedded_base_model
from .storage import RegistryStore


class CompositionError(ValueError):
    """The request is well-formed JSON but cannot produce a valid composite."""


@dataclass(frozen=True)
class ComposeRequest:
    cluster_name: str
    client_name: str
    alpha: float
    beta: float
    cluster_version: int | None = None  # None -> the part's active version
    client_version: int | None = None
    base_model: str | None = None  # None -> read from the parts, else the dev default


@dataclass(frozen=True)
class PlannedComposite:
    payload: bytes
    provenance: CompositeProvenance
    cluster: AdapterMetadata
    client: AdapterMetadata
    save_kwargs: dict


def _coefficient(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompositionError(f"{label} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise CompositionError(f"{label} must be finite, got {value!r}")
    return float(value)


def _part_ref(value: object, label: str) -> tuple[str, int | None]:
    """Accept ``"name"`` or ``{"name": ..., "version": ...}``."""
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        version = value.get("version")
        if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
            raise CompositionError(f"{label}.version must be an integer, got {version!r}")
        return value["name"], version
    raise CompositionError(f"{label} must be a name or {{'name': ..., 'version': ...}}")


def parse_compose_request(body: object) -> ComposeRequest:
    if not isinstance(body, dict):
        raise CompositionError("body must be a JSON object")
    for key in ("cluster", "client", "alpha", "beta"):
        if key not in body:
            raise CompositionError(f"missing field: {key}")
    cluster_name, cluster_version = _part_ref(body["cluster"], "cluster")
    client_name, client_version = _part_ref(body["client"], "client")
    base_model = body.get("base_model")
    if base_model is not None and (not isinstance(base_model, str) or not base_model):
        raise CompositionError("base_model must be a non-empty string")
    return ComposeRequest(
        cluster_name=cluster_name, client_name=client_name,
        alpha=_coefficient(body["alpha"], "alpha"), beta=_coefficient(body["beta"], "beta"),
        cluster_version=cluster_version, client_version=client_version, base_model=base_model,
    )


def combined_privacy(a: PrivacySpec, b: PrivacySpec) -> PrivacySpec:
    if a.epsilon is None or b.epsilon is None:
        return PrivacySpec(epsilon=None, delta=a.delta + b.delta)
    return PrivacySpec(epsilon=a.epsilon + b.epsilon, delta=a.delta + b.delta)


def _base_model(requested: str | None, embedded: dict[str, str | None]) -> str:
    """The composite's base model: the one its parts were trained on.

    Parts that embed ``base_model_name_or_path`` must agree with each other and
    with an explicit request; a composite labelled with the wrong base would be
    loaded onto the wrong model. With nothing embedded, the request (or the
    1.3B dev default) stands.
    """
    named = {label: base for label, base in embedded.items() if base}
    if len(set(named.values())) > 1:
        raise CompositionError(
            "parts were trained on different base models: "
            + ", ".join(f"{label} -> {base}" for label, base in named.items())
        )
    found = next(iter(named.values()), None)
    if requested and found and requested != found:
        raise CompositionError(
            f"base_model {requested!r} does not match the parts' embedded base model {found!r}"
        )
    return requested or found or DEFAULT_BASE_MODEL


def _resolve(store: RegistryStore, name: str, version: int | None,
             kind: AdapterKind) -> AdapterMetadata:
    if version is None:
        version = store.get_active(name)
        if version is None:
            store.list_versions(name)  # raises AdapterNotFound for an unknown name
            raise CompositionError(f"{name} has no active version")
    meta = store.get_metadata(name, version)
    if meta.ref.kind is not kind:
        raise CompositionError(
            f"{name} v{version} has kind {meta.ref.kind.value!r}; expected {kind.value!r}"
        )
    return meta


def plan_composite(store: RegistryStore, req: ComposeRequest,
                   target_name: str) -> PlannedComposite:
    """Build the composite payload in memory. Writes nothing.

    Raises AdapterNotFound for unknown parts, KindMismatch if ``target_name``
    already holds non-composite versions, and CompositionError (incl. the
    builder's CompositeError) for anything that cannot merge exactly.
    """
    store.check_kind(target_name, AdapterKind.COMPOSITE)
    cluster = _resolve(store, req.cluster_name, req.cluster_version, AdapterKind.CLUSTER)
    client = _resolve(store, req.client_name, req.client_version, AdapterKind.CLIENT)
    cluster_payload = store.load_payload(cluster.ref.name, cluster.ref.version)
    client_payload = store.load_payload(client.ref.name, client.ref.version)
    try:
        built = build_composite(PartSpec(cluster_payload, cluster.hparams, req.alpha),
                                PartSpec(client_payload, client.hparams, req.beta))
        base_model = _base_model(req.base_model, {
            f"{cluster.ref.name} v{cluster.ref.version}": embedded_base_model(cluster_payload),
            f"{client.ref.name} v{client.ref.version}": embedded_base_model(client_payload),
        })
    except ValueError as e:
        raise CompositionError(str(e)) from e
    provenance = CompositeProvenance(
        cluster_name=cluster.ref.name, cluster_version=cluster.ref.version,
        client_name=client.ref.name, client_version=client.ref.version,
        alpha=req.alpha, beta=req.beta, base_model=base_model,
    )
    save_kwargs = {
        "kind": AdapterKind.COMPOSITE,
        "hparams": built.hparams,
        "privacy": combined_privacy(cluster.privacy, client.privacy),
        "aggregation": None,
        "round": client.round,
        "seed": client.seed,
        "cluster_id": cluster.ref.cluster_id or client.ref.cluster_id,
        "source_clients": (client.ref.name,),
        "composed_from": provenance,
    }
    return PlannedComposite(built.payload, provenance, cluster, client, save_kwargs)


def _existing_identical(store: RegistryStore, name: str, plan: PlannedComposite,
                        sha256: str) -> AdapterMetadata | None:
    if name not in store.list_adapters():
        return None
    for v in reversed(store.list_versions(name)):
        meta = store.get_metadata(name, v)
        if meta.composed_from == plan.provenance and meta.sha256 == sha256:
            return meta
    return None


def store_composite(store: RegistryStore, name: str,
                    plan: PlannedComposite) -> tuple[AdapterMetadata, bool]:
    """Persist ``plan`` as ``name`` and make it active. Returns (metadata, created).

    An identical earlier composite (same provenance, same bytes) is reused
    rather than duplicated, so re-running a promotion is idempotent.
    """
    sha256 = hashlib.sha256(plan.payload).hexdigest()
    existing = _existing_identical(store, name, plan, sha256)
    if existing is not None:
        store.set_active(name, existing.ref.version)
        return existing, False
    return store.save(name, plan.payload, set_active=True, **plan.save_kwargs), True
