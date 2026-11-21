"""Regenerate ``legacy_volume/``: a registry volume as the file-based store wrote it.

Run once, with the registry code from before metadata moved into SQLite
(commit 3ae2346 and earlier), from the repository root::

    python services/registry/tests/fixtures/make_legacy_volume.py

It writes ``legacy_volume/adapters/...`` through that store's own ``save``,
``set_active`` and ``record_promotion``, then ``legacy_volume.expected.json``:
what that same store answered for every read, so ``test_migrate`` can check
that the database-backed store answers identically after importing it.

Contents: ``cluster-web`` v1 (cluster), ``flask`` v1 and v2 (client) and
``composite-flask`` v1 composed from cluster-web v1 + flask v2; an ``active``
file per adapter and two PROMOTE lines in ``flask/promotions.jsonl``.
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from contracts import (
    AdapterKind,
    AggregationMethod,
    CompositeProvenance,
    LoRAHyperParams,
    PrivacySpec,
    PromotionAction,
    PromotionDecision,
)
from registry.lineage import build_lineage
from registry.storage import RegistryStore

HERE = Path(__file__).resolve().parent
VOLUME = HERE / "legacy_volume"
EXPECTED = HERE / "legacy_volume.expected.json"


def tiny_safetensors(values: list[float]) -> bytes:
    data = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps({"lora_A": {"dtype": "F32", "shape": [len(values)],
                                    "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


def build(store: RegistryStore) -> None:
    store.save("cluster-web", tiny_safetensors([0.1, 0.2]), kind=AdapterKind.CLUSTER,
               hparams=LoRAHyperParams(rank=8), privacy=PrivacySpec(epsilon=4.0, delta=1e-5),
               aggregation=AggregationMethod.SVD_EXACT, round=1, cluster_id="web",
               source_clients=("flask", "requests"), set_active=True)
    v1 = store.save("flask", tiny_safetensors([0.3, 0.4]), kind=AdapterKind.CLIENT,
                    hparams=LoRAHyperParams(rank=8), cluster_id="web", seed=1)
    v2 = store.save("flask", tiny_safetensors([0.5, 0.6]), kind=AdapterKind.CLIENT,
                    hparams=LoRAHyperParams(rank=8), cluster_id="web", seed=2)
    for meta in (v1, v2):
        store.set_active("flask", meta.ref.version)
        store.record_promotion("flask", PromotionDecision(
            adapter=meta.ref, action=PromotionAction.PROMOTE,
            active_version_after=meta.ref.version, reason=f"fixture promote v{meta.ref.version}"))
    store.save("composite-flask", tiny_safetensors([0.7, 0.8]), kind=AdapterKind.COMPOSITE,
               hparams=LoRAHyperParams(rank=16, lora_alpha=16), cluster_id="web",
               source_clients=("flask",), set_active=True,
               composed_from=CompositeProvenance("cluster-web", 1, "flask", 2, alpha=0.5, beta=1.0))


def expected(store: RegistryStore) -> dict:
    out: dict = {"adapters": store.list_adapters(), "versions": {}, "metadata": {},
                 "active": {}, "promotions": {}, "lineage": {}}
    for name in out["adapters"]:
        out["versions"][name] = store.list_versions(name)
        out["metadata"][name] = {str(v): store.get_metadata(name, v).to_json()
                                 for v in out["versions"][name]}
        out["active"][name] = store.get_active(name)
        out["promotions"][name] = [d.to_json() for d in store.list_promotions(name)]
        out["lineage"][name] = build_lineage(store, name)
    return out


def main() -> None:
    shutil.rmtree(VOLUME, ignore_errors=True)
    store = RegistryStore(VOLUME)
    build(store)
    EXPECTED.write_text(json.dumps(expected(store), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
