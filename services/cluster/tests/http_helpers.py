"""Shared HTTP-test helpers: state reset + upload bodies (builders only)."""

from __future__ import annotations

from cluster import server
from cluster.adapter_format import LoRAAdapter
from cluster.schemas.messages import TensorPayload


def reset_server_state() -> tuple[dict, dict]:
    """Empty the in-memory service state in place; returns what to restore."""
    default = server._state
    saved = (dict(server._clusters), dict(server._membership))
    default.uploads.clear()
    default.round_id, default.active, default.last_manifest = 0, None, None
    default.last_round.clear()
    server._clusters.clear()
    server._clusters[server.DEFAULT_CLUSTER_ID] = default
    server._membership.clear()
    server.configure_identity(None)
    server.configure_snapshot_sink(None)
    return saved


def restore_server_state(saved: tuple[dict, dict]) -> None:
    server._clusters.clear()
    server._clusters.update(saved[0])
    server._membership.clear()
    server._membership.update(saved[1])
    server.configure_identity(None)
    server.configure_snapshot_sink(None)


def upload_body(client_id, adapter: LoRAAdapter, round_id=0, n=10, **extra):
    return {
        "client_id": client_id,
        "round_id": round_id,
        "rank": adapter.rank,
        "target_modules": list(adapter.target_modules),
        "alpha": adapter.alpha,
        "num_layers": adapter.num_layers,
        "num_examples": n,
        "tensors": [
            TensorPayload.from_numpy(k, v).model_dump() for k, v in adapter.to_state_dict().items()
        ],
        **extra,
    }
