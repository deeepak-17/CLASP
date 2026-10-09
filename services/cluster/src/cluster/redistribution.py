"""Cluster-LoRA redistribution (Week 4 Tue/Thu, P2).

After aggregation the cluster adapter has to go back to the cluster's clients.
This module owns the three pieces of that hand-back:

  * ``build_broadcast``      LoRAAdapter -> ``ClusterAdapterBroadcast`` (the wire
                             message; PEFT-keyed tensors + adapter_config).
  * ``adapter_from_broadcast``  the inverse, what a receiving client runs — kept
                             here so the round-trip is one tested unit.
  * ``redistribute``         deliver a payload to a list of clients through a
                             caller-supplied ``send`` callable, retrying failed
                             deliveries with exponential backoff and returning a
                             per-client ``DeliveryReport`` (never silently
                             dropping a client).

``send`` is injected (``send(client_id, payload) -> None``, raise on failure) so
the same logic serves the in-process simulation, the HTTP service and a real
Flower/gRPC push without this module knowing about any transport.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from cluster.adapter_format import LoRAAdapter
from cluster.schemas.messages import ClusterAdapterBroadcast, TensorPayload


class RedistributionError(RuntimeError):
    """Raised by ``redistribute(require_all=True)`` when a client never received."""


def build_broadcast(
    adapter: LoRAAdapter,
    *,
    cluster_id: str,
    round_id: int,
    num_clients: int,
    aggregation: str = "svd",
    source_clients: Sequence[str] = (),
    epsilon: float | None = None,
) -> ClusterAdapterBroadcast:
    """``source_clients`` names the contributors (the registry records it);
    ``epsilon`` is the cluster's privacy budget (D7), ``None`` when unknown."""
    return ClusterAdapterBroadcast(
        cluster_id=cluster_id,
        round_id=round_id,
        rank=adapter.rank,
        target_modules=adapter.target_modules,
        alpha=adapter.alpha,
        num_layers=adapter.num_layers,
        num_clients=num_clients,
        source_clients=tuple(source_clients),
        aggregation=aggregation,
        epsilon=epsilon,
        peft_config=adapter.to_peft_config(),
        tensors=[
            TensorPayload.from_numpy(name, arr)
            for name, arr in adapter.to_peft_state_dict().items()
        ],
    )


def adapter_from_broadcast(broadcast: ClusterAdapterBroadcast) -> LoRAAdapter:
    """Decode a received broadcast back into a ``LoRAAdapter`` (client side)."""
    return LoRAAdapter.from_state_dict(
        {t.name: t.to_numpy() for t in broadcast.tensors},
        rank=broadcast.rank,
        alpha=broadcast.alpha,
        target_modules=broadcast.target_modules,
        num_layers=broadcast.num_layers,
    )


@dataclass
class DeliveryReport:
    delivered: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # client_id -> last error
    attempts: dict[str, int] = field(default_factory=dict)

    @property
    def all_delivered(self) -> bool:
        return not self.failed


def redistribute(
    payload: object,
    client_ids: Iterable[str],
    send: Callable[[str, object], None],
    *,
    max_retries: int = 3,
    backoff_s: float = 0.1,
    backoff_factor: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    require_all: bool = False,
) -> DeliveryReport:
    """Send ``payload`` to every client; retry each failed send up to
    ``max_retries`` more times with ``backoff_s * backoff_factor**n`` waits.

    A client that still fails is recorded in ``report.failed`` (and raises
    ``RedistributionError`` at the end when ``require_all``) — it is never
    skipped silently, and one client's failure never blocks the others.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    report = DeliveryReport()
    for client_id in client_ids:
        attempts = 0
        last_error = ""
        while True:
            attempts += 1
            try:
                send(client_id, payload)
            except Exception as exc:  # noqa: BLE001 - any transport error is retried
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts > max_retries:
                    report.failed[client_id] = last_error
                    break
                sleep(backoff_s * (backoff_factor ** (attempts - 1)))
            else:
                report.delivered.append(client_id)
                break
        report.attempts[client_id] = attempts
    if require_all and report.failed:
        raise RedistributionError(
            f"{len(report.failed)} client(s) did not receive the adapter: "
            f"{sorted(report.failed)}"
        )
    return report
