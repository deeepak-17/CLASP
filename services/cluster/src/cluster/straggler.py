"""Straggler policy (Week 7 Thu, P2): timeout-skip + sample-weighting.

A federated round should not block on, or be corrupted by, clients that are
too slow or that return nothing usable. ``apply_policy`` takes the updates that
actually arrived for a round and decides, deterministically:

  * which updates are *accepted* (finished within ``timeout_s`` and carry data),
  * which are *skipped*, and why (``"timeout"`` / ``"no_examples"``),
  * the aggregation weight of each accepted update (``"samples"`` = FedAvg-style
    ``num_examples`` weighting, or ``"uniform"``),
  * whether the round still has a *quorum* (enough accepted clients relative to
    the number that were expected).

It is pure bookkeeping — no aggregation math lives here. The aggregation core
(``cluster.aggregation``) already takes ``(adapters, weights)``; this module
only decides which adapters/weights are handed to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cluster.adapter_format import LoRAAdapter


class QuorumNotMetError(RuntimeError):
    """Too few usable client updates arrived to aggregate this round."""


@dataclass(frozen=True)
class StragglerPolicy:
    """How a round treats slow / empty clients.

    ``timeout_s=None`` disables the timeout (nobody is a straggler by time).
    Quorum is ``max(min_clients, ceil(min_fraction * expected))`` accepted
    updates; the defaults accept any single usable update, so a policy only
    changes behavior once the caller opts in.
    """

    timeout_s: float | None = None
    min_clients: int = 1
    min_fraction: float = 0.0
    weighting: str = "samples"  # "samples" | "uniform"

    def __post_init__(self) -> None:
        if self.weighting not in ("samples", "uniform"):
            raise ValueError(f"unknown weighting {self.weighting!r}")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive or None")
        if self.min_clients < 1:
            raise ValueError("min_clients must be >= 1")
        if not 0.0 <= self.min_fraction <= 1.0:
            raise ValueError("min_fraction must be within [0, 1]")

    def required(self, expected: int) -> int:
        return max(self.min_clients, math.ceil(self.min_fraction * expected))


@dataclass
class ClientUpdate:
    """One client's contribution to a round, as received by the cluster."""

    client_id: str
    adapter: LoRAAdapter
    num_examples: int
    duration_s: float | None = None
    loss: float | None = None


@dataclass
class PolicyOutcome:
    accepted: list[ClientUpdate]
    weights: list[float]
    skipped: dict[str, str] = field(default_factory=dict)  # client_id -> reason
    expected: int = 0
    required: int = 1

    @property
    def quorum_met(self) -> bool:
        return len(self.accepted) >= self.required


def apply_policy(
    updates: list[ClientUpdate],
    policy: StragglerPolicy | None = None,
    expected: int | None = None,
) -> PolicyOutcome:
    """Filter ``updates`` through ``policy``; ``expected`` defaults to len(updates).

    Pass ``expected`` = the number of clients the round *asked* (so clients that
    never answered at all still count against the quorum fraction).
    """
    policy = policy or StragglerPolicy()
    expected = len(updates) if expected is None else expected
    accepted: list[ClientUpdate] = []
    skipped: dict[str, str] = {}
    for upd in updates:
        if upd.num_examples <= 0:
            skipped[upd.client_id] = "no_examples"
        elif (
            policy.timeout_s is not None
            and upd.duration_s is not None
            and upd.duration_s > policy.timeout_s
        ):
            skipped[upd.client_id] = "timeout"
        else:
            accepted.append(upd)
    if policy.weighting == "samples":
        weights = [float(u.num_examples) for u in accepted]
    else:
        weights = [1.0 for _ in accepted]
    return PolicyOutcome(
        accepted=accepted,
        weights=weights,
        skipped=skipped,
        expected=expected,
        required=policy.required(expected),
    )
