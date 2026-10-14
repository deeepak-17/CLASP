"""FedProx's proximal term for the edge's client training (D2: it stays client-side).

FedProx (Li et al., 2020) trains each client on

    task_loss + (mu/2) * ||w - w0||^2

where ``w`` are the trainable parameters and ``w0`` the values they had when
the round started. On the edge that is the client LoRA adapter as initialized
on top of the frozen merged cluster layer (D3): its starting point is exactly
the global model the cluster sent back, so the term keeps each laptop's update
close to it and the cluster's average stays meaningful when clients drift.

The term is added as a gradient, ``mu * (w - w0)``, rather than to the loss:

* without DP it goes in after the backward passes of an optimizer step and
  before clipping, which is the same thing as adding it to the loss;
* under DP-SGD it must go in after Opacus has clipped, noised and scaled the
  per-sample gradients (the optimizer's step hook). Opacus replaces ``p.grad``
  when it adds the noise, so anything added earlier is lost, and anything
  added later is post-processing of a private quantity: it depends only on
  ``w`` and ``w0``, never on the data, so epsilon is unchanged.

``cluster.client.LoRAClient`` uses the same objective and the same default mu
in the cluster's own simulation.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

#: The mu ``cluster.client.LoRAClient`` trains with. 0 switches FedProx off.
DEFAULT_MU = 0.01


def snapshot_anchor(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    """``w0``: detached copies of the trainable parameters, taken at round start."""
    return [p.detach().clone() for p in params]


def add_proximal_grad(params: Sequence[torch.nn.Parameter], anchors: Sequence[torch.Tensor],
                      mu: float) -> float:
    """Add ``mu * (w - w0)`` to each parameter's gradient; return ``(mu/2)·||w - w0||²``."""
    if mu == 0:
        return 0.0
    if len(params) != len(anchors):
        raise ValueError(f"{len(params)} parameters but {len(anchors)} anchors")
    total = 0.0
    with torch.no_grad():
        for p, p0 in zip(params, anchors):
            d = p.detach() - p0
            total += float(d.pow(2).sum())
            if p.grad is None:
                p.grad = mu * d
            else:
                p.grad.add_(d, alpha=mu)
    return 0.5 * mu * total
