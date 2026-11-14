"""FedProx's proximal term in the edge's client training (D2: it stays client-side).

The local objective is task_loss + (mu/2)·||w - w0||², w0 being the trainable
parameters at the start of the round (the client adapter as initialized on top
of the merged cluster layer). These tests pin the gradient arithmetic, where it
enters each training loop, and the CLI default:

* non-DP: added to the accumulated gradient BEFORE clipping, as if it were
  part of the loss;
* DP: added AFTER Opacus has clipped, noised and scaled the gradient (the
  optimizer's step hook), so it is post-processing of the private gradient and
  leaves epsilon untouched. Added any earlier, Opacus would overwrite it.
"""
from __future__ import annotations

import math

import pytest
import torch

from edge import fedprox, train_client
from tests.test_d3_dp import _TinyLM


def _params(*values):
    return [torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)) for v in values]


def test_add_proximal_grad_adds_mu_times_the_displacement():
    params = _params([1.0, 2.0], [[3.0]])
    anchors = [torch.tensor([0.0, 1.0]), torch.tensor([[1.0]])]
    params[0].grad = torch.tensor([0.5, 0.5])
    params[1].grad = torch.tensor([[1.0]])
    term = fedprox.add_proximal_grad(params, anchors, mu=0.1)
    assert torch.allclose(params[0].grad, torch.tensor([0.5 + 0.1, 0.5 + 0.1]))
    assert torch.allclose(params[1].grad, torch.tensor([[1.0 + 0.2]]))
    # (mu/2)·||w - w0||² = 0.05 · (1 + 1 + 4)
    assert term == pytest.approx(0.3)


def test_add_proximal_grad_creates_a_missing_grad():
    params = _params([2.0])
    term = fedprox.add_proximal_grad(params, [torch.tensor([0.0])], mu=0.5)
    assert torch.allclose(params[0].grad, torch.tensor([1.0]))
    assert term == pytest.approx(1.0)


def test_mu_zero_changes_nothing():
    params = _params([2.0])
    params[0].grad = torch.tensor([0.25])
    assert fedprox.add_proximal_grad(params, [torch.tensor([0.0])], mu=0.0) == 0.0
    assert torch.equal(params[0].grad, torch.tensor([0.25]))


def test_anchor_is_a_detached_copy():
    params = _params([1.0])
    anchors = fedprox.snapshot_anchor(params)
    with torch.no_grad():
        params[0].add_(5.0)
    assert anchors[0].item() == 1.0 and not anchors[0].requires_grad


class _SpyAdamW(torch.optim.AdamW):
    """Records the gradient each parameter carries when the update is applied."""

    seen: list = []

    def step(self, closure=None):
        _SpyAdamW.seen.append([p.grad.detach().clone() for g in self.param_groups
                               for p in g["params"] if p.grad is not None])
        return super().step(closure)


def _chunks(n=24, width=6):
    gen = torch.Generator().manual_seed(0)
    return torch.randint(0, _TinyLM.VOCAB, (n, width), generator=gen).tolist()


def _distance(model, start):
    return math.sqrt(sum(float((p.detach() - s).pow(2).sum())
                         for p, s in zip([p for p in model.parameters() if p.requires_grad], start)))


def _run_train(mu, grad_clip=1.0, n=24):
    torch.manual_seed(0)
    model = _TinyLM()
    start = fedprox.snapshot_anchor([p for p in model.parameters() if p.requires_grad])
    chunks = _chunks(n)
    schedule = [[i] for i in range(n)]
    result = train_client.train(model, chunks, schedule, pad_token_id=0, lr=1e-2, grad_accum=1,
                                warmup_ratio=0.1, grad_clip=grad_clip, log_every=5,
                                fedprox_mu=mu)
    return model, start, result


def test_train_with_fedprox_ends_closer_to_where_it_started():
    plain, start, r0 = _run_train(mu=0.0)
    prox, start2, r1 = _run_train(mu=10.0)
    assert _distance(prox, start2) < _distance(plain, start)
    assert r0["fedprox"]["mu"] == 0.0 and r0["fedprox"]["applied"] is None
    assert r1["fedprox"]["mu"] == 10.0
    assert r1["fedprox"]["applied"] == "pre-clip"
    assert r1["fedprox"]["final_prox_term"] > 0


def test_train_adds_the_proximal_gradient_before_clipping(monkeypatch):
    """With a tiny clip and a huge mu, the step gradient must still respect the
    clip: the proximal gradient was inside the clipped vector, not added after."""
    monkeypatch.setattr(torch.optim, "AdamW", _SpyAdamW)
    _SpyAdamW.seen = []
    _run_train(mu=100.0, grad_clip=1e-3, n=6)
    assert len(_SpyAdamW.seen) == 6
    for grads in _SpyAdamW.seen:
        norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
        assert norm <= 1e-3 * (1 + 1e-4)


def _dp_api():
    pytest.importorskip("opacus")
    security = pytest.importorskip("security")
    if not all(hasattr(security, n) for n in ("DPConfig", "make_private", "PrivacyAccountant",
                                              "get_privacy_engine")):
        pytest.skip("installed security package has no DP API yet")


def test_train_dp_adds_the_proximal_gradient_after_the_noise(monkeypatch):
    """What the inner optimizer applies = Opacus's noised gradient + mu·(w - w0).

    ``add_proximal_grad`` is wrapped to record the gradient it found (already
    clipped, noised and scaled — Opacus calls the step hook after all three)
    and the displacement it added. Had it run before ``add_noise``, Opacus would
    have replaced ``p.grad`` and the update would not contain the addition.
    """
    _dp_api()
    monkeypatch.setattr(torch.optim, "AdamW", _SpyAdamW)
    _SpyAdamW.seen = []
    found = []
    real = fedprox.add_proximal_grad

    def spy(params, anchors, mu):
        found.append(([p.grad.detach().clone() for p in params],
                      [(p.detach() - a).clone() for p, a in zip(params, anchors)]))
        return real(params, anchors, mu)

    monkeypatch.setattr(fedprox, "add_proximal_grad", spy)
    torch.manual_seed(0)
    mu = 0.5
    result = train_client.train_dp(_TinyLM(), _chunks(40), lr=1e-2, warmup_ratio=0.1,
                                   log_every=1, epochs=1, batch_size=8, target_epsilon=8.0,
                                   delta=1e-5, max_grad_norm=1.0, max_steps=3, fedprox_mu=mu)
    assert result["optimizer_steps"] == 3
    assert len(found) == len(_SpyAdamW.seen) == 3
    for (before, disp), applied in zip(found, _SpyAdamW.seen):
        for b, d, a in zip(before, disp, applied):
            assert torch.allclose(a, b + mu * d, atol=1e-6)
    # the anchor really is the starting point: by the last step the client has moved
    assert any(float(d.abs().max()) > 0 for d in found[-1][1])
    assert result["fedprox"]["applied"] == "post-noise (Opacus step hook)"
    assert result["fedprox"]["mu"] == mu


def test_fedprox_mu_defaults_to_the_cluster_clients_value():
    args = train_client.build_parser().parse_args(["--client", "web/client-flask"])
    assert args.fedprox_mu == fedprox.DEFAULT_MU == 0.01
