"""Flower client for the cluster service.

Two client flavours, matching the plan:
  * ``DummyClient``  — Week 1/2 skeleton: round-trips tensors untouched (plus
    a deterministic marker increment so the round-trip is observable).
  * ``LoRAClient``   — Week 3: trains a toy-but-real torch LoRA model on its
    own local partition with a FedProx proximal term (Week 2, Tue).

FedProx (Li et al., 2020): local objective = task_loss + (mu/2) * ||w - w_global||^2,
where w are the trainable LoRA parameters and w_global is the adapter received
from the cluster server at the start of the round.

DEMO NOTE (Week 5, gap fix): ``torch`` is only needed by ``ToyLoRAModel`` and
``LoRAClient``. It used to be imported unconditionally at module load, which
meant this whole module — including the torch-free ``DummyClient`` — failed
to import in any environment without torch (e.g. a CI/sandbox without GPU
deps). The import is now optional: ``DummyClient`` and the rest of the
package work with numpy alone, and ``ToyLoRAModel``/``LoRAClient`` are only
defined when torch is actually importable, so the Week 5 demo can fall back
to the DummyClient engine instead of crashing outright.
"""

from __future__ import annotations

import numpy as np
from flwr.client import NumPyClient

from cluster.adapter_format import (
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    TARGET_MODULES,
    LoRAAdapter,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class DummyClient(NumPyClient):
    """Week 1 skeleton client: proves the payload round-trips intact."""

    def __init__(self, client_id: str, num_examples: int = 1):
        self.client_id = client_id
        self.num_examples = num_examples

    def fit(self, parameters: list[np.ndarray], config: dict):
        # Echo the payload with a visible, deterministic change so the server
        # can assert the round actually touched this client.
        bump = float(config.get("bump", 1.0))
        updated = [p + bump for p in parameters]
        return updated, self.num_examples, {"client_id": self.client_id}

    def evaluate(self, parameters: list[np.ndarray], config: dict):
        return 0.0, self.num_examples, {}


if TORCH_AVAILABLE:

    class ToyLoRAModel(torch.nn.Module):
        """Frozen 'base model' + trainable LoRA A/B per target module.

        Each target module is a linear map h <- W0 h + (alpha/r) * B (A h), chained
        in sequence — small enough to train on CPU in tests, structurally identical
        to how the real adapter composes onto attention projections.
        """

        def __init__(
            self,
            dim: int = 32,
            rank: int = DEFAULT_RANK,
            alpha: float = DEFAULT_ALPHA,
            target_modules: tuple[str, ...] = TARGET_MODULES,
            seed: int = 0,
        ):
            super().__init__()
            self.dim = dim
            self.rank = rank
            self.alpha = alpha
            self.target_modules = target_modules
            gen = torch.Generator().manual_seed(seed)
            self.base = torch.nn.ParameterDict()
            self.lora_a = torch.nn.ParameterDict()
            self.lora_b = torch.nn.ParameterDict()
            for name in target_modules:
                w0 = torch.randn(dim, dim, generator=gen) / dim**0.5
                self.base[name] = torch.nn.Parameter(w0, requires_grad=False)
                self.lora_a[name] = torch.nn.Parameter(
                    0.02 * torch.randn(rank, dim, generator=gen)
                )
                self.lora_b[name] = torch.nn.Parameter(torch.zeros(dim, rank))

        @property
        def scaling(self) -> float:
            return self.alpha / self.rank

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for name in self.target_modules:
                h = h @ self.base[name].T + self.scaling * (h @ self.lora_a[name].T) @ self.lora_b[name].T
            return h

        # --- adapter (trainable-params-only) round-trip --------------------------

        def get_adapter(self) -> LoRAAdapter:
            # ToyLoRAModel is a single (conceptual) layer, so everything
            # lives at layer index 0 of the adapter's (layer, module) map.
            modules = {
                0: {
                    name: {
                        "lora_A": self.lora_a[name].detach().numpy().copy(),
                        "lora_B": self.lora_b[name].detach().numpy().copy(),
                    }
                    for name in self.target_modules
                }
            }
            return LoRAAdapter(
                rank=self.rank, alpha=self.alpha,
                target_modules=self.target_modules, num_layers=1, modules=modules,
            )

        def set_adapter(self, adapter: LoRAAdapter) -> None:
            adapter.validate()
            with torch.no_grad():
                for name in self.target_modules:
                    self.lora_a[name].copy_(torch.from_numpy(adapter.modules[0][name]["lora_A"]).float())
                    self.lora_b[name].copy_(torch.from_numpy(adapter.modules[0][name]["lora_B"]).float())

    class LoRAClient(NumPyClient):
        """Week 3 client: real local training with a FedProx proximal term."""

        def __init__(
            self,
            client_id: str,
            model: "ToyLoRAModel",
            data: tuple["torch.Tensor", "torch.Tensor"],
            mu: float = 0.01,
            lr: float = 1e-2,
            local_steps: int = 20,
        ):
            self.client_id = client_id
            self.model = model
            self.x, self.y = data
            self.mu = mu
            self.lr = lr
            self.local_steps = local_steps

        def _trainable(self) -> list["torch.nn.Parameter"]:
            return [p for p in self.model.parameters() if p.requires_grad]

        def train_local(self, global_adapter: LoRAAdapter) -> float:
            """Run local FedProx steps from the given global adapter; return final loss."""
            self.model.set_adapter(global_adapter)
            global_params = [p.detach().clone() for p in self._trainable()]
            opt = torch.optim.SGD(self._trainable(), lr=self.lr)
            loss = torch.tensor(float("nan"))
            for _ in range(self.local_steps):
                opt.zero_grad()
                pred = self.model(self.x)
                loss = torch.nn.functional.mse_loss(pred, self.y)
                prox = sum(
                    ((p - g) ** 2).sum() for p, g in zip(self._trainable(), global_params)
                )
                (loss + 0.5 * self.mu * prox).backward()
                opt.step()
            return float(loss.detach())

        # --- Flower NumPyClient API ----------------------------------------------

        def fit(self, parameters: list[np.ndarray], config: dict):
            global_adapter = LoRAAdapter.from_ndarrays(
                parameters,
                rank=self.model.rank,
                alpha=self.model.alpha,
                target_modules=self.model.target_modules,
            )
            final_loss = self.train_local(global_adapter)
            updated = self.model.get_adapter().to_ndarrays()
            return updated, len(self.x), {"client_id": self.client_id, "loss": final_loss}

        def evaluate(self, parameters: list[np.ndarray], config: dict):
            adapter = LoRAAdapter.from_ndarrays(
                parameters,
                rank=self.model.rank,
                alpha=self.model.alpha,
                target_modules=self.model.target_modules,
            )
            self.model.set_adapter(adapter)
            with torch.no_grad():
                loss = torch.nn.functional.mse_loss(self.model(self.x), self.y)
            return float(loss), len(self.x), {}

else:  # pragma: no cover - exercised only in torch-less envs

    class _RequiresTorch:
        """Placeholder used when torch is not installed.

        Keeps ``from cluster.client import ToyLoRAModel, LoRAClient`` working
        (needed by simulation.py's top-level imports) while giving a clear
        error only if code actually tries to instantiate the real thing.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ToyLoRAModel/LoRAClient require torch, which is not installed "
                "in this environment. Install it with `pip install torch` "
                "(see services/cluster/README.md), or run the demo with "
                "--engine dummy to use the torch-free DummyClient fallback."
            )

    ToyLoRAModel = _RequiresTorch  # type: ignore[assignment,misc]
    LoRAClient = _RequiresTorch  # type: ignore[assignment,misc]
