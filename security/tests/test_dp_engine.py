"""Tests for security.dp.engine — make_private() Opacus wrapper."""
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from security.dp.config import DPConfig

# Guard: skip the entire module if opacus is not installed.
opacus = pytest.importorskip("opacus", reason="opacus required for engine tests")

from security.dp.engine import get_privacy_engine, make_private


def _tiny_model() -> nn.Module:
    """A minimal model that Opacus can instrument."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


def _tiny_loader(n: int = 32, dim: int = 8) -> DataLoader:
    """A tiny DataLoader for testing."""
    x = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestMakePrivateCalibrated:
    """With epochs=, sigma is calibrated so the run stays within target_epsilon (D7)."""

    def test_epochs_path_calibrates_sigma(self) -> None:
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = DPConfig(enabled=True, max_grad_norm=1.0, target_epsilon=8.0, delta=1e-5)

        model_dp, opt_dp, _ = make_private(model, optimizer, _tiny_loader(), cfg, epochs=2)
        assert opt_dp.noise_multiplier > 0
        assert opt_dp.noise_multiplier != cfg.noise_multiplier  # calibrated, not the default
        assert get_privacy_engine(model_dp) is not None

    def test_epochs_path_spends_at_most_the_target(self) -> None:
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = _tiny_loader()
        cfg = DPConfig(enabled=True, max_grad_norm=1.0, target_epsilon=8.0, delta=1e-5)
        epochs = 2

        model_dp, opt_dp, loader_dp = make_private(model, optimizer, loader, cfg, epochs=epochs)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for x, y in loader_dp:
                if len(x) == 0:  # Poisson sampling can draw an empty batch
                    continue
                opt_dp.zero_grad()
                loss_fn(model_dp(x), y).backward()
                opt_dp.step()
        spent = get_privacy_engine(model_dp).get_epsilon(cfg.delta)
        assert 0 < spent <= cfg.target_epsilon + 1e-6


class TestMakePrivateEnabled:
    """When DP is enabled, make_private wraps model and optimizer."""

    def test_returns_wrapped_objects(self) -> None:
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = _tiny_loader()
        cfg = DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0)

        model_dp, opt_dp, loader_dp = make_private(model, optimizer, loader, cfg)
        # The model should now be a GradSampleModule or wrapped.
        assert model_dp is not None
        assert opt_dp is not None
        assert loader_dp is not None

    def test_privacy_engine_attached(self) -> None:
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = _tiny_loader()
        cfg = DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0)

        model_dp, _, _ = make_private(model, optimizer, loader, cfg)
        engine = get_privacy_engine(model_dp)
        assert engine is not None

    def test_training_step_works(self) -> None:
        """A full forward-backward-step should succeed without errors."""
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = _tiny_loader(n=16)
        cfg = DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0)

        model_dp, opt_dp, loader_dp = make_private(model, optimizer, loader, cfg)
        model_dp.train()
        criterion = nn.CrossEntropyLoss()

        for batch in loader_dp:
            x, y = batch
            opt_dp.zero_grad()
            out = model_dp(x)
            loss = criterion(out, y)
            loss.backward()
            opt_dp.step()
            break  # one step is enough to prove the machinery works


class TestMakePrivateDisabled:
    """When DP is disabled, make_private is a pass-through."""

    def test_returns_same_objects(self) -> None:
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = _tiny_loader()
        cfg = DPConfig(enabled=False)

        model_out, opt_out, loader_out = make_private(model, optimizer, loader, cfg)
        assert model_out is model
        assert opt_out is optimizer
        assert loader_out is loader

    def test_no_privacy_engine(self) -> None:
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = _tiny_loader()
        cfg = DPConfig(enabled=False)

        model_out, _, _ = make_private(model, optimizer, loader, cfg)
        engine = get_privacy_engine(model_out)
        assert engine is None


class TestGradientNoising:
    """Statistical test: DP-SGD should add noise to gradients."""

    def test_gradients_differ_with_noise(self) -> None:
        """Train two identical models — one with DP, one without.

        With noise_multiplier > 0, the gradients should differ.
        """
        torch.manual_seed(42)
        model_clean = _tiny_model()
        model_dp = _tiny_model()
        # Make them start identical.
        model_dp.load_state_dict(model_clean.state_dict())

        x = torch.randn(8, 8)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Clean pass
        opt_clean = torch.optim.SGD(model_clean.parameters(), lr=0.01)
        for bx, by in loader:
            opt_clean.zero_grad()
            criterion(model_clean(bx), by).backward()
            opt_clean.step()
            break

        # DP pass
        opt_dp = torch.optim.SGD(model_dp.parameters(), lr=0.01)
        cfg = DPConfig(enabled=True, noise_multiplier=5.0, max_grad_norm=1.0)
        model_dp, opt_dp, loader_dp = make_private(model_dp, opt_dp, loader, cfg)
        model_dp.train()
        for bx, by in loader_dp:
            opt_dp.zero_grad()
            criterion(model_dp(bx), by).backward()
            opt_dp.step()
            break

        # Extract the underlying module from GradSampleModule.
        inner = model_dp._module if hasattr(model_dp, "_module") else model_dp
        # Params should differ because noise was added.
        for p_c, p_d in zip(model_clean.parameters(), inner.parameters()):
            if not torch.allclose(p_c, p_d, atol=1e-6):
                return  # at least one param differs — test passes
        # If ALL params are identical with noise_multiplier=5.0, something is wrong.
        pytest.fail("All parameters identical after DP step — noise may not be applied")
