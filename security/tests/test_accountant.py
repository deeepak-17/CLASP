"""Tests for security.dp.accountant — RDP-based privacy accountant."""
from __future__ import annotations

import pytest

from security.dp.config import DPConfig

# Accountant tests that need Opacus are guarded — they skip gracefully if
# opacus is not installed (CI may not have GPU deps).
opacus = pytest.importorskip("opacus", reason="opacus required for accountant tests")

from security.dp.accountant import BudgetExhaustedError, PrivacyAccountant  # noqa: E402 — after the importorskip guard


class TestPrivacyAccountantBasic:
    """Core accountant behaviour: step, epsilon, budget."""

    def _make(self, **overrides) -> PrivacyAccountant:
        # Use a realistic sample_rate (1/60 ≈ batch_size=1 over 60 blocks)
        # and a generous target_epsilon so basic tests don't hit the budget cap.
        defaults = {
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "target_epsilon": 50.0,  # high cap so basic tests don't exhaust
            "delta": 1e-5,
        }
        defaults.update(overrides)
        cfg = DPConfig(**defaults)
        sample_rate = overrides.pop("_sample_rate", 0.016)
        return PrivacyAccountant(cfg, sample_rate=sample_rate)

    def test_initial_epsilon_is_zero(self) -> None:
        acc = self._make()
        assert acc.get_epsilon() == 0.0

    def test_step_increases_epsilon(self) -> None:
        acc = self._make()
        e1 = acc.step(num_steps=10)
        assert e1 > 0.0

    def test_more_steps_more_epsilon(self) -> None:
        acc = self._make()
        e10 = acc.step(num_steps=10)
        e20 = acc.step(num_steps=10)  # now 20 total
        assert e20 > e10

    def test_budget_remaining_decreases(self) -> None:
        acc = self._make(target_epsilon=50.0)
        initial = acc.budget_remaining
        acc.step(num_steps=5)
        assert acc.budget_remaining < initial

    def test_is_exhausted_false_initially(self) -> None:
        acc = self._make()
        assert acc.is_exhausted is False


class TestBudgetExhaustion:
    """Budget enforcement: BudgetExhaustedError when ε exceeds target."""

    def test_exhaustion_raises(self) -> None:
        # Very tight budget with high sample_rate should exhaust quickly.
        cfg = DPConfig(noise_multiplier=0.1, max_grad_norm=1.0,
                       target_epsilon=0.01, delta=1e-5)
        acc = PrivacyAccountant(cfg, sample_rate=1.0)
        with pytest.raises(BudgetExhaustedError):
            acc.step(num_steps=10000)


class TestPrivacySpecConversion:
    """get_privacy_spec() returns a valid PrivacySpec from contracts."""

    def test_spec_after_steps(self) -> None:
        cfg = DPConfig(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5,
                       target_epsilon=50.0)
        acc = PrivacyAccountant(cfg, sample_rate=0.016)
        acc.step(num_steps=10)
        spec = acc.get_privacy_spec()
        assert spec.epsilon is not None
        assert spec.epsilon > 0
        assert spec.delta == 1e-5
        assert spec.noise_multiplier == 1.0
        assert spec.max_grad_norm == 1.0

    def test_spec_zero_steps(self) -> None:
        cfg = DPConfig(target_epsilon=50.0)
        acc = PrivacyAccountant(cfg, sample_rate=0.016)
        spec = acc.get_privacy_spec()
        # Zero steps means no privacy cost yet.
        assert spec.epsilon == 0.0


class TestComputeEpsilon:
    """Static epsilon computation for known parameters."""

    def test_more_steps_more_epsilon(self) -> None:
        cfg = DPConfig(noise_multiplier=1.0, delta=1e-5, target_epsilon=50.0)
        acc = PrivacyAccountant(cfg, sample_rate=0.016)
        e10 = acc.compute_epsilon(steps=10, sample_rate=0.01)
        e100 = acc.compute_epsilon(steps=100, sample_rate=0.01)
        assert e100 > e10

    def test_more_noise_less_epsilon(self) -> None:
        cfg_lo = DPConfig(noise_multiplier=0.5, delta=1e-5, target_epsilon=50.0)
        cfg_hi = DPConfig(noise_multiplier=2.0, delta=1e-5, target_epsilon=50.0)
        acc_lo = PrivacyAccountant(cfg_lo, sample_rate=0.016)
        acc_hi = PrivacyAccountant(cfg_hi, sample_rate=0.016)
        e_lo = acc_lo.compute_epsilon(steps=50, sample_rate=0.01)
        e_hi = acc_hi.compute_epsilon(steps=50, sample_rate=0.01)
        assert e_hi < e_lo  # more noise → less epsilon (more private)
