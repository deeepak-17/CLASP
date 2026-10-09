"""Tests for security.dp.config — DPConfig dataclass and PrivacySpec conversion."""
from __future__ import annotations

import pytest

from security.dp.config import DPConfig


class TestDPConfigDefaults:
    """Verify the default DPConfig matches the D7 design document."""

    def test_defaults(self) -> None:
        cfg = DPConfig()
        assert cfg.enabled is True
        assert cfg.noise_multiplier == 1.0
        assert cfg.max_grad_norm == 1.0
        assert cfg.delta == 1e-5
        assert cfg.target_epsilon == 8.0
        assert cfg.secure_mode is False

    def test_custom_values(self) -> None:
        cfg = DPConfig(enabled=False, noise_multiplier=0.5, max_grad_norm=2.0,
                       delta=1e-6, target_epsilon=4.0, secure_mode=True)
        assert cfg.enabled is False
        assert cfg.noise_multiplier == 0.5
        assert cfg.target_epsilon == 4.0


class TestDPConfigToggle:
    """The enabled flag is the master on/off switch for DP."""

    def test_disabled(self) -> None:
        cfg = DPConfig(enabled=False)
        assert cfg.enabled is False

    def test_enabled(self) -> None:
        cfg = DPConfig(enabled=True)
        assert cfg.enabled is True


class TestDPConfigToPrivacySpec:
    """to_privacy_spec() converts to the contracts type."""

    def test_enabled_with_epsilon(self) -> None:
        cfg = DPConfig(noise_multiplier=1.1, max_grad_norm=0.8, delta=1e-5)
        spec = cfg.to_privacy_spec(epsilon_spent=3.5)
        assert spec.epsilon == 3.5
        assert spec.delta == 1e-5
        assert spec.noise_multiplier == 1.1
        assert spec.max_grad_norm == 0.8

    def test_disabled_returns_none_epsilon(self) -> None:
        cfg = DPConfig(enabled=False)
        spec = cfg.to_privacy_spec(epsilon_spent=None)
        assert spec.epsilon is None

    def test_no_epsilon_spent(self) -> None:
        cfg = DPConfig()
        spec = cfg.to_privacy_spec(epsilon_spent=None)
        assert spec.epsilon is None


class TestDPConfigValidation:
    """validate() catches bad parameters before Opacus sees them."""

    def test_valid_config_passes(self) -> None:
        cfg = DPConfig()
        cfg.validate()  # should not raise

    def test_negative_noise_multiplier(self) -> None:
        cfg = DPConfig(noise_multiplier=-1.0)
        with pytest.raises(ValueError, match="noise_multiplier"):
            cfg.validate()

    def test_zero_max_grad_norm(self) -> None:
        cfg = DPConfig(max_grad_norm=0.0)
        with pytest.raises(ValueError, match="max_grad_norm"):
            cfg.validate()

    def test_delta_out_of_range(self) -> None:
        cfg = DPConfig(delta=1.5)
        with pytest.raises(ValueError, match="delta"):
            cfg.validate()

    def test_negative_target_epsilon(self) -> None:
        cfg = DPConfig(target_epsilon=-1.0)
        with pytest.raises(ValueError, match="target_epsilon"):
            cfg.validate()


class TestDPConfigFromDict:
    """from_dict() reconstructs a DPConfig from a serialized dict."""

    def test_round_trip(self) -> None:
        original = DPConfig(noise_multiplier=0.7, target_epsilon=4.0)
        d = {
            "enabled": original.enabled,
            "noise_multiplier": original.noise_multiplier,
            "max_grad_norm": original.max_grad_norm,
            "delta": original.delta,
            "target_epsilon": original.target_epsilon,
            "secure_mode": original.secure_mode,
        }
        restored = DPConfig.from_dict(d)
        assert restored == original

    def test_partial_dict_uses_defaults(self) -> None:
        cfg = DPConfig.from_dict({"noise_multiplier": 2.0})
        assert cfg.noise_multiplier == 2.0
        assert cfg.target_epsilon == 8.0  # default
