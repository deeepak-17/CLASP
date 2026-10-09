from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from contracts.types import PrivacySpec


@dataclass(frozen=True)
class DPConfig:
    """
    Configuration for Differential Privacy (DP-SGD).
    """
    enabled: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    target_epsilon: float = 8.0
    secure_mode: bool = False

    def validate(self) -> None:
        """
        Validates the configuration parameters.
        Raises ValueError if any parameter is invalid.
        """
        if self.noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be > 0, got {self.noise_multiplier}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if not (0 < self.delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be > 0, got {self.target_epsilon}")

    # Fields accepted by the old API but no longer used by this DPConfig.
    _IGNORED_LEGACY_KEYS = frozenset({
        "target_delta", "epochs", "batch_size", "physical_batch_size",
        "grad_sample_mode", "accountant_type",
    })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DPConfig:
        """Create a DPConfig from a dictionary, ignoring unknown/legacy keys."""
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_privacy_spec(self, epsilon_spent: float | None = None) -> PrivacySpec:
        """
        Converts the config to a PrivacySpec contract.
        
        Args:
            epsilon_spent: The amount of epsilon spent so far.
        
        Returns:
            A PrivacySpec instance representing the current privacy guarantees.
        """
        return PrivacySpec(
            epsilon=epsilon_spent,
            delta=self.delta,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
