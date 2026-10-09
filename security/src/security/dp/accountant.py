from __future__ import annotations

import logging

try:
    from opacus.accountants import create_accountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    create_accountant = None

from contracts.types import PrivacySpec

from .config import DPConfig

logger = logging.getLogger(__name__)

#: The one accountant the library uses (D7): Opacus's default, PRV. It is what
#: ``make_private`` calibrates sigma with and what its PrivacyEngine reports, so
#: tracking a run with :class:`PrivacyAccountant` gives the same epsilon the
#: noise was calibrated for. RDP is looser: on a run calibrated to epsilon 8
#: with PRV it reads about 9.4, which would wrongly exhaust the budget.
DEFAULT_ACCOUNTANT = "prv"


class BudgetExhaustedError(Exception):
    """Raised when the privacy budget (epsilon) is exhausted."""


class PrivacyAccountant:
    """
    Privacy accountant for tracking epsilon spend.

    Uses the same Opacus accountant as ``make_private`` (``DEFAULT_ACCOUNTANT``,
    PRV) unless ``accountant`` says otherwise, e.g. ``"rdp"`` for the looser
    RDP bound.
    """

    def __init__(self, config: DPConfig, sample_rate: float = 1.0, *,
                 accountant: str = DEFAULT_ACCOUNTANT):
        """
        Initializes the privacy accountant.

        Args:
            config: The DPConfig for the current context.
            sample_rate: The probability of a single sample being included in a batch.
            accountant: Opacus accountant mechanism ("prv", "rdp" or "gdp").
        """
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for PrivacyAccountant.")

        self.config = config
        self.sample_rate = sample_rate
        self.accountant_type = accountant
        self._accountant = create_accountant(accountant)

    def step(self, num_steps: int = 1) -> float:
        """
        Advances the accountant by the given number of steps.
        
        Args:
            num_steps: The number of optimization steps taken.
            
        Returns:
            The current epsilon value.
            
        Raises:
            BudgetExhaustedError: If the target epsilon is exceeded.
        """
        for _ in range(num_steps):
            self._accountant.step(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=self.sample_rate
            )
            
        eps = self.get_epsilon()
        if eps > self.config.target_epsilon:
            raise BudgetExhaustedError(
                f"Privacy budget exhausted: spent {eps:.4f} > target {self.config.target_epsilon:.4f}"
            )
        return eps

    def compute_epsilon(self, steps: int, sample_rate: float, delta: float | None = None) -> float:
        """
        Computes what epsilon would be for given parameters (static-style calculation).
        
        Args:
            steps: The number of optimization steps.
            sample_rate: The sampling probability for each step.
            delta: The target delta (defaults to config delta).
            
        Returns:
            The computed epsilon.
        """
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required.")
            
        target_delta = delta if delta is not None else self.config.delta
        
        # We can just create a temporary accountant for this calculation
        temp_accountant = create_accountant(self.accountant_type)
        for _ in range(steps):
            temp_accountant.step(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=sample_rate
            )
        return temp_accountant.get_epsilon(delta=target_delta)

    def get_epsilon(self) -> float:
        """Returns the current spent epsilon."""
        try:
            return self._accountant.get_epsilon(delta=self.config.delta)
        except (ValueError, IndexError):
            return 0.0

    def get_privacy_spec(self) -> PrivacySpec:
        """Returns the PrivacySpec representing the current state."""
        return self.config.to_privacy_spec(epsilon_spent=self.get_epsilon())

    @property
    def budget_remaining(self) -> float:
        """Returns the remaining privacy budget (target - spent)."""
        return max(0.0, self.config.target_epsilon - self.get_epsilon())

    @property
    def is_exhausted(self) -> bool:
        """Returns True if the privacy budget has been exceeded."""
        return self.get_epsilon() > self.config.target_epsilon
