"""DP-SGD configuration for CLASP's privacy-preserving LoRA training.

References
----------
- FlashDP (NeurIPS 2025) — cache-friendly per-layer DP-SGD
- DP-SGD-RC (ICML 2026) — randomized clipping via Hutch++
- LA-LoRA (ICLR 2026) — alternating LoRA updates under DP
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DPConfig:
    """Configuration for differentially-private training via Opacus.

    Defaults are tuned for CLASP's LoRA fine-tuning on a 4 GB VRAM GPU
    (RTX 2050).  Ghost clipping is **mandatory** — standard per-sample
    gradient storage would OOM instantly.

    Attributes
    ----------
    target_epsilon : float
        Privacy budget.  ε ≤ 8.0 is the standard for utility-preserving FL.
    target_delta : float
        Probability of accidental privacy breach.  Set to 1/N where N is
        the training-set size.
    max_grad_norm : float
        Per-sample gradient clip norm.  Applied *before* noise injection.
    noise_multiplier : float | None
        Gaussian noise scale σ.  If ``None``, auto-calibrated from
        ``target_epsilon`` via ``opacus.accountants.utils.get_noise_multiplier``.
    epochs : int
        Number of training epochs (used for noise calibration).
    batch_size : int
        *Logical* batch size for DP-SGD.  With 4 GB VRAM the physical
        micro-batch may be 1–4; Opacus virtual batching bridges the gap.
    physical_batch_size : int
        Actual per-step micro-batch that fits in VRAM.  Opacus accumulates
        ``batch_size // physical_batch_size`` micro-batches before adding
        noise.
    grad_sample_mode : str
        Opacus gradient-sample strategy.  ``"ghost"`` avoids per-sample
        gradient materialization (O(params) memory instead of O(B×params)).
    accountant_type : str
        Privacy accountant.  ``"rdp"`` uses Rényi DP for tight composition.
    """

    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = None
    epochs: int = 10
    batch_size: int = 32
    physical_batch_size: int = 4
    grad_sample_mode: str = "ghost"
    accountant_type: str = "rdp"

    def __post_init__(self) -> None:
        if self.grad_sample_mode not in ("ghost", "hooks", "ew"):
            raise ValueError(
                f"Unsupported grad_sample_mode: {self.grad_sample_mode!r}. "
                "Use 'ghost' for memory-efficient training."
            )
        if self.accountant_type not in ("rdp", "gdp", "prv"):
            raise ValueError(
                f"Unsupported accountant_type: {self.accountant_type!r}."
            )
        if self.physical_batch_size > self.batch_size:
            raise ValueError(
                "physical_batch_size must be ≤ batch_size."
            )
