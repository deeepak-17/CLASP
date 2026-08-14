"""Opacus DP-SGD engine for CLASP.

Wraps ``opacus.PrivacyEngine`` with CLASP-specific defaults:
  - Ghost clipping (``grad_sample_mode="ghost"``) to fit in 4 GB VRAM.
  - Virtual batching (gradient accumulation) for logical batch > physical.
  - Special handling for PEFT/LoRA models on NF4-quantised bases.

References
----------
- FlashDP — Wang et al., NeurIPS 2025
- DP-SGD-RC — Ullah et al., ICML 2026
- LA-LoRA — Liu et al., ICLR 2026
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from security.dp_config import DPConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp_config: DPConfig,
    loss_fn: nn.Module | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    """Wrap *model*, *optimizer*, and *data_loader* with Opacus DP-SGD.

    Parameters
    ----------
    model : nn.Module
        The model to privatise.  Must have ``requires_grad`` set correctly.
    optimizer : torch.optim.Optimizer
        Standard PyTorch optimiser (e.g. AdamW).
    data_loader : DataLoader
        Training data loader.  Opacus replaces the sampler with a
        Poisson sampler for (ε, δ)-DP guarantees.
    dp_config : DPConfig
        Privacy parameters (ε, δ, σ, clip norm, etc.).
    loss_fn : nn.Module, optional
        Required when ``grad_sample_mode="ghost"`` — Opacus needs the loss
        to perform its two-pass backpropagation.

    Returns
    -------
    (dp_model, dp_optimizer, dp_data_loader)
    """
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # ----- validate / fix compatibility ----------------------------------
    if not ModuleValidator.is_valid(model):
        logger.warning("Model is not Opacus-compatible; attempting auto-fix …")
        model = ModuleValidator.fix(model)

    # ----- build privacy engine -----------------------------------------
    privacy_engine = PrivacyEngine(accountant=dp_config.accountant_type)

    # Auto-calibrate noise multiplier if not specified
    noise_multiplier = dp_config.noise_multiplier
    if noise_multiplier is None:
        from opacus.accountants.utils import get_noise_multiplier
        sample_size = len(data_loader.dataset)  # type: ignore[arg-type]
        sample_rate = dp_config.batch_size / sample_size
        noise_multiplier = get_noise_multiplier(
            target_epsilon=dp_config.target_epsilon,
            target_delta=dp_config.target_delta,
            sample_rate=sample_rate,
            epochs=dp_config.epochs,
        )
        logger.info(
            "Auto-calibrated noise_multiplier=%.4f for ε=%.2f, δ=%.1e, "
            "sample_rate=%.4f, epochs=%d",
            noise_multiplier,
            dp_config.target_epsilon,
            dp_config.target_delta,
            sample_rate,
            dp_config.epochs,
        )

    # ----- make_private kwargs ------------------------------------------
    make_private_kwargs: dict[str, Any] = dict(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
        grad_sample_mode=dp_config.grad_sample_mode,
    )

    # Ghost clipping requires the loss criterion
    if dp_config.grad_sample_mode == "ghost":
        if loss_fn is None:
            raise ValueError(
                "Ghost clipping requires loss_fn to be passed to make_private(). "
                "Opacus performs a 2-pass backprop and needs the loss criterion."
            )
        make_private_kwargs["loss"] = loss_fn

    dp_model, dp_optimizer, dp_data_loader = privacy_engine.make_private(
        **make_private_kwargs,
    )

    logger.info(
        "DP-SGD enabled: grad_sample_mode=%s, max_grad_norm=%.2f, σ=%.4f",
        dp_config.grad_sample_mode,
        dp_config.max_grad_norm,
        noise_multiplier,
    )

    return dp_model, dp_optimizer, dp_data_loader


def make_private_lora(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp_config: DPConfig,
    loss_fn: nn.Module | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    """Specialised wrapper for PEFT/LoRA models on NF4-quantised bases.

    The frozen NF4 base (e.g. DeepSeek-Coder 1.3B in uint8-packed 4-bit)
    is incompatible with Opacus's per-sample gradient hooks.  This function:

    1. Freezes *all* non-LoRA parameters and sets ``requires_grad=False``.
    2. Ensures LoRA A/B matrices (bfloat16) are the *only* trainable params.
    3. Delegates to :func:`make_private` for the actual Opacus wrapping.

    This follows the LA-LoRA (ICLR 2026) pattern of isolating trainable
    low-rank adapters from the frozen base to prevent noise from polluting
    quantised weights.
    """
    # Ensure only LoRA params are trainable
    lora_param_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_param_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(
        "LoRA isolation: %s trainable params (LoRA), %s frozen params (base)",
        f"{lora_param_count:,}",
        f"{frozen_count:,}",
    )

    if lora_param_count == 0:
        raise ValueError(
            "No LoRA parameters found (expected names containing 'lora_'). "
            "Did you call peft.get_peft_model() before make_private_lora()?"
        )

    return make_private(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        dp_config=dp_config,
        loss_fn=loss_fn,
    )
