from __future__ import annotations

import logging
from typing import Any

try:
    import torch
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    nn = None
    Optimizer = Any
    DataLoader = Any

try:
    import opacus
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    opacus = None
    ModuleValidator = None
    OPACUS_AVAILABLE = False

from .config import DPConfig

logger = logging.getLogger(__name__)


def _rebuild_optimizer(optimizer: Optimizer, new_model: nn.Module) -> Optimizer:
    """Recreate an optimizer for *new_model* preserving the original hyper-params.

    After ``ModuleValidator.fix()`` the returned model is a deep-copy, so the
    old optimizer's param references are stale.  We create a fresh optimizer of
    the same class with matching hyper-parameters.
    """
    # optimizer.defaults contains lr, momentum, weight_decay, etc.
    opt_cls = type(optimizer)
    defaults = {k: v for k, v in optimizer.defaults.items()}
    return opt_cls(new_model.parameters(), **defaults)


def make_private(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    config: DPConfig,
    *,
    epochs: int | None = None,
) -> tuple[nn.Module, Optimizer, DataLoader]:
    """Wrap the model, optimizer, and data_loader with Opacus for DP-SGD.

    If *config.enabled* is ``False``, returns the inputs unchanged.

    Incompatible modules (e.g. ``BatchNorm``) are replaced automatically by
    ``ModuleValidator.fix`` and the optimizer is recreated so its parameter
    references stay consistent.

    Noise calibration
    ~~~~~~~~~~~~~~~~~
    * If *epochs* is given (> 0), Opacus's ``make_private_with_epsilon`` is used
      to automatically calibrate the noise multiplier so that the training run
      stays within ``config.target_epsilon`` at ``config.delta``.
      **This is the recommended path for enforcing ε ≤ 8 (D7).**
    * Otherwise, ``make_private`` is called with the explicit
      ``config.noise_multiplier``; the caller is responsible for choosing a σ
      that respects the target ε.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer to use.
        data_loader: The data loader for training.
        config: The DP configuration.
        epochs: Number of training epochs.  When provided, Opacus will
            auto-calibrate σ to meet ``config.target_epsilon``.

    Returns:
        A tuple of (wrapped_model, wrapped_optimizer, wrapped_data_loader).
    """
    if not config.enabled:
        logger.info("DP-SGD is disabled. Returning model, optimizer, and data loader unchanged.")
        return model, optimizer, data_loader

    if not OPACUS_AVAILABLE or torch is None:
        raise ImportError(
            "Opacus and PyTorch are required for DP-SGD. "
            "Install them with `pip install opacus torch`."
        )

    # Validate the model and fix incompatible modules (e.g., BatchNorm -> GroupNorm)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.info("Found Opacus-incompatible modules, attempting to fix them...")
        model = ModuleValidator.fix(model)
        # fix() returns a deep copy — the old optimizer holds stale references.
        optimizer = _rebuild_optimizer(optimizer, model)
        logger.info("Optimizer recreated for the fixed model.")

    privacy_engine = opacus.PrivacyEngine(secure_mode=config.secure_mode)

    if epochs is not None and epochs > 0:
        # ── Auto-calibrate σ to enforce ε ≤ target_epsilon (D7) ──────────
        wrapped_model, wrapped_optimizer, wrapped_data_loader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=epochs,
                target_epsilon=config.target_epsilon,
                target_delta=config.delta,
                max_grad_norm=config.max_grad_norm,
            )
        )
        # The calibrated sigma lives on the DP optimizer. Opacus 1.6's default
        # PRV accountant has no noise_multiplier attribute, so reading it from
        # the accountant raised AttributeError on every epochs= call.
        actual_sigma = wrapped_optimizer.noise_multiplier
        logger.info(
            "DP-SGD initialized (auto-calibrated): target_epsilon=%s, "
            "target_delta=%s, epochs=%d, calibrated_sigma=%s, "
            "max_grad_norm=%s, secure_mode=%s",
            config.target_epsilon,
            config.delta,
            epochs,
            actual_sigma,
            config.max_grad_norm,
            config.secure_mode,
        )
    else:
        # ── Manual σ (caller is responsible for ε budget) ────────────────
        wrapped_model, wrapped_optimizer, wrapped_data_loader = (
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=config.noise_multiplier,
                max_grad_norm=config.max_grad_norm,
                poisson_sampling=True,
            )
        )
        logger.info(
            "DP-SGD initialized (manual σ): noise_multiplier=%s, "
            "max_grad_norm=%s, secure_mode=%s",
            config.noise_multiplier,
            config.max_grad_norm,
            config.secure_mode,
        )

    # Store privacy engine on the model for easy access later
    wrapped_model._privacy_engine = privacy_engine

    return wrapped_model, wrapped_optimizer, wrapped_data_loader


def get_privacy_engine(model: nn.Module) -> Any | None:
    """Retrieve the PrivacyEngine from a wrapped model.

    Args:
        model: The Opacus-wrapped model.

    Returns:
        The PrivacyEngine if attached, else None.
    """
    if hasattr(model, "_privacy_engine"):
        return model._privacy_engine
    return None
