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

from .accountant import DEFAULT_ACCOUNTANT
from .config import DPConfig

logger = logging.getLogger(__name__)


def _rebuild_optimizer(optimizer: Optimizer, old_model: nn.Module,
                       new_model: nn.Module) -> Optimizer:
    """Recreate *optimizer* for *new_model*, keeping its param groups.

    After ``ModuleValidator.fix()`` the returned model is a deep copy, so the
    old optimizer's param references are stale. Each group is rebuilt with the
    same hyper-parameters (per-group lr, weight decay, ...) over the matching
    parameters of the copy, matched by name. Parameters the original optimizer
    did not hold (e.g. frozen layers) stay out of it. Optimizer state such as
    momentum is not carried over; this runs before training starts.

    Raises:
        ValueError: if a parameter cannot be matched by name in the fixed model
            (``fix()`` replaced the module that held it).
    """
    old_names = {id(p): name for name, p in old_model.named_parameters()}
    new_params = dict(new_model.named_parameters())
    groups = []
    for group in optimizer.param_groups:
        params = []
        for p in group["params"]:
            name = old_names.get(id(p))
            if name is None or name not in new_params:
                raise ValueError(
                    f"cannot rebuild the optimizer after ModuleValidator.fix(): parameter "
                    f"{name or '<not in model>'!r} has no counterpart in the fixed model; "
                    "fix the model before creating the optimizer"
                )
            params.append(new_params[name])
        groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": params})
    return type(optimizer)(groups, **optimizer.defaults)


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
        fixed = ModuleValidator.fix(model)
        # fix() returns a deep copy — the old optimizer holds stale references.
        optimizer = _rebuild_optimizer(optimizer, model, fixed)
        model = fixed
        logger.info("Optimizer recreated for the fixed model.")

    # Pinned (not left to Opacus's default) so PrivacyAccountant and this engine
    # always calibrate and report with the same accountant.
    privacy_engine = opacus.PrivacyEngine(accountant=DEFAULT_ACCOUNTANT,
                                          secure_mode=config.secure_mode)

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
