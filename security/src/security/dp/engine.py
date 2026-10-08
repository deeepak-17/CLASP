from __future__ import annotations

import logging
from typing import Any

try:
    import torch
    import torch.nn as nn
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


def make_private(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    config: DPConfig,
) -> tuple[nn.Module, Optimizer, DataLoader]:
    """
    Wraps the model, optimizer, and data_loader with Opacus for DP-SGD.
    
    If `config.enabled` is False, returns the inputs unchanged.
    Incompatible modules with requires_grad=True will be replaced by Opacus ModuleValidator.
    
    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer to use.
        data_loader: The data loader for training.
        config: The DP configuration.
        
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
        # Note: Opacus fixes the whole model by default. For LoRA with frozen base,
        # ModuleValidator.fix still works and ignores frozen params if they don't get gradients.
        model = ModuleValidator.fix(model)

    privacy_engine = opacus.PrivacyEngine(secure_mode=config.secure_mode)
    
    wrapped_model, wrapped_optimizer, wrapped_data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.max_grad_norm,
        poisson_sampling=True,
    )
    
    # Store privacy engine on the model for easy access later
    wrapped_model._privacy_engine = privacy_engine
    
    logger.info(
        f"DP-SGD initialized: noise_multiplier={config.noise_multiplier}, "
        f"max_grad_norm={config.max_grad_norm}, secure_mode={config.secure_mode}"
    )
    return wrapped_model, wrapped_optimizer, wrapped_data_loader


def get_privacy_engine(model: nn.Module) -> Any | None:
    """
    Retrieves the PrivacyEngine from a wrapped model.
    
    Args:
        model: The Opacus-wrapped model.
        
    Returns:
        The PrivacyEngine if attached, else None.
    """
    if hasattr(model, "_privacy_engine"):
        return model._privacy_engine
    return None
