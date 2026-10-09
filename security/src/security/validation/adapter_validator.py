from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AdapterValidationConfig:
    """Configuration for validating LoRA adapters."""
    max_adapter_size_bytes: int = 500 * 1024 * 1024  # 500MB
    max_tensor_norm: float = 100.0
    max_rank: int = 128
    allowed_dtypes: tuple[str, ...] = ('float32', 'float16', 'bfloat16')
    enable_norm_check: bool = True
    enable_nan_check: bool = True
    enable_inf_check: bool = True


@dataclass
class AdapterValidationResult:
    """Result of an adapter validation check."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_adapter_integrity(state_dict: dict[str, Any], config: AdapterValidationConfig | None = None) -> AdapterValidationResult:
    """
    Validate an adapter for integrity and potential Byzantine faults.
    
    Args:
        state_dict: The adapter's state dictionary.
        config: Validation configuration.
        
    Returns:
        An AdapterValidationResult indicating if the adapter is valid.
    """
    if config is None:
        config = AdapterValidationConfig()
        
    errors = []
    warnings = []
    
    total_size = 0
    max_norm_found = 0.0
    num_tensors = 0
    
    for key, tensor in state_dict.items():
        if not isinstance(tensor, np.ndarray):
            # Try to convert if it's torch tensor, but we assume numpy arrays in this library
            try:
                # If torch tensor, we detach and move to cpu before numpy
                if hasattr(tensor, "detach"):
                    arr = tensor.detach().cpu().numpy()
                else:
                    arr = np.array(tensor)
            except (TypeError, ValueError, RuntimeError):
                errors.append(f"Tensor {key} could not be converted to numpy array")
                continue
        else:
            arr = tensor
            
        num_tensors += 1
        tensor_size = arr.nbytes
        total_size += tensor_size
        
        # dtype check
        if str(arr.dtype) not in config.allowed_dtypes:
            # Handle PyTorch dtype strings if present
            dtype_str = str(arr.dtype).replace('torch.', '')
            if dtype_str not in config.allowed_dtypes:
                errors.append(f"Tensor {key} has disallowed dtype {arr.dtype}")
        
        if config.enable_nan_check and np.isnan(arr).any():
            errors.append(f"Tensor {key} contains NaN values")
            
        if config.enable_inf_check and np.isinf(arr).any():
            errors.append(f"Tensor {key} contains Inf values")
            
        if config.enable_norm_check:
            norm = float(np.linalg.norm(arr))
            max_norm_found = max(max_norm_found, norm)
            
            if norm > config.max_tensor_norm:
                errors.append(f"Tensor {key} norm ({norm:.2f}) exceeds max allowed ({config.max_tensor_norm})")
            elif norm > config.max_tensor_norm * 0.8:
                warnings.append(f"Tensor {key} norm ({norm:.2f}) is close to max allowed ({config.max_tensor_norm})")
                
    if total_size > config.max_adapter_size_bytes:
        errors.append(f"Total adapter size ({total_size} bytes) exceeds limit ({config.max_adapter_size_bytes} bytes)")
        
    metadata = {
        "total_size_bytes": total_size,
        "num_tensors": num_tensors,
        "max_norm_found": max_norm_found
    }
    
    return AdapterValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )


def compute_adapter_hash(state_dict: dict[str, Any]) -> str:
    """
    Compute a SHA-256 hash of the adapter's tensors for integrity checking.
    
    Args:
        state_dict: The adapter state dictionary.
        
    Returns:
        A hex string of the SHA-256 hash.
    """
    hasher = hashlib.sha256()
    
    # Sort keys for deterministic hashing
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode('utf-8'))
        tensor = state_dict[key]
        
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)
            
        # Add the binary representation of the tensor
        hasher.update(arr.tobytes())
        
    return hasher.hexdigest()


def detect_poisoning(adapters: list[dict[str, Any]], weights: list[float] | None = None, threshold_sigma: float = 3.0) -> list[int]:
    """
    Detect potential poisoning across multiple adapters by comparing norms (Byzantine fault detection).
    
    Args:
        adapters: List of adapter state dictionaries.
        weights: Optional aggregation weights.
        threshold_sigma: Number of standard deviations from the mean to consider an adapter suspicious.
        
    Returns:
        List of indices of suspicious adapters.
    """
    if not adapters:
        return []
        
    adapter_norms = []
    
    for adapter in adapters:
        total_norm_sq = 0.0
        for tensor in adapter.values():
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.array(tensor)
            total_norm_sq += float(np.sum(np.square(arr)))
        adapter_norms.append(np.sqrt(total_norm_sq))
        
    norms_array = np.array(adapter_norms)
    mean_norm = np.mean(norms_array)
    std_norm = np.std(norms_array)
    
    suspicious_indices = []
    for i, norm in enumerate(norms_array):
        # Allow zero std to avoid division by zero
        if std_norm > 0 and abs(norm - mean_norm) > threshold_sigma * std_norm:
            suspicious_indices.append(i)
            
    return suspicious_indices


def validate_adapter_metadata(metadata: dict[str, Any]) -> list[str]:
    """
    Validate required adapter metadata fields.
    
    Args:
        metadata: The metadata dictionary to validate.
        
    Returns:
        List of error strings (empty if valid).
    """
    errors = []
    required_fields = ['client_id', 'round_id', 'rank', 'target_modules']
    
    for req_field in required_fields:
        if req_field not in metadata:
            errors.append(f"Missing required metadata field: {req_field}")
            
    # Check types if fields exist
    if 'client_id' in metadata and not isinstance(metadata['client_id'], str):
        errors.append("client_id must be a string")
        
    if 'round_id' in metadata and not isinstance(metadata['round_id'], int):
        errors.append("round_id must be an integer")
        
    if 'rank' in metadata and not isinstance(metadata['rank'], int):
        errors.append("rank must be an integer")
        
    if 'target_modules' in metadata:
        if not isinstance(metadata['target_modules'], list):
            errors.append("target_modules must be a list")
        elif not all(isinstance(m, str) for m in metadata['target_modules']):
            errors.append("target_modules must be a list of strings")
            
    return errors
