from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AdapterValidationConfig:
    """Configuration for validating LoRA adapters."""
    max_adapter_size_bytes: int = 500 * 1024 * 1024  # 500MB
    max_tensor_norm: float = 100.0
    max_rank: int = 16  # D8 caps LoRA rank at 16
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
        
        rank = _lora_rank(key, arr)
        if rank is not None and rank > config.max_rank:
            errors.append(f"Tensor {key} has LoRA rank {rank}, above max_rank {config.max_rank}")

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


def _lora_rank(key: str, arr: np.ndarray) -> int | None:
    """The LoRA rank a factor carries: rows of lora_A, columns of lora_B."""
    if arr.ndim != 2:
        return None
    if "lora_A" in key:
        return int(arr.shape[0])
    if "lora_B" in key:
        return int(arr.shape[1])
    return None


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


#: MAD -> standard deviation for normally distributed data.
_MAD_TO_SIGMA = 1.4826


def detect_poisoning(adapters: list[dict[str, Any]], weights: list[float] | None = None,
                     threshold_sigma: float = 3.0, max_norm_ratio: float = 3.0) -> list[int]:
    """
    Detect potential poisoning across adapters by their update norms (Byzantine faults).

    Mean and standard deviation cannot work at CLASP's scale: with n samples the
    largest possible |z| is (n-1)/sqrt(n), i.e. 1.15 for a 3-client cluster and
    2.04 for all 6 clients, so a 3-sigma rule never fires and a single client at
    1000x the others' norm goes unflagged. This uses robust statistics instead:
    an adapter is flagged when its norm is more than ``max_norm_ratio`` times
    away from the median norm (above or below) **and** its robust z-score,
    |norm - median| / (1.4826 * MAD), exceeds ``threshold_sigma`` (or the MAD is
    zero). Needs at least 3 adapters; with fewer there is no majority to compare
    against and nothing is flagged.

    Args:
        adapters: List of adapter state dictionaries.
        weights: Optional aggregation weights (unused; kept for API stability).
        threshold_sigma: Robust z-score above which an adapter can be flagged.
        max_norm_ratio: How many times above or below the median norm an
            adapter must be before it can be flagged.

    Returns:
        List of indices of suspicious adapters.
    """
    if len(adapters) < 3:
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
        
    norms_array = np.array(adapter_norms, dtype=np.float64)
    median = float(np.median(norms_array))
    mad = float(np.median(np.abs(norms_array - median)))

    suspicious_indices = []
    for i, norm in enumerate(norms_array):
        if median > 0:
            ratio = norm / median
            far = ratio > max_norm_ratio or ratio < 1.0 / max_norm_ratio
        else:
            far = norm > 0
        if not far:
            continue
        robust_z = abs(norm - median) / (_MAD_TO_SIGMA * mad) if mad > 0 else float("inf")
        if robust_z > threshold_sigma:
            suspicious_indices.append(i)
    return suspicious_indices


def validate_adapter_metadata(metadata: Any) -> list[str]:
    """
    Validate required adapter metadata fields.

    Accepts the two upload shapes that exist in CLASP:

    * ``contracts.AdapterUpload`` (seam A), as an instance or its JSON dict:
      ``client_id``, ``round``, and ``hparams.rank`` / ``hparams.target_modules``;
    * the cluster's flat wire format: ``client_id``, ``round_id``, ``rank``,
      ``target_modules``.

    Args:
        metadata: An ``AdapterUpload`` or a metadata dictionary.

    Returns:
        List of error strings (empty if valid).
    """
    if dataclasses.is_dataclass(metadata) and not isinstance(metadata, type):
        metadata = dataclasses.asdict(metadata)
    if not isinstance(metadata, dict):
        return ["metadata must be a dict or a contracts.AdapterUpload"]
    if "hparams" in metadata:
        return _validate_contract_upload(metadata)
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


def _validate_contract_upload(metadata: dict[str, Any]) -> list[str]:
    """``contracts.AdapterUpload``: round and LoRA hparams live where the contract puts them."""
    hparams = metadata.get("hparams")
    flat = {
        "client_id": metadata.get("client_id"),
        "round_id": metadata.get("round"),
        "rank": hparams.get("rank") if isinstance(hparams, dict) else None,
        "target_modules": hparams.get("target_modules") if isinstance(hparams, dict) else None,
    }
    if isinstance(flat["target_modules"], tuple):
        flat["target_modules"] = list(flat["target_modules"])
    renames = {"round_id": "round", "rank": "hparams.rank",
               "target_modules": "hparams.target_modules"}
    present = {k: v for k, v in flat.items() if v is not None}
    errors = validate_adapter_metadata(present)
    for old, new in renames.items():
        errors = [e.replace(old, new) for e in errors]
    if not isinstance(hparams, dict):
        errors.append("hparams must be an object with rank and target_modules")
    return errors
