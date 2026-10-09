"""Tests for security.validation — adapter validation and poisoning detection."""
from __future__ import annotations

import numpy as np

from security.validation.adapter_validator import (
    AdapterValidationConfig,
    compute_adapter_hash,
    detect_poisoning,
    validate_adapter_integrity,
    validate_adapter_metadata,
)


class TestValidateAdapterIntegrity:
    def test_valid_adapter(self) -> None:
        state = {"w": np.ones((4, 4), dtype=np.float32)}
        result = validate_adapter_integrity(state)
        assert result.valid is True
        assert result.errors == []

    def test_nan_detected(self) -> None:
        state = {"w": np.array([float("nan")], dtype=np.float32)}
        result = validate_adapter_integrity(state)
        assert result.valid is False
        assert any("NaN" in e for e in result.errors)

    def test_inf_detected(self) -> None:
        state = {"w": np.array([float("inf")], dtype=np.float32)}
        result = validate_adapter_integrity(state)
        assert result.valid is False
        assert any("Inf" in e for e in result.errors)

    def test_norm_exceeded(self) -> None:
        big = np.ones((1000,), dtype=np.float32) * 1000
        cfg = AdapterValidationConfig(max_tensor_norm=1.0)
        result = validate_adapter_integrity({"w": big}, config=cfg)
        assert result.valid is False
        assert any("norm" in e for e in result.errors)

    def test_disallowed_dtype(self) -> None:
        state = {"w": np.ones((2,), dtype=np.float64)}
        result = validate_adapter_integrity(state)
        assert result.valid is False
        assert any("dtype" in e for e in result.errors)

    def test_size_exceeded(self) -> None:
        big = np.zeros((1024, 1024), dtype=np.float32)
        cfg = AdapterValidationConfig(max_adapter_size_bytes=100)
        result = validate_adapter_integrity({"w": big}, config=cfg)
        assert result.valid is False

    def test_metadata_populated(self) -> None:
        state = {"a": np.zeros((2, 3), dtype=np.float32), "b": np.ones((3,), dtype=np.float32)}
        result = validate_adapter_integrity(state)
        assert result.metadata["num_tensors"] == 2
        assert result.metadata["total_size_bytes"] > 0


class TestComputeAdapterHash:
    def test_deterministic(self) -> None:
        state = {"w": np.ones((4,), dtype=np.float32)}
        h1 = compute_adapter_hash(state)
        h2 = compute_adapter_hash(state)
        assert h1 == h2

    def test_differs_for_different_data(self) -> None:
        h1 = compute_adapter_hash({"w": np.zeros((4,), dtype=np.float32)})
        h2 = compute_adapter_hash({"w": np.ones((4,), dtype=np.float32)})
        assert h1 != h2


class TestDetectPoisoning:
    def test_no_suspicious_in_uniform(self) -> None:
        adapters = [{"w": np.ones((10,), dtype=np.float32)} for _ in range(5)]
        assert detect_poisoning(adapters) == []

    def test_detects_outlier(self) -> None:
        normal = [{"w": np.ones((10,), dtype=np.float32)} for _ in range(10)]
        poisoned = {"w": np.ones((10,), dtype=np.float32) * 10000}
        normal.append(poisoned)
        suspicious = detect_poisoning(normal)
        assert 10 in suspicious  # the outlier index

    def test_empty_returns_empty(self) -> None:
        assert detect_poisoning([]) == []


class TestValidateAdapterMetadata:
    def test_valid_metadata(self) -> None:
        meta = {
            "client_id": "edge-0",
            "round_id": 1,
            "rank": 8,
            "target_modules": ["q_proj", "v_proj"],
        }
        assert validate_adapter_metadata(meta) == []

    def test_missing_fields(self) -> None:
        errors = validate_adapter_metadata({})
        assert len(errors) == 4  # 4 required fields

    def test_wrong_types(self) -> None:
        meta = {
            "client_id": 123,
            "round_id": "one",
            "rank": "high",
            "target_modules": "q_proj",
        }
        errors = validate_adapter_metadata(meta)
        assert len(errors) >= 4
