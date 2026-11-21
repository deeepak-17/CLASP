"""Tests for the alpha/beta sweep's per-point adapter step (CPU only).

The merge math itself is gated by services/edge/tests/test_merge.py. These
tests only check what eval_point adds on top: which adapter a sweep point
hands to the evaluator.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.merge import (
    composite_delta,
    delta_weights,
    load_adapter,
    save_adapter,
)
from eval_point import prepare_adapter

R, IN_F, OUT_F = 16, 32, 24
PREFIXES = ["base_model.model.model.layers.0.self_attn.q_proj",
            "base_model.model.model.layers.0.self_attn.v_proj"]


def _cfg():
    return {"peft_type": "LORA", "r": R, "lora_alpha": R, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "deepseek-ai/deepseek-coder-1.3b-base",
            "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
            "lora_bias": False, "inference_mode": True, "task_type": "CAUSAL_LM",
            "rank_pattern": {}, "alpha_pattern": {}}


def _write(tmp: Path, name: str, seed: int) -> Path:
    gen = torch.Generator().manual_seed(seed)
    sd = {}
    for p in PREFIXES:
        sd[f"{p}.lora_A.weight"] = torch.randn(R, IN_F, generator=gen)
        sd[f"{p}.lora_B.weight"] = torch.randn(OUT_F, R, generator=gen)
    path = tmp / name
    save_adapter(path, sd, _cfg())
    return path


@pytest.fixture
def parts(tmp_path):
    return _write(tmp_path, "cluster", 1), _write(tmp_path, "client", 2)


def test_zero_zero_point_means_base_model_with_no_adapter(parts, tmp_path):
    cluster, client = parts
    assert prepare_adapter(0.0, 0.0, cluster, client, tmp_path / "out") is None


def test_alpha_zero_point_is_exactly_the_client_adapter(parts, tmp_path):
    cluster, client = parts
    out = prepare_adapter(0.0, 1.0, cluster, client, tmp_path / "out")
    got = delta_weights(*load_adapter(out))
    want = delta_weights(*load_adapter(client))
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-5)


def test_mixed_point_matches_alpha_cluster_plus_beta_client(parts, tmp_path):
    cluster, client = parts
    out = prepare_adapter(0.5, 1.0, cluster, client, tmp_path / "out")
    got = delta_weights(*load_adapter(out))
    want = composite_delta([(*load_adapter(cluster), 0.5), (*load_adapter(client), 1.0)])
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-4)
