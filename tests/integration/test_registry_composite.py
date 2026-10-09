"""Registry composite (numpy) == Edge composite (torch), tensor for tensor.

D6 makes the registry build the pre-merged composite at promotion time, but
the edge already owns a reference construction in ``edge.merge.compose``. If
the two ever disagree, the adapter a developer is served is not the adapter
the edge evaluated. This pins them together on the same synthetic inputs:
identical keys, identical shapes, tensors equal to float32 precision, and the
same delta as the slow reference sum (``edge.merge.composite_delta``).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from contracts import LoRAHyperParams

from edge.merge import compose, composite_delta
from edge.wire import peft_key
from registry.composite import PartSpec, build_composite
from safetensors.numpy import load, save

LAYERS, RANK, DIM = 2, 4, 32
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _adapter(seed: int, lora_alpha: int) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in range(LAYERS):
        for m in MODULES:
            sd[peft_key(layer, m, "lora_A")] = rng.standard_normal((RANK, DIM)).astype(np.float32)
            sd[peft_key(layer, m, "lora_B")] = rng.standard_normal((DIM, RANK)).astype(np.float32)
    cfg = {"peft_type": "LORA", "r": RANK, "lora_alpha": lora_alpha,
           "target_modules": list(MODULES), "use_rslora": False}
    return sd, cfg


def _torch(sd: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v.copy()) for k, v in sd.items()}


@pytest.mark.parametrize("alpha,beta,alpha_c,alpha_l", [
    (0.5, 1.0, RANK, RANK),      # the panel config: pinned scaling, alpha=0.5
    (1.0, 1.0, 2 * RANK, RANK),  # cluster scaling 2.0 folded into B
    (0.0, 1.0, RANK, RANK),      # alpha=0 prunes the cluster block
    (0.25, 0.75, 8, 16),
])
def test_registry_composite_matches_edge_compose(alpha, beta, alpha_c, alpha_l):
    cluster_sd, cluster_cfg = _adapter(11, alpha_c)
    client_sd, client_cfg = _adapter(22, alpha_l)

    edge_sd, edge_cfg = compose([(_torch(cluster_sd), cluster_cfg, alpha),
                                 (_torch(client_sd), client_cfg, beta)])

    reg = build_composite(
        PartSpec(save(cluster_sd, metadata={"adapter_config": json.dumps(cluster_cfg)}),
                 LoRAHyperParams(rank=RANK, lora_alpha=alpha_c, target_modules=MODULES), alpha),
        PartSpec(save(client_sd, metadata={"adapter_config": json.dumps(client_cfg)}),
                 LoRAHyperParams(rank=RANK, lora_alpha=alpha_l, target_modules=MODULES), beta),
    )
    reg_sd = load(reg.payload)

    assert set(reg_sd) == set(edge_sd)
    assert reg.rank == edge_cfg["r"]
    for key, tensor in edge_sd.items():
        assert reg_sd[key].shape == tuple(tensor.shape), key
        np.testing.assert_allclose(reg_sd[key], tensor.numpy(), rtol=0, atol=1e-6)

    reference = composite_delta([(_torch(cluster_sd), cluster_cfg, alpha),
                                 (_torch(client_sd), client_cfg, beta)])
    for prefix, dw in reference.items():
        got = (reg_sd[f"{prefix}.lora_B.weight"].astype(np.float64)
               @ reg_sd[f"{prefix}.lora_A.weight"].astype(np.float64))
        np.testing.assert_allclose(got, dw.double().numpy(), rtol=1e-5, atol=1e-5)
