"""Composite-merge correctness gate (P1 Edge, W3 · E3.4).

E3.4 is the gate the rest of the project rests on: if the merge math is wrong,
every personalization number from W8 onward is poisoned and no downstream
measurement can tell. So the identities are asserted mathematically, not
eyeballed against a loss curve.

The three that matter:

    α=0, β=0  ->  ΔW == 0            (composite recovers the base exactly)
    α=0, β=1  ->  ΔW == ΔW_client    (composite == client adapter alone)
    α=1, β=0  ->  ΔW == ΔW_cluster

plus linearity in (α, β), which catches sign errors and scaling that got folded
in twice — the two mistakes that survive the endpoint checks.

`compose` is checked against `composite_delta`, an independent slow
implementation that sums materialized ΔW matrices directly. Testing the
concatenation trick against itself would prove nothing.

Runs on CPU with small synthetic adapters; the real 192-tensor adapter from the
smoke run is used too when it is present.
"""
import json
import math
import os
from pathlib import Path

import pytest
import torch

from edge.merge import (
    CONTRACT_HYPERPARAMS,
    compose,
    composite_delta,
    delta_weights,
    load_adapter,
    make_stub_adapter,
    merge_max_error,
    module_prefixes,
    save_adapter,
    scaling_of,
    shapes_from,
    validate_compatibility,
)

R = 16
IN_F, OUT_F = 32, 24
PREFIXES = ["base_model.model.model.layers.0.self_attn.q_proj",
            "base_model.model.model.layers.0.self_attn.v_proj"]


def make_cfg(**over):
    cfg = {
        "peft_type": "LORA", "r": R, "lora_alpha": R, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "base_model_name_or_path": "deepseek-ai/deepseek-coder-1.3b-base",
        "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
        "lora_bias": False, "inference_mode": True, "task_type": "CAUSAL_LM",
        "rank_pattern": {}, "alpha_pattern": {},
    }
    cfg.update(over)
    return cfg


def make_adapter(seed: int, cfg=None):
    """A dense, nonzero adapter — B is deliberately not zero, unlike a freshly
    initialized one whose product would vanish and hide sign errors."""
    cfg = cfg or make_cfg()
    gen = torch.Generator().manual_seed(seed)
    sd = {}
    for p in PREFIXES:
        sd[f"{p}.lora_A.weight"] = torch.randn(cfg["r"], IN_F, generator=gen)
        sd[f"{p}.lora_B.weight"] = torch.randn(OUT_F, cfg["r"], generator=gen)
    return sd, cfg


@pytest.fixture
def cluster():
    return make_adapter(seed=1)


@pytest.fixture
def client():
    return make_adapter(seed=2)


def max_abs_diff(a: dict, b: dict) -> float:
    assert set(a) == set(b), "module sets differ"
    return max((a[p] - b[p]).abs().max().item() for p in a)


def max_rel_diff(a: dict, b: dict) -> float:
    """Error relative to the magnitude of the reference.

    The right yardstick for anything that is not a bitwise identity. ΔW entries
    reach |80| under rsLoRA scaling, where fp32 rounding alone is ~1.5e-5
    absolute — an absolute threshold would call correct arithmetic a bug. Only
    the exact identities (α=0 cases) are asserted with ==.
    """
    assert set(a) == set(b), "module sets differ"
    return max((a[p] - b[p]).abs().max().item() / max(b[p].abs().max().item(), 1e-12)
               for p in a)


# ~10x fp32 epsilon (1.19e-7): tight enough to catch a real merge error, loose
# enough to survive summing rank-32 products.
FP32_TOL = 1e-6


# --- E3.4: the three identities -------------------------------------------

def test_alpha0_beta0_recovers_base_exactly(cluster, client):
    """The composite must add literally nothing. Not 'nearly nothing' — zero."""
    sd, cfg = compose([(*cluster, 0.0), (*client, 0.0)])
    for prefix, dw in delta_weights(sd, cfg).items():
        assert torch.count_nonzero(dw) == 0, f"{prefix} perturbs the base"


def test_alpha0_beta1_equals_client_alone_bitwise(cluster, client):
    """With the pinned config (lora_alpha == r) every scaling is exactly 1.0,
    so this identity holds bitwise, not just to tolerance."""
    sd, cfg = compose([(*cluster, 0.0), (*client, 1.0)])
    got = delta_weights(sd, cfg)
    want = delta_weights(*client)
    assert max_abs_diff(got, want) == 0.0


def test_alpha1_beta0_equals_cluster_alone_bitwise(cluster, client):
    sd, cfg = compose([(*cluster, 1.0), (*client, 0.0)])
    assert max_abs_diff(delta_weights(sd, cfg), delta_weights(*cluster)) == 0.0


def test_alpha0_beta1_prunes_back_to_rank_r(cluster, client):
    """A zero coefficient should not leave a dead rank-16 block behind: the
    served composite is then literally the client adapter."""
    sd, cfg = compose([(*cluster, 0.0), (*client, 1.0)])
    assert cfg["r"] == R
    assert scaling_of(cfg) == 1.0


# --- E3.4: the general case, against an independent implementation ---------

@pytest.mark.parametrize("alpha,beta", [
    (1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.25, 0.75), (2.0, 0.1), (-0.5, 1.0),
])
def test_composite_matches_reference_sum(cluster, client, alpha, beta):
    parts = [(*cluster, alpha), (*client, beta)]
    sd, cfg = compose(parts)
    assert max_rel_diff(delta_weights(sd, cfg), composite_delta(parts)) < FP32_TOL


def test_composite_is_linear_in_alpha_and_beta(cluster, client):
    """ΔW(α,β) == α·ΔW(1,0) + β·ΔW(0,1). Catches double-applied scaling."""
    alpha, beta = 0.3, 0.7
    got = delta_weights(*compose([(*cluster, alpha), (*client, beta)]))
    only_c = delta_weights(*compose([(*cluster, 1.0), (*client, 0.0)]))
    only_l = delta_weights(*compose([(*cluster, 0.0), (*client, 1.0)]))
    want = {p: alpha * only_c[p] + beta * only_l[p] for p in got}
    assert max_rel_diff(got, want) < FP32_TOL


def test_sign_of_alpha_is_respected(cluster, client):
    """A dropped minus sign would still pass a magnitude-only check."""
    pos = delta_weights(*compose([(*cluster, 1.0), (*client, 0.0)]))
    neg = delta_weights(*compose([(*cluster, -1.0), (*client, 0.0)]))
    assert max_abs_diff({p: -v for p, v in pos.items()}, neg) == 0.0


def test_rank_concatenation_doubles_rank_when_both_live(cluster, client):
    sd, cfg = compose([(*cluster, 0.5), (*client, 0.5)])
    assert cfg["r"] == 2 * R
    assert cfg["lora_alpha"] == 2 * R      # => scaling 1.0
    for p in module_prefixes(sd):
        assert sd[f"{p}.lora_A.weight"].shape == (2 * R, IN_F)
        assert sd[f"{p}.lora_B.weight"].shape == (OUT_F, 2 * R)


def test_all_zero_composite_is_still_a_loadable_adapter(cluster, client):
    """α=β=0 must not produce an empty state dict that fails to load."""
    sd, cfg = compose([(*cluster, 0.0), (*client, 0.0)])
    assert module_prefixes(sd) == PREFIXES
    assert cfg["r"] == R


def test_source_adapters_are_not_mutated(cluster, client):
    before = {k: v.clone() for k, v in client[0].items()}
    compose([(*cluster, 0.5), (*client, 2.0)])
    for k, v in before.items():
        assert torch.equal(client[0][k], v), f"{k} was modified in place"


def test_module_union_keeps_single_sided_modules(client):
    """A module the cluster never touched still belongs in the composite."""
    partial_sd = {k: v for k, v in client[0].items() if PREFIXES[0] in k}
    sd, cfg = compose([(partial_sd, client[1], 1.0), (*client, 1.0)])
    assert module_prefixes(sd) == PREFIXES
    # the shared module gets both blocks, the client-only module just one
    assert sd[f"{PREFIXES[0]}.lora_A.weight"].shape[0] == 2 * R
    assert sd[f"{PREFIXES[1]}.lora_A.weight"].shape[0] == R


# --- scaling variants ------------------------------------------------------

def test_non_unit_scaling_is_folded_in_correctly():
    """lora_alpha != r means scaling != 1. The composite must absorb it."""
    cfg = make_cfg(lora_alpha=32)          # scaling 2.0
    a = make_adapter(seed=3, cfg=cfg)
    b = make_adapter(seed=4, cfg=make_cfg())
    assert scaling_of(cfg) == 2.0
    parts = [(*a, 1.0), (*b, 1.0)]
    assert max_rel_diff(delta_weights(*compose(parts)), composite_delta(parts)) < FP32_TOL


def test_rslora_scaling_is_honoured():
    cfg = make_cfg(use_rslora=True)
    assert scaling_of(cfg) == pytest.approx(R / math.sqrt(R))
    a = make_adapter(seed=5, cfg=cfg)
    b = make_adapter(seed=6, cfg=cfg)
    parts = [(*a, 1.0), (*b, 1.0)]
    sd, out_cfg = compose(parts)
    assert out_cfg["use_rslora"] is False   # folded in, must not re-apply
    assert max_rel_diff(delta_weights(sd, out_cfg), composite_delta(parts)) < FP32_TOL


# --- stand-in cluster adapter ---------------------------------------------

def test_zero_stub_contributes_nothing(client):
    """The W3 reality: no real cluster adapter exists, so α's term must be
    provably inert and any output change attributable to the client alone."""
    stub = make_stub_adapter(client[1], module_prefixes(client[0]),
                             shapes_from(client[0]), mode="zeros")
    got = delta_weights(*compose([(*stub, 1.0), (*client, 1.0)]))
    want = delta_weights(*client)
    assert max_abs_diff(got, want) == 0.0


def test_random_stub_actually_perturbs(client):
    """...whereas the random stand-in must NOT be inert, or it would exercise
    nothing in the α path."""
    stub = make_stub_adapter(client[1], module_prefixes(client[0]),
                             shapes_from(client[0]), mode="random", seed=7)
    got = delta_weights(*compose([(*stub, 1.0), (*client, 1.0)]))
    assert max_abs_diff(got, delta_weights(*client)) > 1e-3


def test_random_stub_is_seeded(client):
    args = (client[1], module_prefixes(client[0]), shapes_from(client[0]))
    a = make_stub_adapter(*args, mode="random", seed=11)[0]
    b = make_stub_adapter(*args, mode="random", seed=11)[0]
    c = make_stub_adapter(*args, mode="random", seed=12)[0]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_stub_mode_is_validated(client):
    with pytest.raises(ValueError, match="zeros"):
        make_stub_adapter(client[1], module_prefixes(client[0]),
                          shapes_from(client[0]), mode="uniform")


# --- compatibility validation (P1 scope: catch silent mismatches) ----------

def test_matching_adapters_validate():
    info = validate_compatibility([make_cfg(), make_cfg()], ["cluster", "client"])
    assert info["ranks"] == {"cluster": R, "client": R}


def test_base_model_mismatch_is_rejected():
    other = make_cfg(base_model_name_or_path="deepseek-ai/deepseek-coder-6.7b-base")
    with pytest.raises(ValueError, match="base model mismatch"):
        validate_compatibility([make_cfg(), other], ["cluster", "client"],
                               contract=None)


def test_target_module_mismatch_is_rejected():
    other = make_cfg(target_modules=["q_proj", "v_proj"])
    with pytest.raises(ValueError, match="target_modules mismatch"):
        validate_compatibility([make_cfg(), other], ["cluster", "client"],
                               contract=None)


def test_rslora_mismatch_is_rejected():
    with pytest.raises(ValueError, match="use_rslora mismatch"):
        validate_compatibility([make_cfg(), make_cfg(use_rslora=True)],
                               ["cluster", "client"], contract=None)


def test_contract_violation_is_rejected():
    """Rank drifting off 16 means someone trained outside D8's envelope."""
    with pytest.raises(ValueError, match="violates contract"):
        validate_compatibility([make_cfg(r=8, lora_alpha=8)], ["client"],
                               contract=CONTRACT_HYPERPARAMS)


def test_differing_ranks_between_adapters_are_allowed():
    """Concatenation handles them; only the contract pins the value."""
    info = validate_compatibility([make_cfg(r=8, lora_alpha=8), make_cfg()],
                                  ["cluster", "client"], contract=None)
    assert info["ranks"] == {"cluster": 8, "client": 16}


def test_differing_ranks_compose_correctly():
    a = make_adapter(seed=8, cfg=make_cfg(r=8, lora_alpha=8))
    b = make_adapter(seed=9, cfg=make_cfg())
    parts = [(*a, 0.5), (*b, 1.0)]
    sd, cfg = compose(parts)
    assert cfg["r"] == 8 + 16
    assert max_rel_diff(delta_weights(sd, cfg), composite_delta(parts)) < FP32_TOL


# --- round trip ------------------------------------------------------------

def test_composite_survives_save_and_load(tmp_path, cluster, client):
    parts = [(*cluster, 0.4), (*client, 0.9)]
    sd, cfg = compose(parts)
    save_adapter(tmp_path / "composite", sd, cfg)
    sd2, cfg2 = load_adapter(tmp_path / "composite")
    assert cfg2["r"] == cfg["r"] and cfg2["lora_alpha"] == cfg["lora_alpha"]
    assert max_rel_diff(delta_weights(sd2, cfg2), composite_delta(parts)) < FP32_TOL


def test_saved_config_is_valid_peft_json(tmp_path, cluster, client):
    sd, cfg = compose([(*cluster, 1.0), (*client, 1.0)])
    save_adapter(tmp_path / "c", sd, cfg)
    loaded = json.loads((tmp_path / "c" / "adapter_config.json").read_text())
    assert loaded["peft_type"] == "LORA"
    assert loaded["task_type"] == "CAUSAL_LM"


# --- against the real trained adapter, when it is around -------------------

# Repo-relative so the test is portable, with an env override for CI or a
# different training output. Weights are gitignored (*.safetensors), so on a
# fresh clone this skips rather than fails — regenerate with:
#   python -m edge.train_client --client web/client-requests --budget-plan budget_plan.json
REAL_ADAPTER = Path(
    os.environ.get(
        "CLASP_TEST_ADAPTER",
        Path(__file__).resolve().parents[1] / "artifacts" / "adapters" / "client-requests",
    )
)


@pytest.mark.skipif(not REAL_ADAPTER.exists(), reason="no trained adapter on disk")
def test_identities_hold_on_the_real_adapter():
    """Same three gates, on the 192-tensor adapter from the E3.1 run.

    Streams one module at a time via `merge_max_error`. Materializing whole ΔW
    sets here is not merely slow: 96 modules x 2048x2048 fp32 is ~1.5 GB per
    set, and holding several at once killed the interpreter with an access
    violation before this was rewritten.
    """
    client_sd, client_cfg = load_adapter(REAL_ADAPTER)
    stub = make_stub_adapter(client_cfg, module_prefixes(client_sd),
                             shapes_from(client_sd), mode="random", seed=3)

    # alpha=0, beta=0 -> exactly the base
    sd, cfg = compose([(*stub, 0.0), (client_sd, client_cfg, 0.0)])
    for prefix in module_prefixes(sd):
        dw = delta_weights(sd, cfg, [prefix])[prefix]
        assert torch.count_nonzero(dw) == 0
        del dw

    # alpha=0, beta=1 -> exactly the client adapter, module by module
    sd, cfg = compose([(*stub, 0.0), (client_sd, client_cfg, 1.0)])
    for prefix in module_prefixes(sd):
        got = delta_weights(sd, cfg, [prefix])[prefix]
        want = delta_weights(client_sd, client_cfg, [prefix])[prefix]
        assert torch.equal(got, want), f"{prefix} differs from the client adapter"
        del got, want

    # general case against the independent reference sum
    parts = [(*stub, 0.5), (client_sd, client_cfg, 1.0)]
    err = merge_max_error(parts, *compose(parts))
    assert err["max_rel_err"] < FP32_TOL, err


@pytest.mark.skipif(not REAL_ADAPTER.exists(), reason="no trained adapter on disk")
def test_real_adapter_satisfies_the_contract():
    _, cfg = load_adapter(REAL_ADAPTER)
    validate_compatibility([cfg], ["client"], contract=CONTRACT_HYPERPARAMS)
