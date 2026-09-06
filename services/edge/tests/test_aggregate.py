"""Cluster aggregation tests (W4 · G2, D2).

D2 requires the SVD aggregation be "unit-tested against the exact product
average, with naive averaging kept only as an ablation baseline". These do
exactly that, and the naive-baseline tests are the ones that justify the
algorithm choice: they demonstrate the cross-term bias rather than asserting it.
"""
import pytest
import torch

from edge.aggregate import (
    aggregate_naive,
    aggregate_svd,
    exact_average_error,
    normalize_weights,
    refactorize,
)
from edge.merge import delta_weights, module_prefixes

R = 16
IN_F, OUT_F = 64, 48
PREFIXES = ["m.layers.0.self_attn.q_proj", "m.layers.0.self_attn.v_proj"]


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


def make_adapter(seed, cfg=None, scale=1.0):
    cfg = cfg or make_cfg()
    gen = torch.Generator().manual_seed(seed)
    sd = {}
    for p in PREFIXES:
        sd[f"{p}.lora_A.weight"] = torch.randn(cfg["r"], IN_F, generator=gen) * scale
        sd[f"{p}.lora_B.weight"] = torch.randn(OUT_F, cfg["r"], generator=gen) * scale
    return sd, cfg


# --- weighting -------------------------------------------------------------

def test_weights_normalize_to_one():
    w = normalize_weights([199, 239, 222])
    assert sum(w) == pytest.approx(1.0)
    assert w[1] > w[2] > w[0]          # ordering preserved


def test_zero_total_weight_is_rejected():  # noqa: F821
    with pytest.raises(ValueError, match="sum to"):
        normalize_weights([0, 0])


# --- refactorization -------------------------------------------------------

def test_refactorize_is_exact_for_a_low_rank_matrix():
    """A matrix that already has rank <= r must survive truncation losslessly."""
    gen = torch.Generator().manual_seed(0)
    b = torch.randn(OUT_F, 4, generator=gen)
    a = torch.randn(4, IN_F, generator=gen)
    dw = b @ a                                    # rank 4
    a2, b2, rel = refactorize(dw, rank=R)
    assert rel < 1e-4
    assert torch.linalg.norm(b2 @ a2 - dw) / torch.linalg.norm(dw) < 1e-4


def test_refactorize_loses_something_on_a_full_rank_matrix():
    """...and must NOT claim to be lossless when it cannot be."""
    dw = torch.randn(OUT_F, IN_F, generator=torch.Generator().manual_seed(1))
    _, _, rel = refactorize(dw, rank=4)
    assert rel > 0.1


def test_refactorize_shapes_and_padding():
    dw = torch.randn(OUT_F, IN_F)
    a, b, _ = refactorize(dw, rank=R)
    assert a.shape == (R, IN_F)
    assert b.shape == (OUT_F, R)


def test_refactorize_pads_when_rank_exceeds_matrix_dimension():
    dw = torch.randn(6, 8)
    a, b, _ = refactorize(dw, rank=R)
    assert a.shape == (R, 8) and b.shape == (6, R)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_refactorize_pad_stays_on_the_input_device():
    """The pad branch must not drag a CUDA factorization back to the CPU.

    An unqualified torch.zeros allocates on the CPU, and torch.cat across
    devices raises. This cannot be reproduced on a CPU-only runner, so CI
    skips it; it is the GPU box that keeps the pad honest.
    """
    dw = torch.randn(6, 8, device="cuda")
    a, b, _ = refactorize(dw, rank=R)
    assert a.shape == (R, 8) and b.shape == (6, R)
    assert a.device.type == "cuda" and b.device.type == "cuda"


# --- SVD aggregation vs the exact average (D2's requirement) ---------------

def test_single_client_aggregation_is_near_lossless():
    """With one client, ΔW̄ IS that client's rank-16 update, so re-factorizing to
    rank 16 must recover it."""
    a = make_adapter(1)
    sd, cfg, stats = aggregate_svd([a], [1.0], rank=R)
    err = exact_average_error([a], [1.0], sd, cfg)
    assert err["max_rel_err"] < 1e-3
    assert stats["reconstruction_error_max"] < 1e-3


def test_identical_clients_aggregate_to_themselves():
    """Three copies of the same adapter average to that adapter — rank stays 16,
    so truncation must cost nothing."""
    a = make_adapter(2)
    clients = [a, a, a]
    sd, cfg, _ = aggregate_svd(clients, [1, 1, 1], rank=R)
    err = exact_average_error(clients, [1, 1, 1], sd, cfg)
    assert err["max_rel_err"] < 1e-3


def test_svd_aggregate_beats_naive_against_the_exact_average():
    """The whole justification for D2. Distinct clients average to rank up to 48;
    SVD keeps the dominant 16, naive factor-averaging injects cross terms."""
    clients = [make_adapter(s) for s in (3, 4, 5)]
    weights = [199, 239, 222]

    sd, cfg, _ = aggregate_svd(clients, weights, rank=R)
    svd_err = exact_average_error(clients, weights, sd, cfg)

    n_sd, n_cfg = aggregate_naive(clients, weights)
    naive_err = exact_average_error(clients, weights, n_sd, n_cfg)

    assert svd_err["max_rel_err"] < naive_err["max_rel_err"]
    assert naive_err["max_rel_err"] > 0.5      # naive is badly wrong, not marginally


def test_weighting_actually_shifts_the_result():
    """A client weighted to ~1 should dominate the aggregate."""
    a, b = make_adapter(6), make_adapter(7)
    sd, cfg, _ = aggregate_svd([a, b], [1_000_000, 1], rank=R)
    got = delta_weights(sd, cfg, [PREFIXES[0]])[PREFIXES[0]]
    want_a = delta_weights(*a, [PREFIXES[0]])[PREFIXES[0]]
    want_b = delta_weights(*b, [PREFIXES[0]])[PREFIXES[0]]
    to_a = torch.linalg.norm(got - want_a) / torch.linalg.norm(want_a)
    to_b = torch.linalg.norm(got - want_b) / torch.linalg.norm(want_b)
    assert to_a < to_b


def test_aggregate_output_is_a_valid_rank_r_adapter():
    clients = [make_adapter(s) for s in (8, 9)]
    sd, cfg, stats = aggregate_svd(clients, [1, 1], rank=R)
    assert cfg["r"] == R
    assert cfg["lora_alpha"] == R          # scaling exactly 1.0
    assert cfg["use_rslora"] is False
    assert module_prefixes(sd) == PREFIXES
    assert stats["n_modules"] == len(PREFIXES)
    for p in PREFIXES:
        assert sd[f"{p}.lora_A.weight"].shape == (R, IN_F)
        assert sd[f"{p}.lora_B.weight"].shape == (OUT_F, R)


def test_aggregate_is_deterministic():
    """svd_lowrank uses randomized projections; the run must still be repeatable
    or a cluster adapter is not reproducible from its inputs."""
    clients = [make_adapter(s) for s in (10, 11, 12)]
    torch.manual_seed(0)
    sd1, cfg1, _ = aggregate_svd(clients, [1, 2, 3], rank=R)
    torch.manual_seed(0)
    sd2, cfg2, _ = aggregate_svd(clients, [1, 2, 3], rank=R)
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k], atol=1e-5), k


def test_reconstruction_error_is_reported_not_hidden():
    """Truncation loss must appear in the stats — three distinct rank-16 clients
    cannot compress to rank 16 for free, and the record should say so."""
    clients = [make_adapter(s) for s in (13, 14, 15)]
    _, _, stats = aggregate_svd(clients, [1, 1, 1], rank=R)
    assert stats["reconstruction_error_mean"] > 0.0
    assert stats["reconstruction_error_max"] >= stats["reconstruction_error_mean"]
    assert stats["reconstruction_error_min"] <= stats["reconstruction_error_mean"]


def test_higher_rank_loses_less():
    clients = [make_adapter(s) for s in (16, 17, 18)]
    _, _, low = aggregate_svd(clients, [1, 1, 1], rank=4)
    _, _, high = aggregate_svd(clients, [1, 1, 1], rank=32)
    assert high["reconstruction_error_mean"] < low["reconstruction_error_mean"]


def test_module_union_across_clients():
    """A module only one client trained still reaches the cluster adapter."""
    full = make_adapter(19)
    partial_sd = {k: v for k, v in full[0].items() if PREFIXES[0] in k}
    sd, _, _ = aggregate_svd([(partial_sd, full[1]), full], [1, 1], rank=R)
    assert module_prefixes(sd) == PREFIXES


def test_non_unit_client_scaling_is_honoured():
    """A client trained with lora_alpha != r contributes its SCALED update."""
    scaled = make_adapter(20, cfg=make_cfg(lora_alpha=32))    # scaling 2.0
    plain = make_adapter(21)
    clients = [scaled, plain]
    sd, cfg, _ = aggregate_svd(clients, [1, 1], rank=32)
    err = exact_average_error(clients, [1, 1], sd, cfg)
    assert err["max_rel_err"] < 0.35     # rank-32 keeps most of a rank-32 average


def test_aggregate_does_not_mutate_clients():
    clients = [make_adapter(s) for s in (22, 23)]
    before = {k: v.clone() for k, v in clients[0][0].items()}
    aggregate_svd(clients, [1, 1], rank=R)
    for k, v in before.items():
        assert torch.equal(clients[0][0][k], v), k
