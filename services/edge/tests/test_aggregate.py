"""Cluster aggregation tests (D2), through edge's delegation to P2's package.

D2 requires the SVD aggregation be "unit-tested against the exact product
average, with naive averaging kept only as an ablation baseline". The D2
mathematics now lives only in ``cluster.aggregation`` (P2); what these check is
that edge's PEFT <-> ``LoRAAdapter`` bridge feeds it correctly and hands back an
adapter ``edge.merge`` can compose — and that the naive-baseline bias is still
demonstrated end to end rather than asserted.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("cluster", reason="edge.aggregate delegates D2 to P2's cluster package")

from edge.aggregate import (  # noqa: E402
    aggregate_naive,
    aggregate_svd,
    exact_average_error,
    from_cluster_adapter,
    normalize_weights,
    to_cluster_adapter,
)
from edge.merge import delta_weights, module_prefixes  # noqa: E402

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
    """The cluster's path is exact (QR + LAPACK SVD, no random projection), so
    the same inputs must give the same cluster adapter, bit for bit."""
    clients = [make_adapter(s) for s in (10, 11, 12)]
    sd1, _, _ = aggregate_svd(clients, [1, 2, 3], rank=R)
    sd2, _, _ = aggregate_svd(clients, [1, 2, 3], rank=R)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k


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


def test_clients_must_share_target_modules():
    """P2's aggregator refuses clients that trained different modules — the
    contract pins q/k/v/o for everyone, so a mismatch is an error, not a union."""
    full = make_adapter(19)
    partial_sd = {k: v for k, v in full[0].items() if PREFIXES[0] in k}
    with pytest.raises(ValueError, match="target_modules"):
        aggregate_svd([(partial_sd, full[1]), full], [1, 1], rank=R)


def test_non_unit_client_scaling_is_honoured():
    """A client trained with lora_alpha != r contributes its SCALED update."""
    scaled = make_adapter(20, cfg=make_cfg(lora_alpha=32))    # scaling 2.0
    plain = make_adapter(21)
    clients = [scaled, plain]
    sd, cfg, _ = aggregate_svd(clients, [1, 1], rank=32)
    err = exact_average_error(clients, [1, 1], sd, cfg)
    # Two rank-16 clients average to rank <= 32, so an exact rank-32 truncation
    # loses nothing — IF the scaling-2.0 client was weighted by its scaled ΔW.
    assert err["max_rel_err"] < 1e-5


def test_aggregate_does_not_mutate_clients():
    clients = [make_adapter(s) for s in (22, 23)]
    before = {k: v.clone() for k, v in clients[0][0].items()}
    aggregate_svd(clients, [1, 1], rank=R)
    for k, v in before.items():
        assert torch.equal(clients[0][0][k], v), k


# --- the bridge itself ------------------------------------------------------

def test_scaling_is_folded_into_lora_b():
    """The cluster computes an unscaled B @ A, so the bridge must fold PEFT's
    s = lora_alpha / r into B: the LoRAAdapter's product IS the client's ΔW."""
    sd, cfg = make_adapter(30, cfg=make_cfg(lora_alpha=32))       # s = 2.0
    ad = to_cluster_adapter(sd, cfg)
    assert ad.alpha == ad.rank                                     # scaling 1.0
    want = delta_weights(sd, cfg, [PREFIXES[0]])[PREFIXES[0]].numpy()
    got = ad.delta_w("q_proj", layer=0)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_round_trip_keeps_the_clients_key_names():
    sd, cfg = make_adapter(31)
    ad = to_cluster_adapter(sd, cfg)
    from edge.aggregate import _key_map
    back = from_cluster_adapter(ad, _key_map([(sd, cfg)]))
    assert set(back) == set(sd)
    for k in sd:
        assert torch.equal(back[k], sd[k].float()), k


def test_edge_result_is_p2s_reference_aggregate():
    """Edge's served path must equal P2's dense ``aggregate_svd`` reference —
    the oracle the cluster's own suite holds ``aggregate_svd_lowrank`` to."""
    from cluster.aggregation import aggregate_svd as cluster_reference

    clients = [make_adapter(s) for s in (32, 33, 34)]
    weights = [80, 52, 170]
    sd, cfg, _ = aggregate_svd(clients, weights, rank=R)
    ref = cluster_reference(iter([to_cluster_adapter(*c) for c in clients]), weights, rank=R)
    for p in PREFIXES:
        module = p.rsplit(".", 1)[1]
        got = delta_weights(sd, cfg, [p])[p].double().numpy()
        want = ref.delta_w(module, layer=0)
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-5, p


def test_stats_name_the_aggregator_that_ran():
    _, _, stats = aggregate_svd([make_adapter(35)], [1.0], rank=R)
    assert stats["aggregator"] == "cluster.aggregation.aggregate_svd_lowrank"
