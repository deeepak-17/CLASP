"""Composite adapter merge — base ⊕ α·cluster ⊕ β·client (P1 Edge, W3 · E3.3).

D6 serves ONE pre-merged composite per developer. No runtime dual-adapter
stacking: the edge loads a single adapter and the NF4 base stays frozen and
untouched.

How the composite is built
--------------------------
A LoRA adapter contributes ΔW = s·B·A per target module, where s is the
adapter's own scaling (lora_alpha/r, or lora_alpha/sqrt(r) under rsLoRA). The
composite must satisfy

    ΔW_composite = α·s_cluster·B_c·A_c + β·s_client·B_l·A_l

Two ways to represent that as a single adapter:

  * **Rank concatenation** (what this does). Stack along the rank axis:
        A_comp = [A_c ; A_l]                       shape (r_c + r_l, in)
        B_comp = [α·s_c·B_c , β·s_l·B_l]           shape (out, r_c + r_l)
    with the composite's own scaling pinned to 1.0. The block-matrix product
    then reproduces the sum EXACTLY — no approximation anywhere.

  * **SVD truncation** back to rank r, the way D2 re-factorizes on the server.
    Smaller, but lossy.

Concatenation is the right call here specifically because of E3.4. The gate is
that α=0,β=0 recovers base and α=0,β=1 equals the client adapter alone. Under
concatenation those hold as identities; under SVD they would hold only to
within truncation error, and a correctness gate that can only say "close
enough" cannot distinguish a rounding artifact from a genuine sign error in the
merge. The cost is a rank-32 served adapter — traded knowingly, and revisitable
once D2's SVD lands and the same machinery can re-factorize.

Because the pinned config has lora_alpha == r, every scaling is exactly 1.0 and
the α=0,β=1 case reduces to multiplication by 1.0, which IEEE-754 performs
exactly. That identity is therefore bitwise, not merely close.

No cluster adapter exists yet
-----------------------------
P2's SVD aggregation does not produce one until W5+, so `make_stub_adapter`
mints a stand-in (zeros, or seeded random) with a matching config. Zeros is the
useful one for E3.4: it makes α's contribution provably inert, so any change in
output has to come from the client term.

Usage:
    python -m edge.merge --cluster stub:zeros --client path/to/adapter \\
        --alpha 0.5 --beta 1.0 --out composite/
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file

# Hyperparameters the contracts pin for Phase II. A rank or target-module
# mismatch between two adapters produces a silently wrong composite rather than
# an error, which is the failure mode P1's scope calls out by name — so the
# check is mandatory before any merge, not advisory.
CONTRACT_HYPERPARAMS = {
    "r": 16,
    "lora_alpha": 16,
    "target_modules": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "base_model_name_or_path": "deepseek-ai/deepseek-coder-1.3b-base",
}

# Adapter-config fields that change the MEANING of the stored tensors. Two
# adapters that disagree on any of these cannot be summed blockwise.
STRUCTURAL_FIELDS = ("peft_type", "use_rslora", "use_dora", "fan_in_fan_out", "lora_bias")


def scaling_of(cfg: Dict) -> float:
    """The multiplier PEFT applies to B·A at forward time."""
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def load_adapter(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Read a PEFT adapter directory into (state_dict, config)."""
    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    weights = path / "adapter_model.safetensors"
    sd = (load_file(str(weights)) if weights.exists()
          else torch.load(path / "adapter_model.bin", map_location="cpu"))
    return sd, cfg


def save_adapter(path: Path, sd: Dict[str, torch.Tensor], cfg: Dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in sd.items()},
              str(path / "adapter_model.safetensors"))


def module_prefixes(sd: Dict[str, torch.Tensor]) -> List[str]:
    """Target-module keys, e.g. '...layers.0.self_attn.q_proj', sorted."""
    return sorted({k.rsplit(".lora_A", 1)[0] for k in sd if ".lora_A" in k})


def _ab(sd: Dict[str, torch.Tensor], prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return sd[f"{prefix}.lora_A.weight"], sd[f"{prefix}.lora_B.weight"]


def delta_weights(sd: Dict[str, torch.Tensor], cfg: Dict,
                  prefixes: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """Materialize ΔW = s·B·A per module.

    This is the ground truth the E3.4 identities are checked against — the
    adapter's actual effect on the base weights, independent of how the A/B
    factors happen to be stored.

    WARNING: ΔW is dense and full-size. On the 1.3B model that is 96 modules of
    2048x2048 fp32, ~1.5 GB for one adapter — materializing several at once will
    exhaust RAM (it took down the test process before `merge_max_error` existed).
    Pass `prefixes` to do one module at a time, or use `merge_max_error`.
    """
    s = scaling_of(cfg)
    out = {}
    for prefix in (module_prefixes(sd) if prefixes is None else prefixes):
        a, b = _ab(sd, prefix)
        out[prefix] = s * (b.float() @ a.float())
    return out


def validate_compatibility(cfgs: Sequence[Dict], names: Sequence[str],
                           contract: Optional[Dict] = CONTRACT_HYPERPARAMS) -> Dict:
    """Reject adapters that cannot be legally composed. Raises ValueError.

    Checked in two directions: adapters against each other (they must describe
    the same base model and the same structural variant of LoRA), and each
    adapter against the frozen contract hyperparameters. Rank is deliberately
    NOT required to match between adapters — concatenation handles differing
    ranks fine — but it IS checked against the contract, because a rank that
    drifted from 16 means someone trained outside D8's envelope.
    """
    problems: List[str] = []
    ref, ref_name = cfgs[0], names[0]

    for cfg, name in zip(cfgs[1:], names[1:]):
        if cfg.get("base_model_name_or_path") != ref.get("base_model_name_or_path"):
            problems.append(
                f"base model mismatch: {ref_name}={ref.get('base_model_name_or_path')!r} "
                f"vs {name}={cfg.get('base_model_name_or_path')!r}")
        if set(cfg.get("target_modules") or []) != set(ref.get("target_modules") or []):
            problems.append(
                f"target_modules mismatch: {ref_name}={sorted(ref.get('target_modules') or [])} "
                f"vs {name}={sorted(cfg.get('target_modules') or [])}")
        for field in STRUCTURAL_FIELDS:
            if cfg.get(field) != ref.get(field):
                problems.append(
                    f"{field} mismatch: {ref_name}={ref.get(field)!r} vs {name}={cfg.get(field)!r}")

    if contract:
        for cfg, name in zip(cfgs, names):
            for key, expected in contract.items():
                actual = cfg.get(key)
                actual_cmp = set(actual) if isinstance(expected, set) and actual else actual
                if actual_cmp != expected:
                    problems.append(
                        f"{name}: {key}={actual!r} violates contract {expected!r}")

    if problems:
        raise ValueError("incompatible adapters:\n  - " + "\n  - ".join(problems))

    return {"base_model": ref.get("base_model_name_or_path"),
            "target_modules": sorted(ref.get("target_modules") or []),
            "ranks": {n: c["r"] for n, c in zip(names, cfgs)},
            "scalings": {n: scaling_of(c) for n, c in zip(names, cfgs)}}


def make_stub_adapter(reference_cfg: Dict, prefixes: Iterable[str],
                      shapes: Dict[str, Tuple[int, int]],
                      mode: str = "zeros", seed: int = 0
                      ) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Mint a stand-in cluster adapter until P2's aggregation produces a real one.

    mode='zeros'  - contributes exactly nothing. The honest default: it lets the
                    merge machinery be exercised end to end while keeping α's
                    term provably inert, so E3.4 can attribute every output
                    change to the client term alone.
    mode='random' - seeded Gaussian on both factors, so ΔW is nonzero and the
                    α path is genuinely exercised. NOT a trained adapter and it
                    will degrade quality — for machinery tests only.
    """
    if mode not in ("zeros", "random"):
        raise ValueError(f"mode must be 'zeros' or 'random', got {mode!r}")
    cfg = dict(reference_cfg)
    r = cfg["r"]
    gen = torch.Generator().manual_seed(seed)
    sd: Dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        in_f, out_f = shapes[prefix]
        if mode == "zeros":
            a = torch.zeros(r, in_f)
            b = torch.zeros(out_f, r)
        else:
            # Same init shape as PEFT's default (A ~ N(0, 1/r), B nonzero here so
            # the product does not vanish the way a freshly-initialized adapter's
            # would).
            a = torch.randn(r, in_f, generator=gen) * (1.0 / math.sqrt(r))
            b = torch.randn(out_f, r, generator=gen) * (1.0 / math.sqrt(out_f))
        sd[f"{prefix}.lora_A.weight"] = a
        sd[f"{prefix}.lora_B.weight"] = b
    return sd, cfg


def shapes_from(sd: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, int]]:
    """{module_prefix: (in_features, out_features)} read off the A/B factors."""
    out = {}
    for prefix in module_prefixes(sd):
        a, b = _ab(sd, prefix)
        out[prefix] = (a.shape[1], b.shape[0])
    return out


def compose(parts: Sequence[Tuple[Dict[str, torch.Tensor], Dict, float]],
            prune_zero: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Concatenate weighted adapters into one composite adapter.

    `parts` is [(state_dict, config, coefficient), ...] — for D6 that is
    [(cluster, cfg_c, α), (client, cfg_l, β)].

    Each part's own scaling is folded into its B block and the composite's
    scaling is pinned to 1.0 (lora_alpha == r), so the stored tensors mean
    exactly what they say and no downstream consumer has to know the parts'
    original alphas.

    `prune_zero` drops parts whose coefficient is 0. That is what makes α=0
    return a rank-16 composite identical to the client adapter rather than a
    rank-32 one padded with zero blocks — same ΔW either way, but the pruned
    form is the one that is bitwise identical to the unmerged baseline, which
    is what E3.4 wants to assert.
    """
    live = [(sd, cfg, coef) for sd, cfg, coef in parts if not (prune_zero and coef == 0.0)]

    # Union of modules: a module touched by only one part still belongs in the
    # composite, contributing that part's block alone.
    all_prefixes = sorted({p for sd, _, _ in live for p in module_prefixes(sd)})

    composite: Dict[str, torch.Tensor] = {}
    for prefix in all_prefixes:
        a_blocks, b_blocks = [], []
        for sd, cfg, coef in live:
            if f"{prefix}.lora_A.weight" not in sd:
                continue
            a, b = _ab(sd, prefix)
            a_blocks.append(a)
            # coef * scaling folded here. When both are exactly 1.0 (the pinned
            # config with beta=1) this multiplication is the identity, so the
            # tensor survives bitwise.
            b_blocks.append(b * (coef * scaling_of(cfg)))
        if not a_blocks:
            continue
        composite[f"{prefix}.lora_A.weight"] = torch.cat(a_blocks, dim=0)
        composite[f"{prefix}.lora_B.weight"] = torch.cat(b_blocks, dim=1)

    total_rank = sum(cfg["r"] for _, cfg, _ in live)
    if total_rank == 0:
        # Everything pruned (α=β=0). Emit a well-formed rank-r adapter of zeros
        # rather than an empty one, so it still loads and provably adds nothing.
        ref_sd, ref_cfg, _ = parts[0]
        composite, _ = make_stub_adapter(ref_cfg, module_prefixes(ref_sd),
                                         shapes_from(ref_sd), mode="zeros")
        total_rank = ref_cfg["r"]

    cfg = dict(parts[0][1])
    cfg["r"] = total_rank
    cfg["lora_alpha"] = total_rank      # => scaling exactly 1.0
    cfg["use_rslora"] = False           # scaling already folded in; do not re-scale
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}
    cfg["inference_mode"] = True
    return composite, cfg


def composite_delta(parts: Sequence[Tuple[Dict[str, torch.Tensor], Dict, float]]
                    ) -> Dict[str, torch.Tensor]:
    """Reference implementation: the weighted sum computed the obvious, slow way.

    Deliberately NOT how `compose` works. E3.4 checks the concatenated composite
    against this, so a bug in the block-matrix reasoning cannot hide behind the
    same bug in its own test.
    """
    total: Dict[str, torch.Tensor] = {}
    for sd, cfg, coef in parts:
        for prefix, dw in delta_weights(sd, cfg).items():
            contribution = coef * dw
            total[prefix] = total[prefix] + contribution if prefix in total else contribution
    return total


def merge_max_error(parts: Sequence[Tuple[Dict[str, torch.Tensor], Dict, float]],
                    sd: Dict[str, torch.Tensor], cfg: Dict) -> Dict[str, float]:
    """Largest discrepancy between the composite and the reference sum.

    Streams module by module and frees as it goes, holding two ΔW matrices at a
    time instead of two full sets — the difference between ~16 MB and ~3 GB on
    the 1.3B model.

    Returns absolute AND relative error. Relative is the one to threshold on:
    ΔW entries here run to |80|, where fp32 rounding alone lands around 1.5e-5
    absolute. Judging that against an absolute tolerance would flag correct
    arithmetic as a merge bug.
    """
    worst_abs, worst_rel = 0.0, 0.0
    for prefix in module_prefixes(sd):
        got = delta_weights(sd, cfg, [prefix])[prefix]
        ref = None
        for part_sd, part_cfg, coef in parts:
            if f"{prefix}.lora_A.weight" not in part_sd:
                continue
            term = coef * delta_weights(part_sd, part_cfg, [prefix])[prefix]
            ref = term if ref is None else ref + term
        if ref is None:
            continue
        abs_err = (got - ref).abs().max().item()
        scale = max(ref.abs().max().item(), 1e-12)
        worst_abs = max(worst_abs, abs_err)
        worst_rel = max(worst_rel, abs_err / scale)
        del got, ref
    return {"max_abs_err": worst_abs, "max_rel_err": worst_rel}


def _resolve(spec: str, reference: Optional[Tuple[Dict, Dict]], seed: int
             ) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Adapter spec -> (state_dict, config). 'stub:zeros' / 'stub:random' mint
    a stand-in shaped like `reference`; anything else is a directory path."""
    if spec.startswith("stub:"):
        if reference is None:
            raise SystemExit("a stub adapter needs a real adapter to copy its shape from")
        ref_sd, ref_cfg = reference
        return make_stub_adapter(ref_cfg, module_prefixes(ref_sd),
                                 shapes_from(ref_sd), mode=spec.split(":", 1)[1], seed=seed)
    return load_adapter(Path(spec))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose base + a*cluster + b*client (D6)")
    ap.add_argument("--client", required=True, help="path to the client adapter")
    ap.add_argument("--cluster", default="stub:zeros",
                    help="path to the cluster adapter, or stub:zeros / stub:random")
    ap.add_argument("--alpha", type=float, default=1.0, help="cluster coefficient")
    ap.add_argument("--beta", type=float, default=1.0, help="client coefficient")
    ap.add_argument("--seed", type=int, default=0, help="seed for a stub cluster adapter")
    ap.add_argument("--no-contract-check", action="store_true",
                    help="skip validation against the frozen contract hyperparams")
    ap.add_argument("--out", required=True, help="destination directory for the composite")
    args = ap.parse_args()

    client_sd, client_cfg = load_adapter(Path(args.client))
    cluster_sd, cluster_cfg = _resolve(args.cluster, (client_sd, client_cfg), args.seed)

    info = validate_compatibility(
        [cluster_cfg, client_cfg], ["cluster", "client"],
        contract=None if args.no_contract_check else CONTRACT_HYPERPARAMS)

    parts = [(cluster_sd, cluster_cfg, args.alpha), (client_sd, client_cfg, args.beta)]
    sd, cfg = compose(parts)

    # Self-check every merge, not just the test suite: the concatenated result
    # must equal the slow reference sum. This is cheap next to a training run
    # and it is the last line of defence before a wrong composite is served.
    err = merge_max_error(parts, sd, cfg)
    if not math.isfinite(err["max_rel_err"]) or err["max_rel_err"] > 1e-5:
        raise SystemExit(
            f"merge self-check FAILED: relative error {err['max_rel_err']:.3e} "
            f"(absolute {err['max_abs_err']:.3e}) exceeds fp32 rounding")

    save_adapter(Path(args.out), sd, cfg)
    meta = {
        "task": "E3.3 composite merge (D6)",
        "alpha": args.alpha, "beta": args.beta,
        "cluster_source": args.cluster, "client_source": args.client,
        "composite_rank": cfg["r"], "composite_scaling": scaling_of(cfg),
        "n_modules": len(module_prefixes(sd)),
        "merge_self_check": err,
        "compatibility": info,
        "d3_deviation": ("cluster is a stand-in; P2's SVD aggregation produces no real "
                         "cluster adapter until W5+" if args.cluster.startswith("stub:") else None),
    }
    (Path(args.out) / "merge_manifest.json").write_text(json.dumps(meta, indent=2),
                                                        encoding="utf-8")
    print(f"composite written to {Path(args.out).resolve()}")
    print(f"  rank {cfg['r']} (scaling {scaling_of(cfg):g}), {meta['n_modules']} modules")
    print(f"  alpha={args.alpha}  beta={args.beta}")
    print(f"  self-check: rel {err['max_rel_err']:.3e}, abs {err['max_abs_err']:.3e}")


if __name__ == "__main__":
    main()
