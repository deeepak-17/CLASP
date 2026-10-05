"""Pre-merged composite adapters (D6): alpha*dW_cluster + beta*dW_client as one LoRA.

Two low-rank deltas sum exactly into one LoRA of rank r_c + r_l by
concatenating factors along the rank axis:

    alpha*s_c*B_c@A_c + beta*s_l*B_l@A_l = [alpha*s_c*B_c | beta*s_l*B_l] @ [A_c ; A_l]

where s = lora_alpha / r (or lora_alpha / sqrt(r) under rsLoRA) is the scaling
PEFT applies at forward time. Each part's coefficient and scaling are folded
into its B block and the composite's own scaling is pinned to 1.0
(lora_alpha == r), so the stored tensors mean exactly what they say. This is
the same construction as ``edge.merge.compose`` — reimplemented in numpy so
the registry never needs torch — and it is exact, not an SVD approximation.

Keys from either convention (PEFT ``...layers.<i>.self_attn.<m>...`` or the
cluster's short ``layers.<i>.<m>...``) are matched on (layer, module); output
keys follow the client adapter, which is what the edge loads.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
from contracts import LoRAHyperParams
from safetensors.numpy import load

_FACTOR_RE = re.compile(r"^(?P<prefix>.+)\.(?P<part>lora_A|lora_B)\.weight$")
_LAYER_MODULE_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.(?:.*\.)?(?P<module>[A-Za-z0-9_]+)$")

#: Fields of a PEFT adapter_config that the registry also records.
_RECORDED_FIELDS = (("rank", "r"), ("lora_alpha", "lora_alpha"))


class CompositeError(ValueError):
    """The parts cannot be merged into one exact composite."""


@dataclass(frozen=True)
class PartSpec:
    """One input to the composite: payload bytes, registry hparams, coefficient."""
    payload: bytes
    hparams: LoRAHyperParams
    coefficient: float


@dataclass(frozen=True)
class Composite:
    payload: bytes
    rank: int
    hparams: LoRAHyperParams


@dataclass(frozen=True)
class _Part:
    factors: dict[str, tuple[str, np.ndarray, np.ndarray]]  # module id -> (prefix, A, B)
    config: dict | None
    scale: float
    rank: int
    coefficient: float


def serialize_canonical(tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> bytes:
    """float32 safetensors with a sorted, 8-byte-aligned header.

    ``safetensors.numpy.save`` orders its header through a hash map, so the
    same tensors can serialize to different bytes run to run. Composites are
    content-addressed by sha256 (D9), so the registry writes them itself.
    """
    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    chunks, offset = [], 0
    for name in sorted(tensors):
        data = np.ascontiguousarray(tensors[name], dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return len(raw).to_bytes(8, "little") + raw + b"".join(chunks)


def _embedded_config(payload: bytes) -> dict | None:
    header_len = int.from_bytes(payload[:8], "little")
    header = json.loads(payload[8 : 8 + header_len].decode("utf-8"))
    raw = (header.get("__metadata__") or {}).get("adapter_config")
    return json.loads(raw) if raw else None


def _module_id(prefix: str) -> str:
    """(layer, module) identity shared by both key conventions."""
    m = _LAYER_MODULE_RE.search(prefix)
    return f"{m['layer']}.{m['module']}" if m else prefix


def _factors(tensors: dict[str, np.ndarray]) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for key, value in tensors.items():
        m = _FACTOR_RE.match(key)
        if m is None:
            raise CompositeError(f"not a LoRA factor key: {key!r}")
        pairs.setdefault(m["prefix"], {})[m["part"]] = value
    out = {}
    for prefix, parts in pairs.items():
        missing = {"lora_A", "lora_B"} - parts.keys()
        if missing:
            raise CompositeError(f"{prefix}: missing {sorted(missing)[0]}")
        a, b = parts["lora_A"], parts["lora_B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise CompositeError(f"{prefix}: inconsistent factor shapes A{a.shape} B{b.shape}")
        out[_module_id(prefix)] = (prefix, a, b)
    return out


def _check_recorded(cfg: dict, hparams: LoRAHyperParams, label: str) -> None:
    for hp_field, cfg_field in _RECORDED_FIELDS:
        if cfg_field in cfg and cfg[cfg_field] != getattr(hparams, hp_field):
            raise CompositeError(
                f"{label}: registry {hp_field}={getattr(hparams, hp_field)} but the payload's "
                f"embedded adapter_config has {cfg_field}={cfg[cfg_field]}"
            )


def _prepare(spec: PartSpec, label: str) -> _Part:
    if not math.isfinite(spec.coefficient):
        raise CompositeError(f"{label}: coefficient must be finite, got {spec.coefficient}")
    try:
        tensors = load(spec.payload)
        cfg = _embedded_config(spec.payload)
    except Exception as e:  # safetensors raises its own error types
        raise CompositeError(f"{label}: unreadable safetensors payload: {e}") from e
    if cfg is not None:
        _check_recorded(cfg, spec.hparams, label)
    rank, lora_alpha = spec.hparams.rank, spec.hparams.lora_alpha
    rslora = bool(cfg and cfg.get("use_rslora"))
    scale = lora_alpha / math.sqrt(rank) if rslora else lora_alpha / rank
    return _Part(_factors(tensors), cfg, scale, rank, spec.coefficient)


def _merge_module(module: str, live: list[_Part]) -> tuple[np.ndarray, np.ndarray]:
    a_blocks, b_blocks = [], []
    for part in live:
        if module not in part.factors:
            continue
        _, a, b = part.factors[module]
        fold = part.coefficient * part.scale
        a_blocks.append(a.astype(np.float32, copy=False))
        b32 = b.astype(np.float32, copy=False)
        # When the fold is exactly 1.0 skip the multiply so tensors survive bitwise.
        b_blocks.append(b32 if fold == 1.0 else b32 * np.float32(fold))
    if len({a.shape[1] for a in a_blocks}) > 1 or len({b.shape[0] for b in b_blocks}) > 1:
        raise CompositeError(f"module {module}: shape mismatch between cluster and client")
    return np.concatenate(a_blocks, axis=0), np.concatenate(b_blocks, axis=1)


def _output_prefix(module: str, cluster: _Part, client: _Part) -> str:
    source = client if module in client.factors else cluster
    return source.factors[module][0]


def _composite_config(rank: int, modules: tuple[str, ...], template: dict | None,
                      dropout: float) -> dict:
    cfg = dict(template) if template else {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
        "lora_dropout": dropout,
    }
    cfg.update({
        "r": rank, "lora_alpha": rank, "use_rslora": False, "rank_pattern": {},
        "alpha_pattern": {}, "inference_mode": True, "target_modules": list(modules),
    })
    return cfg


def build_composite(cluster: PartSpec, client: PartSpec) -> Composite:
    """Merge ``cluster`` (alpha) and ``client`` (beta) into one exact composite.

    A part with coefficient 0 is pruned, so alpha=0 yields the client adapter
    bit for bit (same delta, rank r instead of a zero-padded 2r).
    """
    c, k = _prepare(cluster, "cluster"), _prepare(client, "client")
    live = [p for p in (c, k) if p.coefficient != 0.0]
    if not live:
        raise CompositeError("alpha and beta are both 0 — the composite would be empty")

    modules = sorted({m for p in live for m in p.factors})
    tensors: dict[str, np.ndarray] = {}
    for module in modules:
        a, b = _merge_module(module, live)
        prefix = _output_prefix(module, c, k)
        tensors[f"{prefix}.lora_A.weight"] = a
        tensors[f"{prefix}.lora_B.weight"] = b

    rank = sum(p.rank for p in live)
    target_modules = tuple(sorted({m.rsplit(".", 1)[-1] for m in modules}))
    cfg = _composite_config(rank, target_modules, k.config or c.config, client.hparams.dropout)
    payload = serialize_canonical(
        tensors, {"adapter_config": json.dumps(cfg, sort_keys=True), "format": "pt"}
    )
    hparams = LoRAHyperParams(
        rank=rank, lora_alpha=rank, dropout=client.hparams.dropout,
        target_modules=target_modules, alpha=cluster.coefficient, beta=client.coefficient,
    )
    return Composite(payload=payload, rank=rank, hparams=hparams)
