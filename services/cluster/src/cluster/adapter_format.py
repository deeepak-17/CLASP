"""P1's LoRA adapter format, as wired into the cluster pipeline (Week 2, Thu).

An adapter is, per transformer layer and per target attention module
(q/k/v/o projections):
    lora_A: (r, in_features)   — down-projection
    lora_B: (out_features, r)  — up-projection
    delta_W = lora_B @ lora_A, applied as W0 + (alpha / r) * delta_W

A real trained adapter has one (lora_A, lora_B) pair PER LAYER, not one pair
for the whole model — so ``modules`` is keyed first by layer index, then by
module name: ``modules[layer][module] == {"lora_A": ..., "lora_B": ...}``.
(Bug fix, Week 5 review: this used to be a flat ``{module: pair}`` dict with
no layer index at all, so it could only ever represent a single layer.)

P1 saves adapters as a torch ``state_dict`` with PEFT-style keys carrying a
``layers.<i>.`` segment (e.g. ``...layers.3.self_attn.q_proj.lora_A.weight``);
this module converts between that format and the flat numpy list the
aggregator/Flower wire uses. A key with no ``layers.<i>.`` segment is still
accepted for a single-layer adapter (``num_layers=1``), read as layer 0, so
older single-layer dumps keep working.

Contract constants (contracts v1.0): rank 16, target modules q/k/v/o.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

DEFAULT_RANK = 16
DEFAULT_ALPHA = 32.0
TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

_A_KEY = "lora_A"
_B_KEY = "lora_B"

# Matches a '...layers.<i>....' segment anywhere in a state_dict key, e.g.
# 'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight' -> '3'.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class AdapterFormatError(ValueError):
    """Adapter does not conform to the frozen contract."""


@dataclass
class LoRAAdapter:
    """In-memory adapter: per-(layer, module) numpy (lora_A, lora_B) pairs."""

    rank: int = DEFAULT_RANK
    alpha: float = DEFAULT_ALPHA
    target_modules: tuple[str, ...] = TARGET_MODULES
    num_layers: int = 1
    # layer_index -> module name -> {"lora_A": (r, in), "lora_B": (out, r)}
    modules: dict[int, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def validate(self) -> None:
        if self.num_layers < 1:
            raise AdapterFormatError(f"num_layers must be >= 1, got {self.num_layers}")
        for layer in self.layer_indices:
            if layer not in self.modules:
                raise AdapterFormatError(f"missing layer {layer}")
            layer_modules = self.modules[layer]
            for name in self.target_modules:
                if name not in layer_modules:
                    raise AdapterFormatError(
                        f"layer {layer}: missing target module {name!r}"
                    )
                pair = layer_modules[name]
                if set(pair) != {_A_KEY, _B_KEY}:
                    raise AdapterFormatError(
                        f"layer {layer}, module {name!r} must have exactly lora_A "
                        f"and lora_B, got {sorted(pair)}"
                    )
                a, b = pair[_A_KEY], pair[_B_KEY]
                if a.ndim != 2 or b.ndim != 2:
                    raise AdapterFormatError(
                        f"layer {layer}, module {name!r}: lora_A/lora_B must be 2-D"
                    )
                if a.shape[0] != self.rank:
                    raise AdapterFormatError(
                        f"layer {layer}, module {name!r}: lora_A rank {a.shape[0]} "
                        f"!= contract rank {self.rank}"
                    )
                if b.shape[1] != self.rank:
                    raise AdapterFormatError(
                        f"layer {layer}, module {name!r}: lora_B rank {b.shape[1]} "
                        f"!= contract rank {self.rank}"
                    )

    def delta_w(self, module: str, layer: int = 0) -> np.ndarray:
        """Reconstruct delta_W = B @ A for one (layer, target module) pair (Week 3, Tue)."""
        pair = self.modules[layer][module]
        return pair[_B_KEY] @ pair[_A_KEY]

    # --- flat ndarray list view (order used on the Flower wire) -------------

    def to_ndarrays(self) -> list[np.ndarray]:
        """[A, B] per module, layer-major then target_modules order — the
        Flower parameter order: layer 0's modules, then layer 1's, etc."""
        self.validate()
        out: list[np.ndarray] = []
        for layer in self.layer_indices:
            for name in self.target_modules:
                out.append(self.modules[layer][name][_A_KEY])
                out.append(self.modules[layer][name][_B_KEY])
        return out

    @classmethod
    def from_ndarrays(
        cls,
        arrays: list[np.ndarray],
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
        num_layers: int = 1,
    ) -> LoRAAdapter:
        per_layer = 2 * len(target_modules)
        expected = per_layer * num_layers
        if len(arrays) != expected:
            raise AdapterFormatError(
                f"expected {expected} arrays (A+B per module, {num_layers} "
                f"layer(s)), got {len(arrays)}"
            )
        modules: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        idx = 0
        for layer in range(num_layers):
            layer_modules: dict[str, dict[str, np.ndarray]] = {}
            for name in target_modules:
                layer_modules[name] = {_A_KEY: arrays[idx], _B_KEY: arrays[idx + 1]}
                idx += 2
            modules[layer] = layer_modules
        adapter = cls(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            num_layers=num_layers,
            modules=modules,
        )
        adapter.validate()
        return adapter

    # --- P1 state_dict view --------------------------------------------------

    def to_state_dict(self) -> dict[str, np.ndarray]:
        """PEFT-style keys: 'layers.<i>.<module>.lora_A.weight' / '...lora_B.weight'."""
        self.validate()
        sd: dict[str, np.ndarray] = {}
        for layer in self.layer_indices:
            for name in self.target_modules:
                sd[f"layers.{layer}.{name}.{_A_KEY}.weight"] = self.modules[layer][name][_A_KEY]
                sd[f"layers.{layer}.{name}.{_B_KEY}.weight"] = self.modules[layer][name][_B_KEY]
        return sd

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, np.ndarray],
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
        num_layers: int = 1,
    ) -> LoRAAdapter:
        """Parse PEFT-style keys, e.g.
        'base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight'.

        The layer index is read from a 'layers.<i>.' segment anywhere in the
        key. A key with no such segment is accepted only for a single-layer
        adapter (num_layers=1), where it is assigned to layer 0 — this keeps
        legacy flat (layer-less) single-layer dumps working.
        """
        modules: dict[int, dict[str, dict[str, np.ndarray]]] = {
            layer: {m: {} for m in target_modules} for layer in range(num_layers)
        }
        for key, tensor in state_dict.items():
            layer_match = _LAYER_RE.search(key)
            if layer_match is not None:
                layer = int(layer_match.group(1))
            elif num_layers == 1:
                layer = 0
            else:
                raise AdapterFormatError(
                    f"state_dict key {key!r} has no 'layers.<i>.' segment, "
                    f"but num_layers={num_layers} > 1"
                )
            if layer not in modules:
                raise AdapterFormatError(
                    f"state_dict key {key!r} references layer {layer}, outside "
                    f"the expected range [0, {num_layers})"
                )
            matched = False
            for name in target_modules:
                if matched:
                    break
                for part in (_A_KEY, _B_KEY):
                    if key.endswith(f"{name}.{part}.weight"):
                        # raise on duplicate keys instead of silently
                        # overwriting; also break immediately after the first
                        # match so a single key cannot populate multiple slots.
                        if part in modules[layer][name]:
                            raise AdapterFormatError(
                                f"duplicate state_dict key for layer {layer} "
                                f"{name}.{part}: {key!r} already seen"
                            )
                        modules[layer][name][part] = np.asarray(tensor)
                        matched = True
                        break
            if not matched:
                raise AdapterFormatError(f"unrecognized state_dict key {key!r}")
        adapter = cls(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            num_layers=num_layers,
            modules=modules,
        )
        adapter.validate()
        return adapter


def random_adapter(
    in_features: int,
    out_features: int,
    rank: int = DEFAULT_RANK,
    alpha: float = DEFAULT_ALPHA,
    target_modules: tuple[str, ...] = TARGET_MODULES,
    num_layers: int = 1,
    seed: int | None = None,
    dtype: np.dtype = np.float32,
) -> LoRAAdapter:
    """Adapter with PEFT-style init: A ~ N(0, 0.02), B = 0, one pair per layer."""
    rng = np.random.default_rng(seed)
    modules = {
        layer: {
            name: {
                _A_KEY: (rng.normal(0.0, 0.02, size=(rank, in_features))).astype(dtype),
                _B_KEY: np.zeros((out_features, rank), dtype=dtype),
            }
            for name in target_modules
        }
        for layer in range(num_layers)
    }
    return LoRAAdapter(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        num_layers=num_layers,
        modules=modules,
    )
