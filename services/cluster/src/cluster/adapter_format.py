"""P1's LoRA adapter format, as wired into the cluster pipeline (Week 2, Thu).

An adapter is, per target attention module (q/k/v/o projections):
    lora_A: (r, in_features)   — down-projection
    lora_B: (out_features, r)  — up-projection
    delta_W = lora_B @ lora_A, applied as W0 + (alpha / r) * delta_W

P1 saves adapters as a torch ``state_dict`` with PEFT-style keys
(``...<module>.lora_A.weight`` / ``...<module>.lora_B.weight``); this module
converts between that format and the flat numpy view the aggregator uses.

Contract constants (contracts v1.0): rank 16, target modules q/k/v/o.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_RANK = 16
DEFAULT_ALPHA = 32.0
TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

_A_KEY = "lora_A"
_B_KEY = "lora_B"


class AdapterFormatError(ValueError):
    """Adapter does not conform to the frozen contract."""


@dataclass
class LoRAAdapter:
    """In-memory adapter: per-module (lora_A, lora_B) numpy pairs."""

    rank: int = DEFAULT_RANK
    alpha: float = DEFAULT_ALPHA
    target_modules: tuple[str, ...] = TARGET_MODULES
    # module name -> {"lora_A": (r, in), "lora_B": (out, r)}
    modules: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def validate(self) -> None:
        for name in self.target_modules:
            if name not in self.modules:
                raise AdapterFormatError(f"missing target module {name!r}")
            pair = self.modules[name]
            if set(pair) != {_A_KEY, _B_KEY}:
                raise AdapterFormatError(
                    f"module {name!r} must have exactly lora_A and lora_B, got {sorted(pair)}"
                )
            a, b = pair[_A_KEY], pair[_B_KEY]
            if a.ndim != 2 or b.ndim != 2:
                raise AdapterFormatError(f"module {name!r}: lora_A/lora_B must be 2-D")
            if a.shape[0] != self.rank:
                raise AdapterFormatError(
                    f"module {name!r}: lora_A rank {a.shape[0]} != contract rank {self.rank}"
                )
            if b.shape[1] != self.rank:
                raise AdapterFormatError(
                    f"module {name!r}: lora_B rank {b.shape[1]} != contract rank {self.rank}"
                )

    def delta_w(self, module: str) -> np.ndarray:
        """Reconstruct delta_W = B @ A for one target module (Week 3, Tue)."""
        pair = self.modules[module]
        return pair[_B_KEY] @ pair[_A_KEY]

    # --- flat ndarray list view (order used on the Flower wire) -------------

    def to_ndarrays(self) -> list[np.ndarray]:
        """[A, B] per module, in target_modules order — Flower parameter order."""
        self.validate()
        out: list[np.ndarray] = []
        for name in self.target_modules:
            out.append(self.modules[name][_A_KEY])
            out.append(self.modules[name][_B_KEY])
        return out

    @classmethod
    def from_ndarrays(
        cls,
        arrays: list[np.ndarray],
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
    ) -> "LoRAAdapter":
        if len(arrays) != 2 * len(target_modules):
            raise AdapterFormatError(
                f"expected {2 * len(target_modules)} arrays "
                f"(A+B per module), got {len(arrays)}"
            )
        modules = {
            name: {_A_KEY: arrays[2 * i], _B_KEY: arrays[2 * i + 1]}
            for i, name in enumerate(target_modules)
        }
        adapter = cls(rank=rank, alpha=alpha, target_modules=target_modules, modules=modules)
        adapter.validate()
        return adapter

    # --- P1 state_dict view --------------------------------------------------

    def to_state_dict(self) -> dict[str, np.ndarray]:
        """PEFT-style flat keys: '<module>.lora_A.weight' / '<module>.lora_B.weight'."""
        self.validate()
        sd: dict[str, np.ndarray] = {}
        for name in self.target_modules:
            sd[f"{name}.{_A_KEY}.weight"] = self.modules[name][_A_KEY]
            sd[f"{name}.{_B_KEY}.weight"] = self.modules[name][_B_KEY]
        return sd

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, np.ndarray],
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
    ) -> "LoRAAdapter":
        modules: dict[str, dict[str, np.ndarray]] = {m: {} for m in target_modules}
        for key, tensor in state_dict.items():
            matched = False
            for name in target_modules:
                if matched:
                    break
                for part in (_A_KEY, _B_KEY):
                    if key.endswith(f"{name}.{part}.weight"):
                        # BUG-FIX: raise on duplicate keys instead of silently
                        # overwriting; also break immediately after the first
                        # match so a single key cannot populate multiple slots.
                        if part in modules[name]:
                            raise AdapterFormatError(
                                f"duplicate state_dict key for {name}.{part}: "
                                f"{key!r} already seen"
                            )
                        modules[name][part] = np.asarray(tensor)
                        matched = True
                        break
            if not matched:
                raise AdapterFormatError(f"unrecognized state_dict key {key!r}")
        adapter = cls(rank=rank, alpha=alpha, target_modules=target_modules, modules=modules)
        adapter.validate()
        return adapter


def random_adapter(
    in_features: int,
    out_features: int,
    rank: int = DEFAULT_RANK,
    alpha: float = DEFAULT_ALPHA,
    target_modules: tuple[str, ...] = TARGET_MODULES,
    seed: int | None = None,
    dtype: np.dtype = np.float32,
) -> LoRAAdapter:
    """Adapter with PEFT-style init: A ~ N(0, 0.02), B = 0."""
    rng = np.random.default_rng(seed)
    modules = {
        name: {
            _A_KEY: (rng.normal(0.0, 0.02, size=(rank, in_features))).astype(dtype),
            _B_KEY: np.zeros((out_features, rank), dtype=dtype),
        }
        for name in target_modules
    }
    return LoRAAdapter(rank=rank, alpha=alpha, target_modules=target_modules, modules=modules)
