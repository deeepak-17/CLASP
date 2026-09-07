"""Edge <-> Cluster adapter wire format (Integration Sprint · B3, seam A).

Two key conventions exist in this repository and neither is going away:

    cluster : ``layers.3.q_proj.lora_A.weight``
              (``cluster.adapter_format.LoRAAdapter.to_state_dict``)

    PEFT    : ``base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight``
              (what ``edge.merge.load_adapter`` reads off disk, and what the six
              real trained client adapters actually contain -- verified: 192
              tensors, 24 layers x q/k/v/o, r=16, fp32)

``LoRAAdapter.from_state_dict`` already parses BOTH (it matches a ``layers.<i>.``
segment anywhere in the key plus a ``<module>.<lora_A|lora_B>.weight`` suffix),
and ``LoRAAdapter.to_peft_state_dict`` can emit the PEFT form. This module is
the Edge-side half: it does not assume the cluster emits PEFT keys. Every
payload coming back off the wire is normalised through :func:`to_peft_keys`
first, so if Cluster ever reverts to its short internal convention -- the
sprint's F1 fallback, "let Cluster return its own key convention and apply an
explicit key-map on the Edge side" -- the Edge path keeps working with no
change. The map is explicit, bidirectional, and layer-generic: it is driven by
a regex over the layer index, never by a hard-coded layer or module list.

Dtype contract
--------------
FP32 on the wire, both directions, enforced (not merely documented) by
:func:`as_fp32` and by ``AdapterUpload``'s validator on the cluster side. The
real adapters are already fp32; the aggregation promotes to float64 internally
and casts back, so fp32 is the only dtype that ever crosses a seam.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: The prefix PEFT puts in front of every LoRA key for a HF causal-LM.
PEFT_MODEL_PREFIX = "base_model.model.model"
#: The attention submodule q/k/v/o live under in llama-style models
#: (deepseek-coder-1.3b-base included).
PEFT_ATTN_SEGMENT = "self_attn"

#: Contract (D8): rank 16, q/k/v/o, fp32.
TARGET_MODULES: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
WIRE_DTYPE = np.float32

_PART_RE = r"(lora_A|lora_B)"

#: Matches either convention. Layer index and module name are the only things
#: read out of the key; everything between them is ignored, which is what makes
#: this work for the short cluster form and the full PEFT form alike.
_KEY_RE = re.compile(
    rf"(?:^|\.)layers\.(?P<layer>\d+)\.(?:.*\.)?(?P<module>[A-Za-z0-9_]+)\.{_PART_RE}\.weight$"
)


class WireFormatError(ValueError):
    """An adapter state_dict does not match either supported key convention."""


# --------------------------------------------------------------------------- #
# key translation — the bidirectional map B3 asks for
# --------------------------------------------------------------------------- #
def peft_key(layer: int, module: str, part: str,
             model_prefix: str = PEFT_MODEL_PREFIX,
             attn_segment: str = PEFT_ATTN_SEGMENT) -> str:
    """``base_model.model.model.layers.<i>.self_attn.<module>.<part>.weight``."""
    return f"{model_prefix}.layers.{layer}.{attn_segment}.{module}.{part}.weight"


def cluster_key(layer: int, module: str, part: str) -> str:
    """``layers.<i>.<module>.<part>.weight`` — LoRAAdapter.to_state_dict's form."""
    return f"layers.{layer}.{module}.{part}.weight"


def parse_key(key: str) -> Tuple[int, str, str]:
    """(layer, module, part) from a key in EITHER convention.

    Raises ``WireFormatError`` rather than guessing: a key this cannot parse is
    a key the aggregator would silently drop, which is exactly the failure mode
    that makes a wrong composite instead of an error.
    """
    m = _KEY_RE.search(key)
    if m is None:
        raise WireFormatError(f"unrecognized adapter key {key!r}")
    return int(m.group("layer")), m.group("module"), m.group(3)


def translate_keys(state_dict: Dict[str, np.ndarray], to: str,
                   model_prefix: str = PEFT_MODEL_PREFIX,
                   attn_segment: str = PEFT_ATTN_SEGMENT) -> Dict[str, np.ndarray]:
    """Re-key a whole state_dict into ``to`` ∈ {"peft", "cluster"}.

    Tensors are passed through by reference; only the keys change. A collision
    after translation raises instead of overwriting.
    """
    if to not in ("peft", "cluster"):
        raise ValueError(f"to must be 'peft' or 'cluster', got {to!r}")
    out: Dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        layer, module, part = parse_key(key)
        new = (peft_key(layer, module, part, model_prefix, attn_segment)
               if to == "peft" else cluster_key(layer, module, part))
        if new in out:
            raise WireFormatError(
                f"key collision after translation: {key!r} and another key both "
                f"map to {new!r}")
        out[new] = tensor
    return out


def to_peft_keys(state_dict: Dict[str, np.ndarray], **kw) -> Dict[str, np.ndarray]:
    """Normalize any accepted convention to the PEFT one `edge.merge` reads."""
    return translate_keys(state_dict, "peft", **kw)


def to_cluster_keys(state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize any accepted convention to the short cluster one."""
    return translate_keys(state_dict, "cluster")


# --------------------------------------------------------------------------- #
# structure + dtype
# --------------------------------------------------------------------------- #
def describe(state_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Layer count, target modules, rank and dtypes read off the tensors.

    The upload envelope needs ``num_layers`` and ``target_modules``; deriving
    them from the tensors rather than from a config file means a truncated or
    mis-saved adapter is caught here instead of at the aggregator.
    """
    layers: set[int] = set()
    modules: set[str] = set()
    ranks: set[int] = set()
    dtypes: set[str] = set()
    for key, tensor in state_dict.items():
        layer, module, part = parse_key(key)
        layers.add(layer)
        modules.add(module)
        dtypes.add(str(np.asarray(tensor).dtype))
        arr = np.asarray(tensor)
        if arr.ndim != 2:
            raise WireFormatError(f"{key!r}: expected a 2-D tensor, got {arr.ndim}-D")
        ranks.add(arr.shape[0] if part == "lora_A" else arr.shape[1])
    if not layers:
        raise WireFormatError("empty adapter state_dict")
    expected = sorted(range(max(layers) + 1))
    if sorted(layers) != expected:
        missing = sorted(set(expected) - layers)
        raise WireFormatError(f"non-contiguous layer indices; missing {missing}")
    if len(ranks) != 1:
        raise WireFormatError(f"inconsistent LoRA rank across tensors: {sorted(ranks)}")
    n_expected = 2 * len(modules) * len(layers)
    if len(state_dict) != n_expected:
        raise WireFormatError(
            f"expected {n_expected} tensors (lora_A + lora_B per module, "
            f"{len(layers)} layers x {len(modules)} modules), got {len(state_dict)}")
    ordered = tuple(m for m in TARGET_MODULES if m in modules) + tuple(
        sorted(modules - set(TARGET_MODULES)))
    return {
        "num_layers": len(layers),
        "target_modules": ordered,
        "rank": ranks.pop(),
        "n_tensors": len(state_dict),
        "dtypes": sorted(dtypes),
    }


def as_fp32(state_dict: Dict[str, np.ndarray], strict: bool = False) -> Dict[str, np.ndarray]:
    """Enforce the FP32 wire contract.

    ``strict=True`` refuses to convert instead of silently casting — used where
    a dtype surprise means the producer, not the wire, is wrong.
    """
    out: Dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        arr = np.asarray(tensor)
        if arr.dtype != WIRE_DTYPE:
            if strict:
                raise WireFormatError(
                    f"{key!r}: dtype {arr.dtype} violates the fp32 wire contract")
            arr = arr.astype(WIRE_DTYPE)
        out[key] = np.ascontiguousarray(arr)
    return out


# --------------------------------------------------------------------------- #
# PEFT directory <-> numpy state_dict (torch-free: the numpy safetensors backend)
# --------------------------------------------------------------------------- #
def canonicalize_safetensors(blob: bytes) -> bytes:
    """Rewrite a safetensors blob's header with sorted, compact JSON.

    ``safetensors`` serializes the header — ``__metadata__`` included — from a
    Rust hash map, so the SAME tensors and the SAME metadata come out in a
    different key order from call to call. Measured here: three `save_file`
    calls in one process produced two distinct sha256 digests. That makes an
    adapter's bytes, and therefore the digest the registry records, unstable
    across runs even when nothing about the adapter changed — which quietly
    breaks D9's reproducibility story and makes the digest useless as a content
    fingerprint (it stays perfectly good as the transport integrity check it is
    used for in ``registry_client.fetch_verified``).

    The fix is purely in the header: ``data_offsets`` are relative to the start
    of the byte buffer that FOLLOWS the header, so re-serializing the header
    JSON canonically leaves every offset valid. Same tensors in, same bytes out.

    Mirrored in ``cluster.server`` — see the note on :func:`encode_tensor` for
    why this is duplicated rather than imported across modules.
    """
    if len(blob) < 8:
        raise WireFormatError("not a safetensors blob: shorter than the 8-byte header length")
    header_len = int.from_bytes(blob[:8], "little")
    if header_len <= 0 or 8 + header_len > len(blob):
        raise WireFormatError(f"safetensors header length {header_len} is out of range")
    header = json.loads(blob[8:8 + header_len].decode("utf-8"))
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(canonical).to_bytes(8, "little") + canonical + blob[8 + header_len:]


def load_peft_dir(path: Path | str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Read a PEFT adapter directory as numpy — same two files
    ``edge.merge.load_adapter`` reads, without requiring torch.

    Kept torch-free on purpose: the seam-A path (upload, aggregate, download,
    materialize) then runs in CI and inside the cluster container, neither of
    which has torch. ``edge.merge.load_adapter`` remains the loader for
    anything that has to become a tensor.
    """
    from safetensors.numpy import load_file

    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    sd = load_file(str(path / "adapter_model.safetensors"))
    return sd, cfg


def save_peft_dir(path: Path | str, state_dict: Dict[str, np.ndarray], cfg: Dict,
                  embed_config: bool = True) -> Path:
    """Write ``adapter_config.json`` + ``adapter_model.safetensors``.

    ``embed_config`` also stores the config in the safetensors file's own
    ``__metadata__`` header. That is what makes the payload self-describing
    once the registry has it: the registry stores tensor bytes opaquely and
    serves them back with no config beside them (sprint Defect 4), so carrying
    the config INSIDE the blob means the round trip loses nothing and the Edge
    never has to invent a field it was not given.
    """
    from safetensors.numpy import save

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metadata = {"adapter_config": json.dumps(cfg), "format": "pt"} if embed_config else None
    blob = save(as_fp32(state_dict), metadata=metadata)
    (path / "adapter_model.safetensors").write_bytes(canonicalize_safetensors(blob))
    return path


def serialize(state_dict: Dict[str, np.ndarray], cfg: Optional[Dict] = None) -> bytes:
    """The safetensors bytes the registry stores for one adapter version.

    Canonical header ordering, so the same adapter always serializes to the
    same bytes (see :func:`canonicalize_safetensors`).
    """
    from safetensors.numpy import save

    metadata = {"adapter_config": json.dumps(cfg), "format": "pt"} if cfg else None
    return canonicalize_safetensors(save(as_fp32(state_dict), metadata=metadata))


def deserialize(payload: bytes) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
    """Inverse of :func:`serialize`: (state_dict, embedded config or None)."""
    from safetensors.numpy import load

    header_len = int.from_bytes(payload[:8], "little")
    meta = (json.loads(payload[8:8 + header_len].decode("utf-8")).get("__metadata__") or {})
    sd = load(payload)
    cfg = json.loads(meta["adapter_config"]) if "adapter_config" in meta else None
    return sd, cfg


# --------------------------------------------------------------------------- #
# seam A — the upload envelope
# --------------------------------------------------------------------------- #
def encode_tensor(name: str, array: np.ndarray) -> Dict:
    """One tensor in the seam-A wire encoding: name, dtype, shape, base64 buffer.

    Mirrors ``cluster.schemas.messages.TensorPayload`` but is written out here
    rather than imported: ``contracts`` is meant to be the only cross-module
    import path (docs/architecture.md §2), and edge has to stay importable with
    no cluster package installed — which is what the CI matrix gives it. The two
    encodings are asserted equivalent in ``tests/integration``, which has both.
    """
    import base64

    array = np.ascontiguousarray(array)
    return {
        "name": name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_tensor(payload: Dict) -> np.ndarray:
    """Inverse of :func:`encode_tensor`."""
    import base64

    raw = base64.b64decode(payload["data_b64"])
    dtype = np.dtype(payload["dtype"])
    expected = int(np.prod(payload["shape"])) * dtype.itemsize
    if len(raw) != expected:
        raise WireFormatError(
            f"tensor {payload.get('name')!r}: payload is {len(raw)} bytes, "
            f"shape/dtype imply {expected}")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(payload["shape"])).copy()


def upload_payload(state_dict: Dict[str, np.ndarray], cfg: Dict, *,
                   client_id: str, cluster_id: str, round_id: int,
                   num_examples: int, seed: Optional[int] = None,
                   key_convention: str = "peft") -> Dict:
    """Build the JSON body for ``POST /uploads`` from a real client adapter.

    Carries everything the cluster needs to identify and aggregate the upload:
    the source client and its cluster, the tensor payload, the adapter's own
    hyperparameters (rank / alpha / target modules / layer count), and the
    FedAvg sample weight (``num_examples``). Matches
    ``cluster.schemas.messages.AdapterUpload``.
    """
    sd = as_fp32(translate_keys(state_dict, "peft" if key_convention == "peft" else "cluster"))
    info = describe(sd)
    rank = int(cfg.get("r", info["rank"]))
    if rank != info["rank"]:
        raise WireFormatError(
            f"adapter_config r={rank} disagrees with the tensors' rank {info['rank']}")

    # The cluster reconstructs each client's update as delta_W = B @ A, with no
    # scaling — see LoRAAdapter.delta_w. PEFT's real update is s . B . A. For the
    # six trained clients r == lora_alpha == 16 so s is exactly 1.0 and the two
    # agree, but an adapter whose scaling is not 1.0 would be silently
    # mis-weighted in the average, producing a wrong cluster adapter rather than
    # an error. Fold s into lora_B and declare alpha == rank, so delta_W is
    # preserved exactly and the cluster's unscaled product means what it should.
    scaling = scaling_of_cfg({"r": rank, "lora_alpha": cfg.get("lora_alpha", rank),
                              "use_rslora": cfg.get("use_rslora", False)})
    if scaling != 1.0:
        sd = {k: (v * WIRE_DTYPE(scaling) if parse_key(k)[2] == "lora_B" else v)
              for k, v in sd.items()}
        sd = as_fp32(sd)

    return {
        "client_id": client_id,
        "cluster_id": cluster_id,
        "round_id": round_id,
        "rank": rank,
        "target_modules": list(info["target_modules"]),
        # alpha == rank, so the receiving side's scaling is exactly 1.0 and its
        # B @ A is the client's true delta_W (see the fold above).
        "alpha": float(rank),
        "num_layers": int(info["num_layers"]),
        "num_examples": int(num_examples),
        "seed": seed,
        "tensors": [encode_tensor(name, arr) for name, arr in sd.items()],
    }


def upload_payload_from_dir(adapter_dir: Path | str, **kw) -> Dict:
    """:func:`upload_payload` straight off a trained client's adapter directory."""
    sd, cfg = load_peft_dir(adapter_dir)
    return upload_payload(sd, cfg, **kw)


def tensors_from_payload(tensors: Iterable[Dict]) -> Dict[str, np.ndarray]:
    """Decode a tensor list (upload or broadcast) back to numpy."""
    return {t["name"]: decode_tensor(t) for t in tensors}


def broadcast_to_peft(broadcast: Dict) -> Tuple[Dict[str, np.ndarray], Dict]:
    """A cluster ``/aggregate`` (or ``/active``) response -> (PEFT state_dict, config).

    Key translation happens here unconditionally, so the Edge is indifferent to
    which convention the cluster chose to emit (sprint fallback F1).
    """
    sd = to_peft_keys(tensors_from_payload(broadcast["tensors"]))
    cfg = broadcast.get("peft_config")
    if not cfg:
        raise WireFormatError(
            "cluster broadcast carries no peft_config; cannot build an "
            "adapter_config.json without inventing hyperparameters")
    cfg = dict(cfg)
    # Pin the composition scaling to 1.0 the way edge.aggregate does, so the
    # stored tensors mean exactly delta_W and no consumer has to know the
    # cluster's internal alpha convention.
    cfg["lora_alpha"] = cfg.get("r", broadcast["rank"])
    cfg["use_rslora"] = False
    return as_fp32(sd), cfg


def scaling_of_cfg(cfg: Dict) -> float:
    """PEFT's forward-time multiplier on B@A. Mirrors ``edge.merge.scaling_of``
    without importing torch."""
    import math

    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def delta_w(state_dict: Dict[str, np.ndarray], cfg: Dict, layer: int, module: str,
            ) -> np.ndarray:
    """s . B . A for one (layer, module) — the adapter's actual effect, in float64."""
    s = scaling_of_cfg(cfg)
    a = b = None
    for key, tensor in state_dict.items():
        lay, mod, part = parse_key(key)
        if lay == layer and mod == module:
            if part == "lora_A":
                a = np.asarray(tensor, dtype=np.float64)
            else:
                b = np.asarray(tensor, dtype=np.float64)
    if a is None or b is None:
        raise WireFormatError(f"layer {layer} module {module!r}: missing lora_A/lora_B")
    return s * (b @ a)


def relative_delta_error(sd_a: Dict[str, np.ndarray], cfg_a: Dict,
                         sd_b: Dict[str, np.ndarray], cfg_b: Dict,
                         modules: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Per-module relative Frobenius distance between two adapters' delta_W.

    Compares what the adapters DO, not how they are factorized: an SVD's
    factors are only defined up to sign and up to rotation within a degenerate
    singular subspace, so comparing A and B entrywise would report a difference
    where there is none.
    """
    info = describe(sd_a)
    mods = tuple(modules) if modules else info["target_modules"]
    worst, total, n, worst_at = 0.0, 0.0, 0, ""
    for layer in range(info["num_layers"]):
        for module in mods:
            ref = delta_w(sd_b, cfg_b, layer, module)
            got = delta_w(sd_a, cfg_a, layer, module)
            denom = float(np.linalg.norm(ref))
            rel = float(np.linalg.norm(got - ref) / denom) if denom else 0.0
            total += rel
            n += 1
            if rel > worst:
                worst, worst_at = rel, f"layers.{layer}.{module}"
            del ref, got
    return {"max_rel_err": worst, "mean_rel_err": total / n if n else 0.0,
            "n_modules": n, "max_at": worst_at}
