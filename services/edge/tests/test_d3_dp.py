"""D3 composition order and D7 DP-SGD in client training.

CPU tests cover the bookkeeping: how a round reads whether its clients were
D3-trained, how supplied cluster adapters are parsed, and how the DP path fails
when P3's library is not integrated. The GPU tests check the D3 mechanism
itself on the real model — the frozen cluster layer really is frozen, and with
a fresh client adapter the stacked model is exactly base + alpha*cluster — and
skip on CPU-only CI (see test_setup.py for the house gate).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from edge import train_client
from edge.round import d3_status, parse_cluster_adapters, training_composition

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
CLUSTER_WEB = ARTIFACTS / "round2_cluster" / "cluster-web"


# --------------------------------------------------------------------------- #
# round bookkeeping
# --------------------------------------------------------------------------- #
def _client(tmp_path: Path, name: str, d3: dict | None) -> Path:
    adapter = tmp_path / name / "adapter"
    adapter.mkdir(parents=True)
    (tmp_path / name / "manifest.json").write_text(json.dumps(
        {"d3": d3, "d3_deviation": None if d3 else "trained on frozen base only"}),
        encoding="utf-8")
    return adapter


def test_training_composition_reads_the_manifest(tmp_path):
    d3 = {"alpha": 0.5, "cluster_adapter": "pulled/cluster-web/v2", "other": 1}
    got = training_composition(_client(tmp_path, "a", d3))
    assert got["d3"] == {"alpha": 0.5, "cluster_adapter": "pulled/cluster-web/v2"}
    assert training_composition(_client(tmp_path, "b", None))["d3"] is None


def test_training_composition_without_a_manifest(tmp_path):
    (tmp_path / "x" / "adapter").mkdir(parents=True)
    assert training_composition(tmp_path / "x" / "adapter")["d3"] is None


@pytest.mark.parametrize("flags,prefix", [
    ([True, True], "closed"), ([False, False], "open"), ([True, False], "partial"),
])
def test_d3_status_summarizes_the_round(flags, prefix):
    results = {f"c{i}": {"training": {"d3": {"alpha": 0.5} if f else None}}
               for i, f in enumerate(flags)}
    assert d3_status(results).startswith(prefix)


def test_parse_cluster_adapters(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert parse_cluster_adapters([f"web={tmp_path / 'web'}"]) == {"web": tmp_path / "web"}
    assert parse_cluster_adapters(None) == {}
    with pytest.raises(SystemExit, match="CLUSTER=DIR"):
        parse_cluster_adapters(["web"])
    with pytest.raises(SystemExit, match="not an adapter directory"):
        parse_cluster_adapters([f"scientific={tmp_path}"])


# --------------------------------------------------------------------------- #
# DP plumbing (CPU)
# --------------------------------------------------------------------------- #
def test_dp_blocks_must_be_equal_length():
    with pytest.raises(ValueError, match="equal-length"):
        train_client._Blocks([[1, 2, 3], [1, 2]])
    ds = train_client._Blocks([[1, 2, 3], [4, 5, 6]])
    assert len(ds) == 2 and ds[1].tolist() == [4, 5, 6]


def test_dp_without_p3s_library_fails_with_a_clear_message():
    security = pytest.importorskip("security")
    if hasattr(security, "make_private") or "make_private" in getattr(security, "__all__", ()):
        pytest.skip("P3's DP API is installed; nothing to refuse")
    with pytest.raises(RuntimeError, match="security library"):
        train_client._security_dp()


# --------------------------------------------------------------------------- #
# D3 on the real model (GPU)
# --------------------------------------------------------------------------- #
def _gpu_skip_reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU; CI runners are CPU-only"
    if not (CLUSTER_WEB / "adapter_config.json").exists():
        return "registry-pulled cluster adapter not on disk (gitignored)"
    return None


@pytest.fixture(scope="module")
def stacked():
    # Gate HERE, before the model loads: a skip in the test body would come
    # too late, after the fixture had already tried to put 1.3B on a GPU.
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    from edge.lora_init import attach_lora
    from edge.model_loader import load_model

    model, tokenizer, _ = load_model("dev")
    model = attach_lora(model, r=train_client.LORA_R, lora_alpha=train_client.LORA_ALPHA,
                        target_modules=train_client.LORA_TARGET_MODULES)
    info = train_client.stack_frozen_cluster(model, CLUSTER_WEB, alpha=0.5)
    yield model, tokenizer, info
    del model
    torch.cuda.empty_cache()


def test_only_the_client_adapter_trains(stacked):
    model, _, info = stacked
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".default." in n for n in trainable)
    assert sum(p.numel() for n, p in model.named_parameters() if p.requires_grad) == 6_291_456
    assert info["lora_layers_scaled"] == 96
    assert set(model.base_model.active_adapter) == {"default", "cluster"}


def test_fresh_client_on_frozen_cluster_is_exactly_base_plus_alpha_cluster(stacked):
    """B_client starts at zero, so the stack must equal a single pre-merged
    alpha*cluster composite (edge.merge, D6) — the identity D3 rests on."""
    from edge.merge import compose, load_adapter, save_adapter

    model, tokenizer, _ = stacked
    ids = tokenizer("def add(a, b):\n    return a + b\n", return_tensors="pt").input_ids
    ids = ids.to(model.device)
    model.eval()
    with torch.no_grad():
        got = model(ids).logits.float()
    sd, cfg = load_adapter(CLUSTER_WEB)
    ref_sd, ref_cfg = compose([(sd, cfg, 0.5)])
    ref_dir = Path(tempfile.mkdtemp())
    save_adapter(ref_dir, ref_sd, ref_cfg)
    model.load_adapter(str(ref_dir), adapter_name="ref")
    model.base_model.set_adapter("ref")
    with torch.no_grad():
        want = model(ids).logits.float()
    model.base_model.set_adapter(["default", "cluster"])
    model.delete_adapter("ref")
    assert torch.equal(got, want)


def test_saving_writes_the_client_adapter_only(stacked, tmp_path):
    model, _, _ = stacked
    model.save_pretrained(str(tmp_path), selected_adapters=[train_client.CLIENT_ADAPTER_NAME])
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert not (tmp_path / train_client.CLUSTER_ADAPTER_NAME).exists()
    from safetensors.torch import load_file

    keys = load_file(str(tmp_path / "adapter_model.safetensors")).keys()
    assert len(keys) == 192 and not any("cluster" in k for k in keys)
