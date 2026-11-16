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


class _TinyLM(torch.nn.Module):
    """Just enough of a causal LM for ``train_dp``: config, base_model,
    device and a forward that takes input_ids/attention_mask/labels."""

    VOCAB = 32

    def __init__(self):
        super().__init__()
        from types import SimpleNamespace

        self.emb = torch.nn.Embedding(self.VOCAB, 8)
        self.head = torch.nn.Linear(8, self.VOCAB)
        self.config = SimpleNamespace(use_cache=True)
        self.base_model = SimpleNamespace(
            model=SimpleNamespace(gradient_checkpointing_disable=lambda: None))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, labels=None):
        from types import SimpleNamespace

        logits = self.head(self.emb(input_ids))
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.VOCAB), labels.view(-1))
        return SimpleNamespace(loss=loss)


def test_train_dp_runs_on_p3s_api_and_calibrates_to_the_budget():
    """The edge's exact DP calls against the installed security library.

    Skips only when the DP API is absent; when it is present any signature
    drift (DPConfig fields, make_private(epochs=), PrivacyAccountant,
    get_privacy_engine) fails here.
    """
    pytest.importorskip("opacus")
    security = pytest.importorskip("security")
    if not all(hasattr(security, n) for n in ("DPConfig", "make_private", "PrivacyAccountant",
                                              "get_privacy_engine")):
        pytest.skip("installed security package has no DP API yet (P3 branch not integrated)")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    chunks = torch.randint(0, _TinyLM.VOCAB, (40, 6), generator=gen).tolist()
    model = _TinyLM()
    result = train_client.train_dp(model, chunks, lr=1e-2, warmup_ratio=0.1, log_every=1,
                                   epochs=2, batch_size=8, target_epsilon=8.0,
                                   delta=1e-5, max_grad_norm=1.0)
    dp, privacy = result["dp"], result["privacy"]
    assert result["optimizer_steps"] > 0
    # sigma was calibrated to the budget, not taken from DPConfig's default.
    assert privacy["noise_multiplier"] != security.DPConfig().noise_multiplier
    assert 0 < privacy["epsilon"] <= 8.0
    assert dp["within_d7_budget"] is True
    assert privacy["epsilon"] == dp["epsilon_engine"]
    assert dp["epsilon_rdp_security"] > 0


def test_train_dp_honours_max_steps():
    """--max-steps under --dp is a hard cap, not rounded up to a full epoch."""
    pytest.importorskip("opacus")
    security = pytest.importorskip("security")
    if not all(hasattr(security, n) for n in ("DPConfig", "make_private", "PrivacyAccountant",
                                              "get_privacy_engine")):
        pytest.skip("installed security package has no DP API yet (P3 branch not integrated)")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    chunks = torch.randint(0, _TinyLM.VOCAB, (40, 6), generator=gen).tolist()
    result = train_client.train_dp(_TinyLM(), chunks, lr=1e-2, warmup_ratio=0.1, log_every=1,
                                   epochs=2, batch_size=8, target_epsilon=8.0,
                                   delta=1e-5, max_grad_norm=1.0, max_steps=1)
    assert result["planned_steps"] == 1
    assert result["optimizer_steps"] == 1
    assert result["dp"]["max_logical_steps"] == 1


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
    # the config is tracked; the weights are gitignored, and without them PEFT
    # takes the directory path for a Hub repo id
    if not (CLUSTER_WEB / "adapter_model.safetensors").exists():
        return "registry-pulled cluster adapter weights not on disk (gitignored)"
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


# --------------------------------------------------------------------------- #
# D7: one budget for the whole run of two rounds (U3, C9)
# --------------------------------------------------------------------------- #
def _tiny_dp_run(**kw):
    # CI's edge job installs neither sibling package: skip there, as the tests above do
    pytest.importorskip("opacus")
    pytest.importorskip("security.dp.ledger")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    chunks = torch.randint(0, _TinyLM.VOCAB, (40, 6), generator=gen).tolist()
    return train_client.train_dp(_TinyLM(), chunks, lr=1e-2, warmup_ratio=0.1, log_every=1,
                                 epochs=2, batch_size=8, target_epsilon=8.0,
                                 delta=1e-5, max_grad_norm=1.0, **kw)


def test_train_dp_asks_for_the_two_round_sigma(monkeypatch):
    pytest.importorskip("opacus")
    ledger = pytest.importorskip("security.dp.ledger")

    calls = []
    real = ledger.calibrate_run_sigma

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(ledger, "calibrate_run_sigma", spy)
    result = _tiny_dp_run()
    # 40 blocks in logical batches of 8: 5 steps an epoch, 2 epochs
    assert calls == [{"steps_per_round": 10, "sample_rate": pytest.approx(0.2),
                      "rounds_total": 2, "budget": 8.0, "delta": 1e-5}]
    privacy, dp = result["privacy"], result["dp"]
    assert privacy["noise_multiplier"] == pytest.approx(real(**calls[0]), abs=1e-6)
    assert privacy["rounds_total"] == 2 and dp["rounds_total"] == 2
    # one round spends its share, and two such rounds stay within the run's 8
    assert 0 < privacy["epsilon"] <= dp["round_epsilon_budget"] + 1e-3 < 8.0
    assert dp["within_d7_budget"] is True
    two = ledger._compose([(privacy["noise_multiplier"], privacy["sample_rate"],
                            2 * privacy["steps"])], 1e-5, "prv")
    assert two <= 8.0 + 1e-3


def test_train_dp_for_a_one_round_run_uses_less_noise():
    pytest.importorskip("opacus")
    two = _tiny_dp_run()["privacy"]["noise_multiplier"]
    one = _tiny_dp_run(rounds_total=1)["privacy"]["noise_multiplier"]
    assert one < two


def test_manifest_records_sample_rate_steps_and_run_id(tmp_path):
    pytest.importorskip("opacus")
    from edge import wire

    result = _tiny_dp_run()
    recorded = train_client.manifest_privacy(result["privacy"], run_id="check-1", delta=1e-5)
    assert recorded["run_id"] == "check-1" and recorded["rounds_total"] == 2
    assert recorded["sample_rate"] == pytest.approx(0.2)
    assert recorded["steps"] == result["optimizer_steps"] > 0
    # the manifest's block is still readable as the upload's privacy block
    adapter = tmp_path / "client-x" / "adapter"
    adapter.mkdir(parents=True)
    (tmp_path / "client-x" / "manifest.json").write_text(
        json.dumps({"privacy": recorded}), encoding="utf-8")
    sent = wire.privacy_from_training_manifest(adapter)
    assert sent["epsilon"] == recorded["epsilon"] and sent["steps"] == recorded["steps"]
    assert "run_id" not in sent

    off = train_client.manifest_privacy(None, run_id="check-1", delta=1e-5)
    assert off == {"epsilon": None, "delta": 1e-5, "noise_multiplier": None,
                   "max_grad_norm": None}


def test_run_flags_default_to_run_1_of_two_rounds():
    args = train_client.build_parser().parse_args(["--client", "web/client-flask"])
    assert args.run_id == "run-1" and args.rounds_total == 2
    args = train_client.build_parser().parse_args(
        ["--client", "web/client-flask", "--run-id", "check-1", "--rounds-total", "1"])
    assert args.run_id == "check-1" and args.rounds_total == 1
