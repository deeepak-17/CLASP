"""Regression tests for the code- and security-review findings."""
from __future__ import annotations

import json

import pytest
from contracts import AdapterKind, AdapterRef, LoRAHyperParams, PromotionAction, PromotionDecision
from registry.composite import CompositeError, PartSpec, build_composite
from registry.retention import plan_retention
from registry.runs import run_id_for, validate_config
from safetensors.numpy import load, save

from .conftest import make_safetensors
from .test_composite import MODULES, R, _hp, make_adapter
from .test_promote_api import _promote_body


# -- composite: rank comes from the tensors, not just hparams ----------------- #
def test_composite_rejects_hparams_rank_that_disagrees_with_tensors():
    eight = make_adapter(1, rank=8, lora_alpha=8, embed_cfg=False)
    with pytest.raises(CompositeError, match="rank"):
        build_composite(PartSpec(eight, _hp(rank=16, lora_alpha=8), 1.0),
                        PartSpec(make_adapter(2), _hp(), 1.0))


@pytest.mark.parametrize("rank", [0, -1])
def test_composite_rejects_nonpositive_rank(rank):
    with pytest.raises(CompositeError, match="rank"):
        build_composite(PartSpec(make_adapter(1, embed_cfg=False), _hp(rank=rank), 1.0),
                        PartSpec(make_adapter(2), _hp(), 1.0))


@pytest.mark.parametrize("field", ["use_dora", "fan_in_fan_out", "lora_bias"])
def test_composite_rejects_structural_mismatch(field):
    sd = load(make_adapter(2))
    cfg = {"peft_type": "LORA", "r": R, "lora_alpha": R, "target_modules": list(MODULES),
           field: True}
    odd = save(sd, metadata={"adapter_config": json.dumps(cfg)})
    with pytest.raises(CompositeError, match=field):
        build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(odd, _hp(), 1.0))


def test_composite_rejects_non_object_embedded_config():
    odd = save(load(make_adapter(2)), metadata={"adapter_config": "[1, 2]"})
    with pytest.raises(CompositeError, match="adapter_config"):
        build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(odd, _hp(), 1.0))


def test_composite_config_template_is_allow_listed():
    sd = load(make_adapter(2))
    cfg = {"peft_type": "LORA", "r": R, "lora_alpha": R, "target_modules": list(MODULES),
           "auto_mapping": {"evil": "x"}, "base_model_name_or_path": "base"}
    client = save(sd, metadata={"adapter_config": json.dumps(cfg)})
    out = build_composite(PartSpec(make_adapter(1), _hp(), 1.0), PartSpec(client, _hp(), 1.0))
    header_len = int.from_bytes(out.payload[:8], "little")
    got = json.loads(json.loads(out.payload[8:8 + header_len])["__metadata__"]["adapter_config"])
    assert "auto_mapping" not in got and got["base_model_name_or_path"] == "base"


def test_cluster_only_module_uses_the_client_key_convention():
    """beta=0 keeps only the cluster; modules the client lacks still get PEFT keys."""
    def short(layer, m, part):
        return f"layers.{layer}.{m}.{part}.weight"
    cluster = make_adapter(1, keyer=short, modules=("q_proj", "v_proj"))
    client = make_adapter(2, modules=("q_proj",))
    out = build_composite(PartSpec(cluster, _hp(), 1.0),
                          PartSpec(client, LoRAHyperParams(rank=R, lora_alpha=R,
                                                           target_modules=("q_proj",)), 0.0))
    assert all(k.startswith("base_model.model.model.layers.") for k in load(out.payload))


# -- retention: forward restore protects the version that went live ----------- #
def test_forward_restore_protects_its_target():
    restore = PromotionDecision(AdapterRef("f", 1, AdapterKind.CLIENT),
                                PromotionAction.PROMOTE, 3, "operator restore")
    plan = plan_retention(range(1, 8), active=7, decisions=(restore,), referenced=set(),
                          keep_last=1)
    assert 3 in plan.keep and 1 in plan.delete


# -- runs: resume survives extending a sweep; names are path-safe ------------- #
BASE = {"name": "x", "seed": 0, "gpu_hours_estimate": 0.1, "sweep": {"rank": [4, 8]}}


def test_run_id_is_stable_when_the_sweep_grows():
    grown = {**BASE, "sweep": {"rank": [4, 8, 16]}, "seeds": [0]}
    assert run_id_for(BASE, {"rank": 4}) == run_id_for(grown, {"rank": 4})
    assert run_id_for(BASE, {"rank": 4}) != run_id_for({**BASE, "lr": 1e-4}, {"rank": 4})


@pytest.mark.parametrize("name", ["../x", "a/b", "", ".hidden", "x\n"])
def test_experiment_name_must_be_path_safe(name):
    assert any("name" in p for p in validate_config({**BASE, "name": name}))


def test_non_list_sweep_value_is_a_problem_not_a_crash():
    problems = validate_config({**BASE, "sweep": {"rank": 4}})
    assert any("sweep" in p for p in problems)


def test_retry_clears_stale_outputs(tmp_path):
    import shlex
    import sys

    import yaml
    from registry.runs import ResultStore, run_sweep
    py = shlex.quote(sys.executable)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({**BASE, "sweep": {"rank": [4]}}))
    store = ResultStore(tmp_path / "s")
    dirty = (f"{py} -c \"import pathlib,sys; d=pathlib.Path(sys.argv[1]); d.mkdir(parents=True,"
             "exist_ok=True); (d/'stale.txt').write_text('x'); sys.exit(1)\" {outputs_dir}")
    run_sweep(cfg, dirty, store)
    run_sweep(cfg, f"{py} -c \"pass\"", store)
    assert store.list_runs("x")[0]["outputs"] == []


# -- API validation ---------------------------------------------------------- #
def _upload(client, name, blob=None, meta=None):
    return client.post(
        f"/adapters/{name}/versions",
        files={"file": ("a.safetensors", blob or make_safetensors(), "application/octet-stream")},
        data={"meta": json.dumps(meta or {"kind": "client"})})


def test_promote_with_null_in_project_is_422(client):
    _upload(client, "flask")
    _upload(client, "flask")
    body = _promote_body("flask", 2, good=True)
    body["eval"]["in_project"] = None
    assert client.post("/adapters/flask/promote", json=body).status_code == 422


def test_name_with_trailing_newline_is_rejected(client):
    assert _upload(client, "flask%0A").status_code == 422
    assert client.get("/adapters/flask%0A/versions").status_code == 422


def test_set_active_must_be_boolean(client):
    assert _upload(client, "flask", meta={"kind": "client", "set_active": "false"}).status_code == 422


def test_upload_over_the_size_cap_is_413(client, monkeypatch):
    monkeypatch.setenv("CLASP_MAX_UPLOAD_BYTES", "64")
    big = make_safetensors({"w": [0.0] * 64})
    assert _upload(client, "flask", blob=big).status_code == 413


def test_payload_with_bad_tensor_offsets_is_rejected(client):
    blob = bytearray(make_safetensors())
    good = bytes(blob)
    header_len = int.from_bytes(good[:8], "little")
    header = json.loads(good[8:8 + header_len])
    header["lora_A"]["data_offsets"] = [0, 9999]
    raw = json.dumps(header).encode()
    bad = len(raw).to_bytes(8, "little") + raw + good[8 + header_len:]
    assert _upload(client, "flask", blob=bad).status_code == 422


def test_oversized_header_is_rejected():
    from registry.storage import is_safetensors
    huge = (9_000_000).to_bytes(8, "little") + b"{" + b" " * 9_000_000
    assert not is_safetensors(huge)
