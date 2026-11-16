"""The edge laptop's setup steps (scripts/make_edge_bundle.py) say what actually works."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_edge_bundle.py"


def _bundle():
    spec = importlib.util.spec_from_file_location("make_edge_bundle", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_fine_tune_steps_install_dp_and_train_like_the_page():
    steps = _bundle().run_steps("client-numpy", "scientific", "http://10.0.0.2:8002",
                                ["round1"], certs=False, corpus=True)
    assert "-e security" in steps.split("## Optional")[1]  # --dp needs P3's library
    train = next(line for line in steps.splitlines() if "edge.train_client" in line)
    assert "--dp" in train and "--seq-len 512" in train
    assert "artifacts/live-" in train  # a set the page lists as trained here

def test_the_bundle_carries_this_edges_upload_key_and_no_other(tmp_path, monkeypatch):
    """U4: each edge signs with its own key; a bundle must not leak another's."""
    import zipfile

    mod = _bundle()
    artifacts = tmp_path / "artifacts"
    for client in ("client-flask", "client-numpy"):
        adapter = artifacts / "round1" / client / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (adapter.parent / "manifest.json").write_text(
            json.dumps({"client_id": f"web/{client}"}), encoding="utf-8")
        edge = tmp_path / "certs" / "edges" / client
        edge.mkdir(parents=True)
        for name in ("ca.pem", "client.pem", "client.key"):
            (edge / name).write_text(f"{name} of {client}", encoding="utf-8")
        (edge / "hmac.key").write_text(f"key-of-{client}", encoding="utf-8")
    cluster_dir = tmp_path / "certs" / "cluster"
    cluster_dir.mkdir()
    (cluster_dir / "upload_hmac.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod, "ARTIFACTS", artifacts)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "CORPUS", tmp_path / "no-corpus")

    out = mod.build("client-flask", "https://10.0.0.1:8002", tmp_path / "edge-client-flask.zip",
                    ["round1"], tmp_path / "certs", False)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "edge-certs/client-flask/hmac.key" in names
        assert z.read("edge-certs/client-flask/hmac.key") == b"key-of-client-flask"
        everything = b"".join(z.read(n) for n in names)
        steps = z.read("EDGE_client-flask.md").decode("utf-8")
    assert not any("client-numpy" in n or "upload_hmac" in n for n in names)
    assert b"key-of-client-numpy" not in everything
    assert "hmac.key" in steps and "signed" in steps


def test_a_bundle_without_an_upload_key_still_builds(tmp_path, monkeypatch):
    import zipfile

    mod = _bundle()
    artifacts = tmp_path / "artifacts"
    adapter = artifacts / "round1" / "client-flask" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter.parent / "manifest.json").write_text(
        json.dumps({"client_id": "web/client-flask"}), encoding="utf-8")
    edge = tmp_path / "certs" / "edges" / "client-flask"
    edge.mkdir(parents=True)
    for name in ("ca.pem", "client.pem", "client.key"):
        (edge / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(mod, "ARTIFACTS", artifacts)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "CORPUS", tmp_path / "no-corpus")
    out = mod.build("client-flask", "https://10.0.0.1:8002", tmp_path / "b.zip", ["round1"],
                    tmp_path / "certs", False)
    with zipfile.ZipFile(out) as z:
        assert "edge-certs/client-flask/hmac.key" not in z.namelist()
