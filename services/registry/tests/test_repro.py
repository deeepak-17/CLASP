"""Reproducibility pack: build it, verify it, and catch tampering or gaps."""
from __future__ import annotations

import io
import json
import shlex
import sys
import tarfile

import yaml
from registry.repro import build_pack, main, verify_pack
from registry.runs import ResultStore, run_sweep

from .conftest import make_safetensors

PY = shlex.quote(sys.executable)
WRITER = (f"{PY} -c \"import pathlib,sys; d=pathlib.Path(sys.argv[1]); "
          "d.mkdir(parents=True, exist_ok=True); (d/'m.txt').write_text('ok')\" {outputs_dir}")
CFG = {"name": "rank-sweep", "seed": 0, "gpu_hours_estimate": 0.1,
       "sweep": {"rank": [4, 8]}, "budget": {"gpu_hours_total": 1}}


def _repo(tmp_path):
    exp = tmp_path / "repo" / "experiments" / "rank-sweep"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text(yaml.safe_dump(CFG))
    (exp / "RESULTS.md").write_text("# results\n")
    store = ResultStore(tmp_path / "store")
    run_sweep(exp / "config.yaml", WRITER, store)
    return tmp_path / "repo", store


def _members(pack):
    with tarfile.open(pack) as tar:
        return {m.name for m in tar.getmembers()}


def test_pack_contains_configs_runs_env_and_manifest(tmp_path):
    repo, store = _repo(tmp_path)
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "pack.tar.gz")
    names = _members(pack)
    assert "MANIFEST.json" in names and "REPRODUCE.md" in names
    assert "experiments/rank-sweep/config.yaml" in names
    assert "experiments/rank-sweep/RESULTS.md" in names
    assert "environment/packages.json" in names
    assert sum(n.endswith("/run.json") for n in names) == 2
    assert not any("/outputs/" in n for n in names)  # outputs listed by hash, not copied


def test_pack_with_outputs_copies_them(tmp_path):
    repo, store = _repo(tmp_path)
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "p.tar.gz", with_outputs=True)
    assert sum(n.endswith("/outputs/m.txt") for n in _members(pack)) == 2


def test_verify_passes_on_an_untouched_pack(tmp_path):
    repo, store = _repo(tmp_path)
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "pack.tar.gz")
    report = verify_pack(pack)
    assert report.ok, report.problems
    assert report.runs == 2 and report.configs == 1


def _rewrite(pack, out, name, data: bytes):
    with tarfile.open(pack) as src, tarfile.open(out, "w:gz") as dst:
        for m in src.getmembers():
            body = src.extractfile(m).read() if m.isfile() else None
            if m.name == name:
                body, m.size = data, len(data)
            dst.addfile(m, io.BytesIO(body) if body is not None else None)
    return out


def test_verify_catches_a_tampered_file(tmp_path):
    repo, store = _repo(tmp_path)
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "pack.tar.gz")
    bad = _rewrite(pack, tmp_path / "bad.tar.gz", "experiments/rank-sweep/config.yaml",
                   b"name: rank-sweep\nseed: 1\ngpu_hours_estimate: 0.1\n")
    report = verify_pack(bad)
    assert not report.ok
    assert any("sha256" in p and "config.yaml" in p for p in report.problems)


def test_verify_catches_an_incomplete_run(tmp_path):
    repo, store = _repo(tmp_path)
    run = store.list_runs("rank-sweep")[0]
    run_path = store.run_dir("rank-sweep", run["run_id"]) / "run.json"
    broken = {**run, "status": "failed", "manifest": {**run["manifest"], "gpu_hours_actual": None}}
    run_path.write_text(json.dumps(broken))
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "pack.tar.gz")
    report = verify_pack(pack)
    assert not report.ok
    assert any("not completed" in p for p in report.problems)
    assert any("gpu_hours_actual" in p for p in report.problems)


def test_pack_snapshots_a_live_registry(tmp_path, running):
    port = running({})
    from registry.demo import Registry

    reg = Registry(f"http://127.0.0.1:{port}")
    assert reg.save("flask", make_safetensors(), {"kind": "client"})[0] == 201
    repo, store = _repo(tmp_path)
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "pack.tar.gz",
                      registry_url=f"http://127.0.0.1:{port}")
    names = _members(pack)
    assert "registry/flask/lineage.json" in names
    assert "registry/flask/promotions.json" in names
    assert verify_pack(pack).ok


def test_cli_pack_and_verify(tmp_path, capsys):
    repo, store = _repo(tmp_path)
    out = tmp_path / "cli.tar.gz"
    assert main(["pack", "--repo", str(repo), "--store", str(store.root), "--out", str(out)]) == 0
    assert main(["verify", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["verify", str(tmp_path / "missing.tar.gz")]) == 2


def test_verify_reports_a_corrupt_manifest_cleanly(tmp_path):
    repo, store = _repo(tmp_path)
    pack = build_pack(repo_root=repo, store=store, out=tmp_path / "pack.tar.gz")
    bad = _rewrite(pack, tmp_path / "bad.tar.gz", "MANIFEST.json", b"{not json")
    report = verify_pack(bad)
    assert not report.ok and "not valid JSON" in report.problems[0]
