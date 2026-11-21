"""scripts/make_users.py: the users file of a page, written without a password argument."""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _answers(monkeypatch, mod, *passwords):
    asked = []
    queue = list(passwords)

    def ask(prompt):
        asked.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr(mod, "ask_password", ask)
    return asked


def test_it_writes_an_operator_the_panel_can_verify(tmp_path, monkeypatch, capsys):
    from security.auth.users import UserStore

    mod = _load("make_users")
    out = tmp_path / "users" / "panel.json"
    asked = _answers(monkeypatch, mod, "panel pass phrase", "panel pass phrase")
    assert mod.main(["--out", str(out), "--add", "operator:operator"]) == 0
    assert len(asked) == 2 and "operator" in asked[0]          # asked, and asked again to confirm
    assert UserStore(out).verify("operator", "panel pass phrase") == {"role": "operator",
                                                                      "subject": None}
    printed = capsys.readouterr()
    assert "panel pass phrase" not in printed.out + printed.err
    assert "panel pass phrase" not in out.read_text(encoding="utf-8")


def test_a_developer_names_the_client_and_users_accumulate(tmp_path, monkeypatch):
    from security.auth.users import UserStore

    mod = _load("make_users")
    out = tmp_path / "client-flask.json"
    _answers(monkeypatch, mod, "first pass phrase", "first pass phrase",
             "second pass phrase", "second pass phrase")
    assert mod.main(["--out", str(out), "--add", "asha:developer:client-flask",
                     "--add", "ravi:developer:client-flask"]) == 0
    store = UserStore(out)
    assert store.usernames() == ["asha", "ravi"]
    assert store.verify("ravi", "second pass phrase")["subject"] == "client-flask"


def test_the_password_is_never_an_argument(tmp_path, monkeypatch, capsys):
    mod = _load("make_users")
    _answers(monkeypatch, mod)
    for argv in (["--out", str(tmp_path / "u.json"), "--add", "operator:operator:hunter2222"],
                 ["--out", str(tmp_path / "u.json"), "--add", "operator:operator",
                  "--password", "hunter2222"]):
        with pytest.raises(SystemExit):
            mod.main(argv)
    assert not (tmp_path / "u.json").exists()
    help_text = mod.build_parser().format_help()
    assert "--password" not in help_text


def test_passwords_that_differ_or_are_too_short_write_nothing(tmp_path, monkeypatch, capsys):
    mod = _load("make_users")
    out = tmp_path / "u.json"
    _answers(monkeypatch, mod, "one pass phrase", "another pass phrase")
    assert mod.main(["--out", str(out), "--add", "operator:operator"]) == 1
    assert "did not match" in capsys.readouterr().err and not out.exists()
    _answers(monkeypatch, mod, "short", "short")
    assert mod.main(["--out", str(out), "--add", "operator:operator"]) == 1
    assert "at least 8" in capsys.readouterr().err and not out.exists()


def test_a_bad_spec_is_refused_before_any_prompt(tmp_path, monkeypatch):
    mod = _load("make_users")
    asked = _answers(monkeypatch, mod)
    for spec in ("operator", "operator:root", "dev:developer", ":operator"):
        with pytest.raises(SystemExit):
            mod.main(["--out", str(tmp_path / "u.json"), "--add", spec])
    assert asked == []


def test_the_edge_bundle_carries_that_edges_users_file_and_starts_with_it(tmp_path, monkeypatch):
    from security.auth.users import UserStore

    bundle = _load("make_edge_bundle")
    artifacts = tmp_path / "artifacts"
    adapter = artifacts / "round1" / "client-flask" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter.parent / "manifest.json").write_text(json.dumps({"client_id": "web/client-flask"}),
                                                  encoding="utf-8")
    users = tmp_path / "users"
    UserStore(users / "client-flask.json").add("asha", "a developer pass phrase", "developer",
                                               subject="client-flask")
    UserStore(users / "client-numpy.json").add("ravi", "another pass phrase 1", "developer",
                                               subject="client-numpy")
    UserStore(users / "panel.json").add("operator", "the panel pass phrase", "operator")
    monkeypatch.setattr(bundle, "ARTIFACTS", artifacts)
    monkeypatch.setattr(bundle, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bundle, "CORPUS", tmp_path / "no-corpus")

    out = bundle.build("client-flask", "http://10.0.0.1:8002", tmp_path / "b.zip", ["round1"],
                       None, False, users_dir=users)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        start = z.read("start_edge_client-flask.ps1").decode("utf-8")
        steps = z.read("EDGE_client-flask.md").decode("utf-8")
        carried = json.loads(z.read("edge-users/client-flask.json"))
    assert "edge-users/client-flask.json" in names
    assert not any("numpy" in n or "panel" in n for n in names)
    assert list(carried["users"]) == ["asha"]
    assert "--users edge-users/client-flask.json" in start
    assert "log in" in steps.lower()

    plain = bundle.build("client-flask", "http://10.0.0.1:8002", tmp_path / "c.zip", ["round1"],
                         None, False, users_dir=tmp_path / "no-users")
    with zipfile.ZipFile(plain) as z:
        assert not any(n.startswith("edge-users/") for n in z.namelist())
        assert "--users" not in z.read("start_edge_client-flask.ps1").decode("utf-8")
