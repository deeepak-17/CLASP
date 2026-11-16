"""scripts/run_services.py starts the stack behind mTLS unless told otherwise (U6).

With no flags the registry and the cluster get the three ``CLASP_TLS_*``
variables from ``.runtime/certs`` (generated when missing); ``--insecure``
starts plain HTTP. Nothing here starts a process: the environment each service
would get is what is checked, and the certificate script call is patched.
"""
from __future__ import annotations

import importlib.util
import json
import ssl
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TLS_VARS = ("CLASP_TLS_CERT", "CLASP_TLS_KEY", "CLASP_TLS_CA")


@pytest.fixture
def rs():
    spec = importlib.util.spec_from_file_location("run_services", REPO / "scripts" / "run_services.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _certs(root: Path) -> Path:
    """The layout scripts/make_dev_certs.sh writes (empty files: nothing reads them here)."""
    for rel in ("ca.pem", "registry/registry.pem", "registry/registry.key", "registry/ca.pem",
                "cluster/cluster.pem", "cluster/cluster.key", "cluster/ca.pem",
                "cluster/upload_hmac.json", "demo-ui/demo-ui.pem", "demo-ui/demo-ui.key",
                "demo-ui/ca.pem", "pages/pages.pem", "pages/pages.key",
                "secrets/eval_hmac.key", "secrets/jwt.key"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return root


# ---- flags ---------------------------------------------------------------------------
def test_with_no_flags_the_certificates_are_runtime_certs(rs):
    args = rs.build_parser().parse_args([])
    assert rs.mtls_dir_of(args) == rs.RUNTIME / "certs"
    assert rs.DEFAULT_MTLS_DIR == rs.RUNTIME / "certs"


def test_insecure_means_no_certificates_and_excludes_mtls_dir(rs, tmp_path):
    assert rs.mtls_dir_of(rs.build_parser().parse_args(["--insecure"])) is None
    explicit = rs.build_parser().parse_args(["--mtls-dir", str(tmp_path)])
    assert rs.mtls_dir_of(explicit) == tmp_path
    with pytest.raises(SystemExit):
        rs.mtls_dir_of(rs.build_parser().parse_args(["--insecure", "--mtls-dir", str(tmp_path)]))


# ---- the environment each service gets -------------------------------------------------
@pytest.mark.parametrize("name", ["registry", "cluster"])
def test_by_default_a_service_gets_the_three_tls_variables(rs, tmp_path, name):
    certs = _certs(tmp_path / "certs")
    env = rs._env(name, "0.0.0.0", certs, tmp_path / "data")
    assert env["CLASP_TLS_CERT"] == str(certs / name / f"{name}.pem")
    assert env["CLASP_TLS_KEY"] == str(certs / name / f"{name}.key")
    assert env["CLASP_TLS_CA"] == str(certs / name / "ca.pem")
    assert env["CLASP_REGISTRY_URL"].startswith("https://")
    assert env["CLASP_CLUSTER_URL"].startswith("https://")


@pytest.mark.parametrize("name", ["registry", "cluster", "demo-ui"])
def test_insecure_yields_none_of_them_even_when_the_shell_has_them(rs, tmp_path, monkeypatch, name):
    for var in (*TLS_VARS, "CLASP_TLS_CLIENT_CERT", "CLASP_TLS_CLIENT_KEY",
                "CLASP_PAGE_TLS_CERT", "CLASP_PAGE_TLS_KEY", "CLASP_EVAL_HMAC_SECRET_FILE",
                "CLASP_UPLOAD_HMAC_SECRETS"):
        monkeypatch.setenv(var, "/left/over/from/the/shell")
    env = rs._env(name, "127.0.0.1", None, tmp_path / "data")
    for var in (*TLS_VARS, "CLASP_TLS_CLIENT_CERT", "CLASP_TLS_CLIENT_KEY",
                "CLASP_PAGE_TLS_CERT", "CLASP_PAGE_TLS_KEY", "CLASP_EVAL_HMAC_SECRET_FILE",
                "CLASP_UPLOAD_HMAC_SECRETS"):
        assert var not in env, var
    assert env["CLASP_REGISTRY_URL"].startswith("http://")


def test_the_panel_presents_its_client_certificate_and_serves_https(rs, tmp_path):
    certs = _certs(tmp_path / "certs")
    env = rs._env("demo-ui", "0.0.0.0", certs, tmp_path / "data")
    assert env["CLASP_TLS_CLIENT_CERT"] == str(certs / "demo-ui" / "demo-ui.pem")
    assert env["CLASP_TLS_CA"] == str(certs / "demo-ui" / "ca.pem")
    assert "CLASP_TLS_CERT" not in env                     # it is a client of the services
    assert env["CLASP_PAGE_TLS_CERT"] == str(certs / "pages" / "pages.pem")
    assert env["CLASP_PAGE_TLS_KEY"] == str(certs / "pages" / "pages.key")
    assert env["CLASP_JWT_SECRET_FILE"] == str(certs / "secrets" / "jwt.key")


def test_the_secrets_go_to_the_services_that_use_them(rs, tmp_path):
    certs = _certs(tmp_path / "certs")
    registry = rs._env("registry", "0.0.0.0", certs, tmp_path / "data")
    cluster = rs._env("cluster", "0.0.0.0", certs, tmp_path / "data")
    panel = rs._env("demo-ui", "0.0.0.0", certs, tmp_path / "data")
    key = str(certs / "secrets" / "eval_hmac.key")
    assert registry["CLASP_EVAL_HMAC_SECRET_FILE"] == key      # verifies
    assert panel["CLASP_EVAL_HMAC_SECRET_FILE"] == key         # its evaluator signs
    assert "CLASP_EVAL_HMAC_SECRET_FILE" not in cluster
    assert cluster["CLASP_UPLOAD_HMAC_SECRETS"] == str(certs / "cluster" / "upload_hmac.json")
    assert "CLASP_UPLOAD_HMAC_SECRETS" not in registry and "CLASP_UPLOAD_HMAC_SECRETS" not in panel
    assert "CLASP_JWT_SECRET_FILE" not in registry and "CLASP_JWT_SECRET_FILE" not in cluster


def test_a_secret_file_that_is_not_there_is_not_configured(rs, tmp_path):
    certs = _certs(tmp_path / "certs")
    (certs / "secrets" / "eval_hmac.key").unlink()
    (certs / "cluster" / "upload_hmac.json").unlink()
    assert "CLASP_EVAL_HMAC_SECRET_FILE" not in rs._env("registry", "0.0.0.0", certs, tmp_path)
    assert "CLASP_UPLOAD_HMAC_SECRETS" not in rs._env("cluster", "0.0.0.0", certs, tmp_path)


def test_each_service_has_its_own_audit_file(rs, tmp_path):
    certs = _certs(tmp_path / "certs")
    files = {n: rs._env(n, "0.0.0.0", certs, tmp_path)["CLASP_AUDIT_LOG"]
             for n in ("registry", "cluster", "demo-ui")}
    assert len(set(files.values())) == 3
    assert all(Path(f).parent == rs.RUNTIME / "audit" for f in files.values())


def test_the_panels_users_file_is_passed_when_it_exists(rs, tmp_path, monkeypatch):
    certs = _certs(tmp_path / "certs")
    monkeypatch.setattr(rs, "USERS_FILE", tmp_path / "users" / "panel.json")
    assert "CLASP_USERS_FILE" not in rs._env("demo-ui", "0.0.0.0", certs, tmp_path)
    rs.USERS_FILE.parent.mkdir()
    rs.USERS_FILE.write_text("{}", encoding="utf-8")
    assert rs._env("demo-ui", "0.0.0.0", certs, tmp_path)["CLASP_USERS_FILE"] == str(rs.USERS_FILE)
    assert "CLASP_USERS_FILE" not in rs._env("registry", "0.0.0.0", certs, tmp_path)


def test_the_panel_is_started_through_its_own_launcher(rs):
    command = rs._commands("0.0.0.0")["demo-ui"]
    assert command[1:] == ["-m", "demo_ui.serve"]


# ---- generating the certificates ---------------------------------------------------------
def test_missing_certificates_are_generated_once_with_this_machines_addresses(rs, tmp_path,
                                                                             monkeypatch):
    calls = []

    def fake(out_dir, lan_ips):
        calls.append((out_dir, lan_ips))
        _certs(out_dir)

    monkeypatch.setattr(rs, "generate_certs", fake)
    monkeypatch.setattr(rs, "lan_addresses", lambda: ["192.168.43.10", "10.77.0.1"])
    certs = tmp_path / "certs"
    assert rs.ensure_certs(certs) is True
    assert calls == [(certs, "192.168.43.10,10.77.0.1")]
    assert rs.ensure_certs(certs) is False      # already there: left alone
    assert len(calls) == 1


def test_an_explicit_lan_ips_setting_wins(rs, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "generate_certs", lambda d, ips: (calls.append(ips), _certs(d)))
    monkeypatch.setattr(rs, "lan_addresses", lambda: ["192.168.43.10"])
    monkeypatch.setenv("CLASP_LAN_IPS", "10.77.0.1")
    rs.ensure_certs(tmp_path / "certs")
    assert calls == ["10.77.0.1"]


def test_when_the_certificates_cannot_be_made_the_message_says_what_to_do(rs, tmp_path,
                                                                         monkeypatch):
    def broken(_dir, _ips):
        raise OSError("bash not found")

    monkeypatch.setattr(rs, "generate_certs", broken)
    with pytest.raises(SystemExit) as stop:
        rs.ensure_certs(tmp_path / "certs")
    text = str(stop.value)
    assert "make_dev_certs.sh" in text and "--insecure" in text


def test_the_real_script_makes_what_the_launcher_expects(rs, tmp_path):
    """Runs scripts/make_dev_certs.sh for real (needs bash and openssl)."""
    import shutil

    if not (shutil.which("openssl") and shutil.which("bash")):
        pytest.skip("needs bash and openssl")
    certs = tmp_path / "certs"
    rs.generate_certs(certs, "192.168.43.10")
    for name in ("registry", "cluster", "demo-ui"):
        env = rs._env(name, "0.0.0.0", certs, tmp_path / "data")
        for var, value in env.items():
            if var.startswith(("CLASP_TLS", "CLASP_PAGE_TLS")) or var.endswith(
                    ("_SECRET_FILE", "HMAC_SECRETS")):
                assert Path(value).is_file(), (name, var, value)
    table = json.loads((certs / "cluster" / "upload_hmac.json").read_text(encoding="utf-8"))
    assert "client-flask" in table
    # the pages certificate names this machine's LAN address, for the browsers on the hotspot
    import subprocess

    text = subprocess.run(["openssl", "x509", "-in", str(certs / "pages" / "pages.pem"), "-noout",
                           "-text"], capture_output=True, text=True, check=True).stdout
    assert "192.168.43.10" in text and "localhost" in text
    assert "TLS Web Server Authentication" in text


# ---- the pages over HTTPS -------------------------------------------------------------------
def test_the_panel_launcher_serves_https_when_the_page_certificate_is_set(tmp_path, monkeypatch):
    from demo_ui import serve

    plain = serve.build_config({})
    assert plain.ssl is None and plain.port == 8010
    with pytest.raises(serve.PageTLSError, match="CLASP_PAGE_TLS_KEY"):
        serve.build_config({"CLASP_PAGE_TLS_CERT": "x.pem"})
    with pytest.raises(serve.PageTLSError, match="not found"):
        serve.build_config({"CLASP_PAGE_TLS_CERT": str(tmp_path / "no.pem"),
                            "CLASP_PAGE_TLS_KEY": str(tmp_path / "no.key")})


def test_the_panel_answers_over_https_with_the_pages_certificate(tmp_path):
    import socket
    import threading
    import time

    import uvicorn
    from security.mtls.ca import create_ca
    from security.mtls.certs import generate_service_cert

    from demo_ui import serve
    from cryptography.hazmat.primitives import serialization

    ca_key, ca_cert = create_ca(common_name="Test CA", key_size=2048)
    key, cert = generate_service_cert(ca_key, ca_cert, "pages", sans=["localhost", "127.0.0.1"])
    (tmp_path / "ca.pem").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "pages.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "pages.key").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = serve.build_config({
        "CLASP_PAGE_TLS_CERT": str(tmp_path / "pages.pem"),
        "CLASP_PAGE_TLS_KEY": str(tmp_path / "pages.key"),
        "CLASP_DEMO_UI_HOST": "127.0.0.1", "CLASP_DEMO_UI_PORT": str(port)})
    assert config.ssl is not None
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        ctx = ssl.create_default_context(cafile=str(tmp_path / "ca.pem"))   # no client certificate
        with urllib.request.urlopen(f"https://localhost:{port}/api/health", timeout=10,
                                    context=ctx) as r:
            assert r.status == 200 and "login_required" in json.loads(r.read())
    finally:
        server.should_exit = True
        thread.join(timeout=10)
