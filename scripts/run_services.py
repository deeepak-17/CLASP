"""Start the registry and cluster services without docker (Integration Sprint · B7).

`docker compose up -d registry cluster` is the documented path and the one the
panel demo uses. This is the same two services on the host, for when the
docker daemon is not running — which is exactly the situation the Windows
demo machine was in while this sprint was built.

    python scripts/run_services.py                # mTLS, foreground, ctrl-c to stop
    python scripts/run_services.py --stop         # stop a background set
    python scripts/run_services.py --background   # detach and return
    python scripts/run_services.py --ui           # + demo_ui on :8010 (laptop A), over HTTPS
    python scripts/run_services.py --mtls-dir DIR # certificates from somewhere else
    python scripts/run_services.py --insecure     # plain HTTP, for debugging only

mTLS is the default (U6): the certificates come from ``.runtime/certs``
(``--mtls-dir``), and are generated there with ``scripts/make_dev_certs.sh``
when they are missing, with this machine's LAN addresses in them. Alongside the
certificates each service gets what else that script wrote: the registry and
the panel's evaluator the evaluation-signing key, the cluster the edges' upload
keys, the panel its login-token key and the pages' HTTPS certificate. The panel
asks for a login when ``.runtime/users/panel.json`` exists
(``scripts/make_users.py``). ``--insecure`` gives none of this: plain HTTP, no
signatures checked, and each service prints its own warning.

Multi-laptop demo (docs/multi-laptop-demo.md): services bind 0.0.0.0 by default
so edge laptops on the same network can reach laptop A; ``--host 127.0.0.1``
keeps them on this machine only. The addresses to give the edges are printed
on start.

Each service runs through its own launcher (``registry.serve``,
``cluster.serve``, ``demo_ui.serve``), so the registry and the cluster require
client certificates and TLS 1.3, exactly as under docker compose. A service
started by hand without the ``CLASP_TLS_*`` variables still runs plain HTTP,
with a warning.

Registry state goes to ``.runtime/registry-data`` under the repo (gitignored),
which is the host-side equivalent of compose's ``registry-data`` volume: the
same D9 requirement that versioned safetensors survive a restart.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / ".runtime"
PORTS = {"registry": 8004, "cluster": 8002, "demo-ui": 8010}
ALL = tuple(PORTS)
#: Where scripts/make_dev_certs.sh writes, and where the services read from.
DEFAULT_MTLS_DIR = RUNTIME / "certs"
#: The panel's users (scripts/make_users.py); with it the panel asks for a login.
USERS_FILE = RUNTIME / "users" / "panel.json"
#: Everything a service must not inherit from the shell: it is set per service below.
MANAGED_VARS = (
    "CLASP_TLS_CERT", "CLASP_TLS_KEY", "CLASP_TLS_CA", "CLASP_TLS_CLIENT_CERT",
    "CLASP_TLS_CLIENT_KEY", "CLASP_PAGE_TLS_CERT", "CLASP_PAGE_TLS_KEY",
    "CLASP_EVAL_HMAC_SECRET_FILE", "CLASP_UPLOAD_HMAC_SECRETS", "CLASP_JWT_SECRET_FILE",
    "CLASP_USERS_FILE", "CLASP_AUDIT_LOG",
)


def bash() -> str:
    """Git Bash on Windows (``bash`` alone resolves to WSL's System32 stub there)."""
    import shutil

    if os.name == "nt":
        git = shutil.which("git")
        if git:
            # git.exe sits in <Git>/cmd, <Git>/bin or <Git>/mingw64/bin
            for root in Path(git).resolve().parents[:3]:
                if (root / "bin" / "bash.exe").exists() and (root / "usr" / "bin").is_dir():
                    return str(root / "bin" / "bash.exe")
        raise OSError("Git Bash (with openssl) was not found")
    found = shutil.which("bash")
    if not found:
        raise OSError("bash was not found")
    return found


def generate_certs(out_dir: Path, lan_ips: str) -> None:
    """Run ``scripts/make_dev_certs.sh`` into ``out_dir`` with ``lan_ips`` in the
    server certificates (comma-separated; may be empty)."""
    env = dict(os.environ, CLASP_LAN_IPS=lan_ips, MSYS_NO_PATHCONV="1")
    out_dir = Path(out_dir).resolve()
    try:
        target = out_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        target = out_dir.as_posix()
    subprocess.run([bash(), "scripts/make_dev_certs.sh", target], cwd=str(REPO_ROOT), env=env,
                   check=True, stdout=subprocess.DEVNULL)


def ensure_certs(mtls_dir: Path) -> bool:
    """Generate the certificates when ``mtls_dir`` has none; True if it did.

    The LAN addresses in them are ``CLASP_LAN_IPS`` when set, else this
    machine's own, so edge laptops on the hotspot can dial it by address.
    """
    mtls_dir = Path(mtls_dir)
    if (mtls_dir / "ca.pem").exists():
        return False
    lan_ips = os.environ.get("CLASP_LAN_IPS") or ",".join(lan_addresses())
    print(f"no certificates in {mtls_dir}: generating them (scripts/make_dev_certs.sh, "
          f"LAN addresses: {lan_ips or 'none'})")
    try:
        generate_certs(mtls_dir, lan_ips)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"could not generate certificates in {mtls_dir}: {exc}\n"
            f"Run scripts/make_dev_certs.sh yourself (it needs bash and openssl; Git Bash on "
            f"Windows), or start with --insecure for plain HTTP (debugging only).") from exc
    return True


def _pid_file(name: str) -> Path:
    return RUNTIME / f"{name}.pid"


def stop() -> None:
    for name in ALL:
        pf = _pid_file(name)
        if not pf.exists():
            print(f"{name:9s} not running (no pid file)")
            continue
        pid = int(pf.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"{name:9s} stopped (pid {pid})")
        except OSError as exc:
            print(f"{name:9s} pid {pid} would not stop: {exc}")
        pf.unlink(missing_ok=True)


def lan_addresses() -> List[str]:
    """This machine's IPv4 addresses other than loopback — what edges dial."""
    found = set()
    try:
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:  # the address of the default route, even when the hostname lookup misses it
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1; nothing is sent
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith("127."))


def checks_off(mtls_dir: Path, with_ui: bool) -> List[str]:
    """What the stack will run without, because its file is not in ``mtls_dir``
    (a certificate directory made by an older ``make_dev_certs.sh``) or was
    never written. Each service then falls back to "allowed, with a warning" in
    its own log; this is the same news, for the console."""
    mtls_dir = Path(mtls_dir)
    again = "run scripts/make_dev_certs.sh again"
    off = []
    if not (mtls_dir / "secrets" / "eval_hmac.key").is_file():
        off.append(f"evaluation results are not signed or verified (no secrets/eval_hmac.key: {again})")
    if not (mtls_dir / "cluster" / "upload_hmac.json").is_file():
        off.append(f"edge uploads are not signed or verified (no cluster/upload_hmac.json: {again})")
    if with_ui:
        if not ((mtls_dir / "pages" / "pages.pem").is_file()
                and (mtls_dir / "pages" / "pages.key").is_file()):
            off.append(f"the panel is served over plain HTTP (no pages/ certificate: {again})")
        if not USERS_FILE.is_file():
            off.append("the panel has no login (no users file: scripts/make_users.py)")
    return off


def _commands(host: str) -> Dict[str, List[str]]:
    py = sys.executable
    return {
        "registry": [py, "-m", "registry.serve"],
        "cluster": [py, "-m", "cluster.serve"],
        # host, port and the page's HTTPS certificate come from its environment
        "demo-ui": [py, "-m", "demo_ui.serve"],
    }


def _env(name: str, host: str, mtls_dir: Optional[Path], data_dir: Path) -> Dict[str, str]:
    env = dict(os.environ)
    scheme = "https" if mtls_dir else "http"
    env["CLASP_REGISTRY_DATA"] = str(data_dir)
    env["CLASP_REGISTRY_HOST"] = env["CLASP_CLUSTER_HOST"] = host
    env["CLASP_REGISTRY_PORT"] = str(PORTS["registry"])
    env["CLASP_CLUSTER_PORT"] = str(PORTS["cluster"])
    # Services on this machine reach each other over loopback; "localhost" is
    # in every dev cert's SANs, so hostname checking stays on under mTLS.
    env["CLASP_REGISTRY_URL"] = f"{scheme}://localhost:{PORTS['registry']}"
    env["CLASP_CLUSTER_URL"] = f"{scheme}://localhost:{PORTS['cluster']}"
    env["CLASP_DEMO_UI_HOST"] = host
    env["CLASP_DEMO_UI_PORT"] = str(PORTS["demo-ui"])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    for var in MANAGED_VARS:
        env.pop(var, None)
    if not mtls_dir:
        return env  # --insecure: plain HTTP, nothing signed, nobody logs in
    mtls_dir = Path(mtls_dir)

    def existing(var: str, path: Path) -> None:
        """A key or users file is configured only when it is there: a service
        treats a configured-but-missing one as an error, not as "off"."""
        if path.is_file():
            env[var] = str(path)

    env["CLASP_AUDIT_LOG"] = str(RUNTIME / "audit" / f"{name}.jsonl")
    if name == "demo-ui":
        env["CLASP_TLS_CLIENT_CERT"] = str(mtls_dir / "demo-ui" / "demo-ui.pem")
        env["CLASP_TLS_CLIENT_KEY"] = str(mtls_dir / "demo-ui" / "demo-ui.key")
        env["CLASP_TLS_CA"] = str(mtls_dir / "demo-ui" / "ca.pem")
        # the page itself: HTTPS with a server certificate, people log in (C1)
        if (mtls_dir / "pages" / "pages.pem").is_file() and (mtls_dir / "pages" / "pages.key").is_file():
            env["CLASP_PAGE_TLS_CERT"] = str(mtls_dir / "pages" / "pages.pem")
            env["CLASP_PAGE_TLS_KEY"] = str(mtls_dir / "pages" / "pages.key")
        existing("CLASP_JWT_SECRET_FILE", mtls_dir / "secrets" / "jwt.key")
        existing("CLASP_USERS_FILE", USERS_FILE)
        # the evaluator the panel starts signs its results with this
        existing("CLASP_EVAL_HMAC_SECRET_FILE", mtls_dir / "secrets" / "eval_hmac.key")
    else:
        env["CLASP_TLS_CERT"] = str(mtls_dir / name / f"{name}.pem")
        env["CLASP_TLS_KEY"] = str(mtls_dir / name / f"{name}.key")
        env["CLASP_TLS_CA"] = str(mtls_dir / name / "ca.pem")
        if name == "registry":
            existing("CLASP_EVAL_HMAC_SECRET_FILE", mtls_dir / "secrets" / "eval_hmac.key")
        if name == "cluster":
            existing("CLASP_UPLOAD_HMAC_SECRETS", mtls_dir / "cluster" / "upload_hmac.json")
    return env


def _healthy(name: str, env: Dict[str, str]) -> bool:
    """One probe of the service's /healthz, over mTLS when it is on."""
    import urllib.error
    import urllib.request

    port = PORTS[name]
    ctx = None
    if name == "demo-ui" and env.get("CLASP_PAGE_TLS_CERT"):
        url = f"https://127.0.0.1:{port}/api/health"
        ctx = ssl.create_default_context(cafile=env["CLASP_TLS_CA"])
        ctx.check_hostname = False  # loopback probe; the chain is still verified
    elif name == "demo-ui":
        url = f"http://127.0.0.1:{port}/api/health"
    elif env.get("CLASP_TLS_CERT"):
        url = f"https://127.0.0.1:{port}/healthz"
        ctx = ssl.create_default_context(cafile=env["CLASP_TLS_CA"])
        ctx.check_hostname = False  # loopback probe; the chain is still verified
        ctx.load_cert_chain(env["CLASP_TLS_CERT"], env["CLASP_TLS_KEY"])
    else:
        url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=2, context=ctx) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start(background: bool, host: str, with_ui: bool, mtls_dir: Optional[Path],
          data_dir: Path = RUNTIME / "registry-data") -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "audit").mkdir(parents=True, exist_ok=True)
    if mtls_dir:
        mtls_dir = mtls_dir.resolve()
        ensure_certs(mtls_dir)
        need = [mtls_dir / "registry" / "registry.pem", mtls_dir / "cluster" / "cluster.pem"]
        if with_ui:
            need.append(mtls_dir / "demo-ui" / "demo-ui.pem")
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise SystemExit(f"missing certificates {missing}; run scripts/make_dev_certs.sh first")
    else:
        print("WARNING --insecure: plain HTTP, no client certificates, no signatures checked, "
              "no logins. For debugging only.")
    scheme = "https" if mtls_dir else "http"
    names = [n for n in ALL if with_ui or n != "demo-ui"]
    commands = _commands(host)

    procs = []
    for name in names:
        log_path = RUNTIME / f"{name}.log"
        env = _env(name, host, mtls_dir, data_dir)
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(commands[name], env=env, cwd=str(REPO_ROOT),
                                stdout=log_file, stderr=subprocess.STDOUT)
        _pid_file(name).write_text(str(proc.pid))
        procs.append((name, proc, log_path, env))
        print(f"{name:9s} -> {host}:{PORTS[name]}   (pid {proc.pid}, log {log_path})")

    deadline = time.time() + 90
    for name, proc, log_path, env in procs:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(
                    f"{name} exited with {proc.returncode}; see {log_path}\n"
                    + log_path.read_text(encoding="utf-8")[-2000:])
            if _healthy(name, env):
                print(f"{name:9s} healthy")
                break
            time.sleep(0.5)
        else:
            raise SystemExit(f"{name} did not become healthy within 90s; see {log_path}")

    if host in ("0.0.0.0", "::"):
        ips = lan_addresses()
        print("\nreachable from other machines on this network at:"
              if ips else "\nno LAN address found - is this machine on the hotspot?")
        for ip in ips:
            print(f"  cluster  {scheme}://{ip}:{PORTS['cluster']}    "
                  f"registry {scheme}://{ip}:{PORTS['registry']}"
                  + (f"    demo_ui {scheme}://{ip}:{PORTS['demo-ui']}" if with_ui else ""))
        if ips and mtls_dir:
            print("  (each IP must be in the certificates: CLASP_LAN_IPS=... scripts/make_dev_certs.sh)")
    if mtls_dir:
        for line in checks_off(mtls_dir, with_ui):
            print(f"WARNING {line}")
    if with_ui and mtls_dir:
        page_scheme = "https" if (mtls_dir / "pages" / "pages.pem").is_file() else "http"
        print(f"\nlive panel: {page_scheme}://localhost:{PORTS['demo-ui']}"
              + ("  (asks for a login)" if USERS_FILE.is_file() else ""))
        if page_scheme == "https":
            print(f"  the browser must trust {mtls_dir / 'ca.pem'} (import it once), or it will warn")

    if background:
        print("\nrunning in the background - stop with: python scripts/run_services.py --stop")
        return
    print("\nctrl-c to stop")
    try:
        while all(p.poll() is None for _, p, _, _ in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for name, proc, _log, _env_ in procs:
            proc.terminate()
            _pid_file(name).unlink(missing_ok=True)
        print("stopped")


def mtls_dir_of(args) -> Optional[Path]:
    """The certificate directory to serve with, or ``None`` for ``--insecure``."""
    if args.insecure:
        if args.mtls_dir is not None:
            raise SystemExit("--insecure and --mtls-dir contradict each other: pick one")
        return None
    return args.mtls_dir if args.mtls_dir is not None else DEFAULT_MTLS_DIR


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="run registry + cluster (+ demo_ui) on the host")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (default 0.0.0.0: reachable from the LAN; "
                         "127.0.0.1 for this machine only)")
    ap.add_argument("--ui", action="store_true", help="also start demo_ui on :8010")
    ap.add_argument("--mtls-dir", type=Path, default=None,
                    help="certificates from scripts/make_dev_certs.sh (default .runtime/certs, "
                         "generated when missing): serve mTLS + TLS 1.3")
    ap.add_argument("--insecure", action="store_true",
                    help="plain HTTP with no certificates, signatures or logins (debugging only)")
    ap.add_argument("--registry-data", type=Path, default=RUNTIME / "registry-data",
                    help="where the registry keeps its versions (default .runtime/registry-data)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.stop:
        stop()
    else:
        start(args.background, args.host, args.ui, mtls_dir_of(args), args.registry_data)


if __name__ == "__main__":
    main()
