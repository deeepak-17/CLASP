"""Start the registry and cluster services without docker (Integration Sprint · B7).

`docker compose up -d registry cluster` is the documented path and the one the
panel demo uses. This is the same two services on the host, for when the
docker daemon is not running — which is exactly the situation the Windows
demo machine was in while this sprint was built.

    python scripts/run_services.py                # foreground, ctrl-c to stop
    python scripts/run_services.py --stop         # stop a background set
    python scripts/run_services.py --background   # detach and return
    python scripts/run_services.py --ui           # + demo_ui on :8010 (laptop A)
    python scripts/run_services.py --mtls-dir .runtime/certs   # mTLS (make_dev_certs.sh)

Multi-laptop demo (docs/multi-laptop-demo.md): services bind 0.0.0.0 by default
so edge laptops on the same network can reach laptop A; ``--host 127.0.0.1``
keeps them on this machine only. The addresses to give the edges are printed
on start.

Each service runs through its own launcher (``registry.serve``,
``cluster.serve``), so with ``--mtls-dir`` both require client certificates and
TLS 1.3, exactly as under docker-compose.mtls.yml.

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


def _commands(host: str) -> Dict[str, List[str]]:
    py = sys.executable
    return {
        "registry": [py, "-m", "registry.serve"],
        "cluster": [py, "-m", "cluster.serve"],
        "demo-ui": [py, "-m", "uvicorn", "demo_ui.server:app", "--host", host,
                    "--port", str(PORTS["demo-ui"]), "--log-level", "warning"],
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
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    for var in ("CLASP_TLS_CERT", "CLASP_TLS_KEY", "CLASP_TLS_CA",
                "CLASP_TLS_CLIENT_CERT", "CLASP_TLS_CLIENT_KEY"):
        env.pop(var, None)
    if mtls_dir:
        if name == "demo-ui":
            env["CLASP_TLS_CLIENT_CERT"] = str(mtls_dir / "demo-ui" / "demo-ui.pem")
            env["CLASP_TLS_CLIENT_KEY"] = str(mtls_dir / "demo-ui" / "demo-ui.key")
            env["CLASP_TLS_CA"] = str(mtls_dir / "demo-ui" / "ca.pem")
        else:
            env["CLASP_TLS_CERT"] = str(mtls_dir / name / f"{name}.pem")
            env["CLASP_TLS_KEY"] = str(mtls_dir / name / f"{name}.key")
            env["CLASP_TLS_CA"] = str(mtls_dir / name / "ca.pem")
    return env


def _healthy(name: str, env: Dict[str, str]) -> bool:
    """One probe of the service's /healthz, over mTLS when it is on."""
    import urllib.error
    import urllib.request

    port = PORTS[name]
    ctx = None
    if name == "demo-ui":
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
    if mtls_dir:
        mtls_dir = mtls_dir.resolve()
        need = [mtls_dir / "registry" / "registry.pem", mtls_dir / "cluster" / "cluster.pem"]
        if with_ui:
            need.append(mtls_dir / "demo-ui" / "demo-ui.pem")
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise SystemExit(f"missing certificates {missing}; run scripts/make_dev_certs.sh first")
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
                  + (f"    demo_ui http://{ip}:{PORTS['demo-ui']}" if with_ui else ""))
        if ips and mtls_dir:
            print("  (each IP must be in the certificates: CLASP_LAN_IPS=... scripts/make_dev_certs.sh)")

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


def main() -> None:
    ap = argparse.ArgumentParser(description="run registry + cluster (+ demo_ui) on the host")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (default 0.0.0.0: reachable from the LAN; "
                         "127.0.0.1 for this machine only)")
    ap.add_argument("--ui", action="store_true", help="also start demo_ui on :8010")
    ap.add_argument("--mtls-dir", type=Path, default=None,
                    help="certificates from scripts/make_dev_certs.sh: serve mTLS + TLS 1.3")
    ap.add_argument("--registry-data", type=Path, default=RUNTIME / "registry-data",
                    help="where the registry keeps its versions (default .runtime/registry-data)")
    args = ap.parse_args()
    if args.stop:
        stop()
    else:
        start(args.background, args.host, args.ui, args.mtls_dir, args.registry_data)


if __name__ == "__main__":
    main()
