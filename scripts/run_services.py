"""Start the registry and cluster services without docker (Integration Sprint · B7).

`docker compose up -d registry cluster` is the documented path and the one the
panel demo uses. This is the same two services under uvicorn on the host, for
when the docker daemon is not running — which is exactly the situation the
Windows demo machine was in while this sprint was built.

    python scripts/run_services.py                # foreground, ctrl-c to stop
    python scripts/run_services.py --stop         # stop a background pair
    python scripts/run_services.py --background   # detach and return

Registry state goes to ``.runtime/registry-data`` under the repo (gitignored),
which is the host-side equivalent of compose's ``registry-data`` volume: the
same D9 requirement that versioned safetensors survive a restart.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / ".runtime"
SERVICES = {
    "registry": ("registry.app:app", 8004),
    "cluster": ("cluster.server:app", 8002),
}


def _pid_file(name: str) -> Path:
    return RUNTIME / f"{name}.pid"


def stop() -> None:
    for name in SERVICES:
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


def start(background: bool) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "registry-data").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CLASP_REGISTRY_DATA"] = str(RUNTIME / "registry-data")

    procs = []
    for name, (target, port) in SERVICES.items():
        log_path = RUNTIME / f"{name}.log"
        cmd = [sys.executable, "-m", "uvicorn", target,
               "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        _pid_file(name).write_text(str(proc.pid))
        procs.append((name, port, proc, log_path))
        print(f"{name:9s} -> http://127.0.0.1:{port}   (pid {proc.pid}, log {log_path})")

    import urllib.error
    import urllib.request

    deadline = time.time() + 60
    for name, port, proc, log_path in procs:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(
                    f"{name} exited with {proc.returncode}; see {log_path}\n"
                    + log_path.read_text(encoding="utf-8")[-2000:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                    if r.status == 200:
                        print(f"{name:9s} healthy")
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            raise SystemExit(f"{name} did not become healthy within 60s; see {log_path}")

    if background:
        print("\nrunning in the background — stop with: python scripts/run_services.py --stop")
        return
    print("\nctrl-c to stop")
    try:
        while all(p.poll() is None for _, _, p, _ in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for name, _port, proc, _log in procs:
            proc.terminate()
            _pid_file(name).unlink(missing_ok=True)
        print("stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description="run registry + cluster on the host")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    stop() if args.stop else start(args.background)


if __name__ == "__main__":
    main()
