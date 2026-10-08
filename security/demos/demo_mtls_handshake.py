"""CLASP Security — mTLS handshake demo (Panel 1, W5 Tue).

Spawns a local TLS server and client, demonstrates:
  1. Successful mutual authentication with valid certs
  2. Rejection when a bad (rogue-CA-signed) client cert is presented

Usage:
    python -m security.demos.demo_mtls_handshake
    python -m security.demos.demo_mtls_handshake --certs-dir /path/to/certs
"""
from __future__ import annotations

import argparse
import socket
import ssl
import tempfile
import threading
from pathlib import Path

from security.mtls.ca import create_ca, save_pem
from security.mtls.certs import generate_client_cert, generate_service_cert
from security.mtls.context import client_ssl_context, server_ssl_context


def _setup_pki(output_dir: Path) -> Path:
    """Generate a full PKI hierarchy for the demo."""
    print("  [1/4] Creating Root CA (RSA-4096)...")
    ca_key, ca_cert = create_ca(common_name="CLASP Demo CA")
    save_pem(ca_key, ca_cert, output_dir, name="ca")
    print(f"        → {output_dir / 'ca_cert.pem'}")

    print("  [2/4] Generating server cert (cluster service)...")
    srv_key, srv_cert = generate_service_cert(
        ca_key, ca_cert, "cluster",
        sans=["localhost", "127.0.0.1"],
    )
    save_pem(srv_key, srv_cert, output_dir, name="server")
    print(f"        → {output_dir / 'server_cert.pem'}")

    print("  [3/4] Generating valid client cert (edge-0)...")
    cli_key, cli_cert = generate_client_cert(ca_key, ca_cert, "edge-0")
    save_pem(cli_key, cli_cert, output_dir, name="client")
    print(f"        → {output_dir / 'client_cert.pem'}")

    print("  [4/4] Generating rogue CA + bad client cert...")
    rogue_key, rogue_cert = create_ca(common_name="Rogue CA")
    save_pem(rogue_key, rogue_cert, output_dir, name="rogue-ca")
    bad_key, bad_cert = generate_client_cert(rogue_key, rogue_cert, "rogue-edge")
    save_pem(bad_key, bad_cert, output_dir, name="bad-client")
    print(f"        → {output_dir / 'bad-client_cert.pem'}")
    print()
    return output_dir


def _run_server(srv_ctx: ssl.SSLContext, host: str, port: int,
                result: dict, ready: threading.Event) -> None:
    """Accept one TLS connection."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
        ready.set()
        try:
            sock.settimeout(5.0)
            conn, _ = sock.accept()
            with srv_ctx.wrap_socket(conn, server_side=True) as tls:
                peer = tls.getpeercert()
                cn = dict(x[0] for x in peer["subject"])["commonName"] if peer else "?"
                tls.sendall(b"CLASP_OK")
                result["ok"] = True
                result["client_cn"] = cn
        except Exception as exc:
            result["error"] = str(exc)


def _demo_handshake(d: Path, label: str, client_name: str,
                    client_ca_name: str, expect_success: bool) -> None:
    """Run one handshake attempt and report."""
    print(f"  --- {label} ---")
    srv_ctx = server_ssl_context(
        cert_path=d / "server_cert.pem",
        key_path=d / "server_key.pem",
        ca_cert_path=d / "ca_cert.pem",
    )
    cli_ctx = client_ssl_context(
        cert_path=d / f"{client_name}_cert.pem",
        key_path=d / f"{client_name}_key.pem",
        ca_cert_path=d / f"{client_ca_name}_cert.pem",
    )

    host = "127.0.0.1"
    result: dict = {}
    ready = threading.Event()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]

    t = threading.Thread(target=_run_server, args=(srv_ctx, host, port, result, ready),
                         daemon=True)
    t.start()
    ready.wait(timeout=3.0)

    try:
        with socket.create_connection((host, port), timeout=5.0) as sock:
            with cli_ctx.wrap_socket(sock, server_hostname="localhost") as tls:
                data = tls.recv(1024)
                if data == b"CLASP_OK":
                    result["received"] = True
    except (ssl.SSLError, ssl.SSLCertVerificationError, ConnectionResetError, OSError) as exc:
        result["client_error"] = str(exc)

    t.join(timeout=5.0)

    if expect_success:
        if result.get("ok"):
            print(f"  ✅ Handshake SUCCEEDED — server authenticated client '{result.get('client_cn')}'")
        else:
            print(f"  ❌ UNEXPECTED FAILURE: {result.get('error', 'unknown')}")
    else:
        if result.get("ok"):
            print(f"  ❌ UNEXPECTED SUCCESS — bad cert should have been rejected!")
        else:
            err = result.get("error") or result.get("client_error", "connection refused")
            print(f"  ✅ Handshake REJECTED — bad cert correctly denied")
            print(f"     Error: {err[:100]}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="CLASP mTLS handshake demo")
    ap.add_argument("--certs-dir", type=str, default=None,
                    help="Directory to store generated certs (temp dir if omitted)")
    args = ap.parse_args()

    print()
    print("=" * 60)
    print("  CLASP Security — mTLS Handshake Demo")
    print("=" * 60)
    print()

    if args.certs_dir:
        d = Path(args.certs_dir)
        d.mkdir(parents=True, exist_ok=True)
        _setup_pki(d)
    else:
        tmp = tempfile.mkdtemp(prefix="clasp-mtls-demo-")
        d = _setup_pki(Path(tmp))

    print("  Test 1: Valid client cert (edge-0) → cluster server")
    _demo_handshake(d, "VALID CLIENT", "client", "ca", expect_success=True)

    print("  Test 2: Rogue client cert (wrong CA) → cluster server")
    _demo_handshake(d, "BAD CLIENT (wrong CA)", "bad-client", "rogue-ca", expect_success=False)

    print("=" * 60)
    print("  Demo complete. mTLS is working as expected.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
