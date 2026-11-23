# Reaching laptop A's services over a WireGuard tunnel

**Status: written, not yet followed end to end.** The commands below are for
Windows laptops with WireGuard for Windows. They still have to be run once on
two laptops and corrected where they do not work as written (plan B, task 10.1
step 2).

## Why a private tunnel and not a public address

The registry and the cluster hold adapters trained on private code. Each
adapter is a few million numbers derived from a developer's repository; the
merged cluster adapter is derived from several. Putting ports 8002, 8004 or
8010 on a public address would put those within reach of anyone who can dial
it, with certificate checks as the only barrier.

A tunnel keeps the services off every public interface. The laptops reach
laptop A at a private address that exists only inside the tunnel, and mutual TLS
(`docs/mtls_setup.md`) still applies on top: the tunnel decides who can reach
the port, the certificate decides what they may do.

It also fixes a practical problem. Hotspot addresses change between sessions,
and the certificates name laptop A's address, so every new hotspot meant new
certificates and new edge bundles. A tunnel address never changes.

## Why plain WireGuard

- **Three laptops, fixed peers.** Each laptop is listed once in one
  configuration file. There is nothing to coordinate, so there is no control
  server to run.
- **Headscale** (a self-hosted coordination server for the Tailscale clients)
  becomes worth running with more than a handful of machines, or when peers come
  and go and should not each be edited into a file by hand. It adds a service to
  run and keep up.
- **Tailscale's hosted coordination** is ruled out here: the coordination server
  would be a third party's, and CLASP's claim is that everything stays on
  premises.

## Layout

Laptop A is the hub. Each edge laptop connects to A only.

| Laptop | Tunnel address |
|---|---|
| A (registry, cluster, panel) | `10.77.0.1` |
| B (edge) | `10.77.0.2` |
| C (edge) | `10.77.0.3` |

Laptop A must be reachable from B and C on **UDP 51820**: on the same network
(the hotspot), or through a port forward on A's router.

## 1. Install

On every laptop install WireGuard for Windows (`wireguard.com/install`). The
command-line tools are in `C:\Program Files\WireGuard\`.

## 2. Keys

On each laptop, in PowerShell:

```powershell
cd "C:\Program Files\WireGuard"
.\wg.exe genkey | Tee-Object -FilePath "$env:USERPROFILE\clasp-wg.key" | .\wg.exe pubkey
```

The command prints the laptop's **public** key: send that to laptop A's
operator. The private key stays in `clasp-wg.key` on the laptop that made it and
is never sent anywhere.

## 3. Laptop A: the hub

Save as `C:\clasp\clasp.conf` (private key from A's `clasp-wg.key`, one `[Peer]`
block per edge laptop with that laptop's public key):

```ini
[Interface]
PrivateKey = <A's private key>
Address = 10.77.0.1/24
ListenPort = 51820

[Peer]
# laptop B
PublicKey = <B's public key>
AllowedIPs = 10.77.0.2/32

[Peer]
# laptop C
PublicKey = <C's public key>
AllowedIPs = 10.77.0.3/32
```

`AllowedIPs` with `/32` means A accepts, from each peer, only that peer's own
tunnel address.

Start it as a service (PowerShell as administrator). The tunnel and its network
interface are named after the file, `clasp`:

```powershell
& "C:\Program Files\WireGuard\wireguard.exe" /installtunnelservice C:\clasp\clasp.conf
```

Let the tunnel in, and let the services answer **only** on the tunnel interface:

```powershell
New-NetFirewallRule -DisplayName "CLASP WireGuard" -Direction Inbound -Protocol UDP `
  -LocalPort 51820 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "CLASP services (tunnel only)" -Direction Inbound -Protocol TCP `
  -LocalPort 8002,8004,8005,8010 -Action Allow -InterfaceAlias "clasp"
```

If the earlier rule from `docs/multi-laptop-demo.md` ("CLASP demo", every
interface) exists, remove it, or the services stay reachable from the hotspot:

```powershell
Remove-NetFirewallRule -DisplayName "CLASP demo"
```

## 4. Laptops B and C: the edges

Save as `C:\clasp\clasp.conf` on B (on C: `Address = 10.77.0.3/32`):

```ini
[Interface]
PrivateKey = <this laptop's private key>
Address = 10.77.0.2/32

[Peer]
# laptop A
PublicKey = <A's public key>
Endpoint = <A's address on the shared network>:51820
AllowedIPs = 10.77.0.1/32
PersistentKeepalive = 25
```

`AllowedIPs = 10.77.0.1/32` sends only traffic for laptop A through the tunnel;
everything else on the laptop is untouched. `PersistentKeepalive` keeps the
tunnel open through a NAT.

```powershell
& "C:\Program Files\WireGuard\wireguard.exe" /installtunnelservice C:\clasp\clasp.conf
```

`Endpoint` is the one address that can still change (A's address on the hotspot
or its router's public address). Changing it means editing this one line and
restarting the tunnel; certificates and bundles are not affected.

## 5. Certificates for the tunnel address

On laptop A, in Git Bash, issue the certificates for the tunnel address, so they
stop changing with the hotspot:

```bash
CLASP_LAN_IPS=10.77.0.1 scripts/make_dev_certs.sh
```

Then restart the services and make the edge bundles against the tunnel address:

```powershell
python scripts/run_services.py --ui
python scripts/make_edge_bundle.py client-flask client-numpy --cluster-url https://10.77.0.1:8002
```

An edge that still has certificates for an old address is told so when it
uploads: the message names the address it dialled, the addresses in the
certificate, and the command above.

## 6. Check list

From an edge laptop:

1. The tunnel is up and A answers:

   ```powershell
   & "C:\Program Files\WireGuard\wg.exe" show clasp   # a recent "latest handshake"
   ping 10.77.0.1
   ```

2. The cluster answers over mutual TLS at the tunnel address, with the
   certificate from this laptop's bundle:

   ```powershell
   curl.exe --cacert edge-certs\client-flask\ca.pem --cert edge-certs\client-flask\client.pem `
     --key edge-certs\client-flask\client.key https://10.77.0.1:8002/healthz
   ```

   Expect `200`. The same without `--cert` and `--key` must fail at the
   handshake.

3. From laptop A, `Get-NetFirewallRule -DisplayName "CLASP*"` lists the tunnel
   rule and the tunnel-only services rule, and not "CLASP demo".

## What must still not be exposed

Ports **8002** (cluster), **8004** (registry) and **8010** (panel) must not be
reachable on any public interface, and port 8005 (dashboard) should not be
either. Do not forward them on a router. The only port a router may forward to
laptop A is UDP 51820.

`run_services.py` binds `0.0.0.0` by default, so it is the firewall rule in
step 3 that keeps the services on the tunnel. To bind them to the tunnel address
itself, start with `--host 10.77.0.1` (the tunnel must be up first).

## Stopping

```powershell
& "C:\Program Files\WireGuard\wireguard.exe" /uninstalltunnelservice clasp
```
