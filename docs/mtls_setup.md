# CLASP transport security: mTLS, HTTPS pages, roles, signatures

How the services and the pages talk to each other, as built. Decisions: U4–U6 and
C1–C3, C10, C11 in `docs/decisions/2026-11-14-panel-completion-decisions.md`.

## Summary

| Channel | Protection | Who is identified, and how |
|---|---|---|
| Edge → Cluster (uploads, pulls) | mutual TLS 1.3 | the edge, by its certificate (`client-<repo>`); an upload's `client_id` must equal the certificate's name; the upload body is HMAC-signed with that edge's key |
| Cluster → Registry (publish) | mutual TLS 1.3 | the cluster, by its certificate (`cluster`) |
| Panel and evaluator → Cluster, Registry | mutual TLS 1.3 | by the `demo-ui` certificate; evaluation results are HMAC-signed by the evaluator |
| Operator → Cluster, Registry | mutual TLS 1.3 | by the `clasp-admin` certificate |
| Browser → live panel, dashboard, edge page | HTTPS (server certificate only) | the person, by login (JWT, 8 hours) on the panel and the edge page |

A browser cannot easily present a client certificate, so the three pages use a
server certificate and people log in (C1). Services never log in: they are
identified by certificate (U5).

**API keys.** API-key authentication exists in the security library
(`security.auth.api_key`) and is not used, because services are identified by
certificate.

## It is on by default

- `docker compose up` and `python scripts/run_services.py` start the registry and
  the cluster behind mutual TLS, and the panel over HTTPS. The certificates come
  from `.runtime/certs`; they are generated there the first time
  (`scripts/make_dev_certs.sh`), by the `certs` service under compose and by
  `run_services.py` itself on the host.
- `python scripts/run_services.py --insecure`, or
  `-f docker-compose.yml -f docker-compose.insecure.yml`, starts plain HTTP with
  nothing signed and no logins. For debugging only.
- A service started by hand without `CLASP_TLS_CERT`, `CLASP_TLS_KEY` and
  `CLASP_TLS_CA` still runs plain HTTP, and logs a warning (U6). A partial or
  unreadable TLS setting is a startup error, never a silent fallback.

## The fall-back rule

Applied the same way to every check (U6, C11):

- **Not configured → allowed**, with one warning when the service starts and an
  entry per request in the audit trail (the registry's access log records the
  caller as role `open`).
- **Configured but missing or wrong → refused.** A key file that is named and
  cannot be read, a users file that is corrupt, a signature that does not verify:
  the request is refused, not waved through.

## Certificates

`scripts/make_dev_certs.sh [out_dir]` (default `.runtime/certs`, gitignored)
writes a throwaway CA and everything signed by it. It needs bash and openssl
(Git Bash on Windows). Certificates last 30 days.

```bash
CLASP_LAN_IPS=192.168.43.10 scripts/make_dev_certs.sh
```

| Path under `.runtime/certs` | Certificate name (CN) | Used by |
|---|---|---|
| `registry/` | `registry` | the registry's server certificate (SANs: `registry`, `localhost`, `127.0.0.1`, `CLASP_LAN_IPS`) |
| `cluster/` | `cluster` | the cluster's server certificate (same SANs); also presented to the registry on publish |
| `cluster/upload_hmac.json` | | every edge's upload key, for the cluster |
| `demo-ui/` | `demo-ui` | the panel's and the evaluator's calls to the cluster and the registry |
| `admin/` | `clasp-admin` | the operator (registry GC, restores, the `demo-seed` job) |
| `pages/` | `clasp-pages` | HTTPS for the live panel and the dashboard (server certificate; SANs as above) |
| `edges/<client_id>/` | `<client_id>` | that edge laptop: `client.pem`, `client.key`, `ca.pem`, `hmac.key` (its upload key), `page.pem`, `page.key` (its own page over HTTPS) |
| `client.pem` | `clasp-dev-client` | a certificate with no role, for testing the handshake |
| `secrets/eval_hmac.key` | | signs and verifies evaluation results |
| `secrets/jwt.key` | | signs the panel's login tokens |
| `ca-private/ca.key` | | the CA key: stays on the machine that made it; nothing mounts or copies it |

`CLASP_LAN_IPS` puts laptop A's address into the server certificates so edges can
dial it by address with hostname checking on. Hotspot addresses change: when the
address is not in the certificate, an edge's upload fails with a hint that names
the address it tried, the addresses the certificate lists, and this command.
A fixed tunnel address avoids the problem (`docs/remote-access.md`).

Edge laptops get their directory inside their bundle
(`scripts/make_edge_bundle.py`, which carries `edges/<client_id>/` by default).
A bundle holds that laptop's private key and upload key: hand it over directly.

**TLS policy.** TLS 1.3 only on the service channels, a client certificate signed
by the CA required (`registry.serve`, `cluster.serve`); the pages accept TLS 1.2
and later. The verified certificate name reaches the routes through
`security.mtls.peer.PeerCertH11Protocol`.

## Roles

A service's role comes from the name in its certificate (`security.auth.roles`).
The table can be replaced with a JSON file named in `CLASP_ROLE_TABLE`.

| Certificate name | Role |
|---|---|
| `cluster` | `svc:cluster` |
| `demo-ui` | `svc:evaluation` |
| `clasp-admin` | `admin` |
| a name starting `client-` | `edge` |
| any other CA-signed name | `unknown` (refused by every restricted route) |
| a CA-signed certificate with no CN | `unknown` (never read as "no TLS") |
| no certificate (plain HTTP) | `open` (the fall-back: everything allowed) |

**Registry**

| Routes | Allowed roles |
|---|---|
| `GET /healthz` | everyone |
| all other `GET` except `/access-log` | `svc:cluster`, `svc:evaluation`, `admin` |
| `POST /adapters/{name}/versions` | `svc:cluster`, `admin` |
| `POST .../promote`, `POST .../compose`, `POST /sync/{id}/done`, `POST .../restore` | `svc:evaluation`, `admin` |
| `POST .../gc`, `POST /gc`, `GET /access-log` | `admin` |

An edge gets 403 from every registry route except `/healthz`: edges talk to the
cluster only.

**Cluster**

| Routes | Rule |
|---|---|
| `POST .../uploads` | the certificate's name must equal the upload's `client_id` |
| `GET .../adapters/active` | a member of another cluster is refused |
| `aggregate`, `publish`, `candidate/promote`, `candidate/reject`, `active/restore`, `recluster`, `PUT .../members`, `GET .../adapters/candidate` | `svc:evaluation`, `admin` |

## Signatures (HMAC-SHA256)

- **Evaluation results.** The evaluator (`scripts/evaluate_candidate.py`) signs
  `{adapter, version, candidate_sha256, eval, baseline_guard}` with
  `CLASP_EVAL_HMAC_SECRET` (or the file `CLASP_EVAL_HMAC_SECRET_FILE` names). The
  panel forwards the signed body unchanged. The registry verifies it on
  `/promote`: a wrong or missing signature is 401 and nothing is recorded; a
  `candidate_sha256` that is not the stored version's is 409 ("evaluation was run
  on different bytes").
- **Edge uploads.** Each edge signs its upload (every tensor's raw bytes, dtype
  and shape, and the rest of the body) with its own key
  (`edges/<client_id>/hmac.key`). Edge and cluster use the one recipe in
  `security.auth.hmac_signer` (`sign_upload_body` / `verify_upload_body`). The
  cluster verifies with its copy (`CLASP_UPLOAD_HMAC_SECRETS`): a changed tensor,
  a relabelled shape or a changed sample count is 401 `upload_signature_invalid`
  and the upload is not buffered. Once that file is configured, a client it holds
  no key for is refused too (401 `upload_key_unknown`): it cannot be checked.

## Logins (JWT)

Users are in a JSON file per page, written by `scripts/make_users.py` (scrypt
hashes; the password is prompted for, never an argument; no self-registration).

```powershell
python scripts/make_users.py --out .runtime/users/panel.json --add operator:operator
python scripts/make_users.py --out .runtime/users/client-flask.json --add asha:developer:client-flask
```

- The panel reads `CLASP_USERS_FILE`; `run_services.py` and compose point it at
  `.runtime/users/panel.json`. Role `operator`.
- An edge page reads `--users`; the edge bundle carries
  `.runtime/users/<client_id>.json`. Role `developer` of that client.
- `POST /api/login` gives a token that lasts 8 hours. Every other `/api/*` route
  except the health route needs it. A wrong or expired token is 401 and the page
  shows its login form; the round's state stays on the server. The right token
  with the wrong role is 403.
- Tokens are signed with `CLASP_JWT_SECRET` (or `CLASP_JWT_SECRET_FILE`); with
  neither, a key made for that process, so logins end when the page restarts.
- No users file: the page runs without a login and says so when it starts.

## In front of every route

The registry, the cluster and the panel each put, in the order a request meets
them: security headers; a body limit (registry: the upload cap plus 1 MiB;
cluster: 512 MiB; panel: 1 MiB); a rate limit of 600 requests a minute per caller
in bursts of 60 (`CLASP_RATE_LIMIT_PER_MIN`, `0` = off; the caller is the
certificate name, else the remote address; the health route is exempt); and the
audit recorder.

## Audit

Each service writes JSON lines to `CLASP_AUDIT_LOG`: every refused authorisation,
every accepted mutating call, every privacy-budget refusal and every recorded
privacy row. The registry defaults to `audit.jsonl` under its data root and also
keeps every request in its `access_log` table (`GET /access-log`, role `admin`).
`run_services.py` writes to `.runtime/audit/<service>.jsonl`. No line holds a
secret, a token or a signature.

## Checking it

```bash
# a certificate the CA signed: 200
curl --cacert .runtime/certs/ca.pem --cert .runtime/certs/client.pem \
     --key .runtime/certs/client.key https://localhost:8004/healthz
# no certificate: the handshake is refused
curl --cacert .runtime/certs/ca.pem https://localhost:8004/healthz
# the same certificate has no role: 403 on anything but /healthz
curl --cacert .runtime/certs/ca.pem --cert .runtime/certs/client.pem \
     --key .runtime/certs/client.key https://localhost:8004/adapters
# the operator's certificate: 200
curl --cacert .runtime/certs/ca.pem --cert .runtime/certs/admin/admin.pem \
     --key .runtime/certs/admin/admin.key https://localhost:8004/adapters
```

The panel at `https://localhost:8010` asks for a login. Browsers warn about the
development CA until `.runtime/certs/ca.pem` is imported as a trusted root.

## The library

`security.mtls` also has a Python CA (`create_ca`, `generate_service_cert`,
`generate_client_cert`, `setup_all_certs`), `verify_cert`, server and client SSL
contexts, and expiry checks (`check_expiry`). The tests use them to build
throwaway chains. The demo's certificates come from the shell script above,
because the cluster, the registry and the bundles read that layout.
