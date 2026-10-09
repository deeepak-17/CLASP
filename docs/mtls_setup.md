# CLASP mTLS Setup Guide

## Overview
This document describes the mutual TLS (mTLS) setup for CLASP, securing communications between Edge ↔ Cluster, Cluster ↔ Registry, and Registry ↔ Eval.

## TLS policy (D7)
Every CLASP channel requires **TLS 1.3** with a client certificate signed by the
CLASP CA. `security.mtls.server_ssl_context` and `client_ssl_context` both set
`minimum_version = TLSv1_3`, matching the registry's server (`registry.serve`)
and the edge's client, so no caller has to tighten or loosen a context. All CLASP
services run on Python 3.11 / OpenSSL 3, so no peer needs TLS 1.2.

`security.mtls.verify_cert(cert, ca_cert)` returns True only if the issuer matches
the CA subject, the current time is inside the certificate's validity window, and
the CA's signature verifies.

## Prerequisites
- Python 3.10+
- `cryptography` package (`pip install cryptography`)

## Quick Start
Generate all required certificates for a standard development environment:
```bash
python -m security.mtls.certs --setup-all --output-dir security/certs/
```

## Step-by-Step CA Creation
1. Generate the Root CA private key.
2. Create a self-signed Root CA certificate.
3. This CA will be used to sign all subsequent service and client certificates.

## Service Cert Generation
For each CLASP service (Cluster, Registry, Eval):
1. Generate a private key.
2. Generate a CSR (Certificate Signing Request) specifying the service hostname (e.g., `cluster-service`, `localhost`).
3. Sign the CSR with the Root CA to generate the service certificate.

## Client Cert Generation
For Edge nodes (Clients):
1. Generate a client private key.
2. Generate a CSR for the client (e.g., `CN=edge-client-01`).
3. Sign with the Root CA, explicitly marking it for Client Authentication (Extended Key Usage).

## Docker Compose Volume Mounting
Update `docker-compose.yml` to mount the generated certificates:
```yaml
services:
  cluster:
    volumes:
      - ./security/certs/ca.crt:/etc/ssl/certs/ca.crt:ro
      - ./security/certs/cluster.crt:/etc/ssl/certs/server.crt:ro
      - ./security/certs/cluster.key:/etc/ssl/certs/server.key:ro
```

## Certificate Rotation Procedure
Use the `security.mtls.rotation` module to automate rotation before expiration:
- The rotation agent periodically checks certificate validities.
- If a cert is within the renewal window (e.g., 7 days before expiry), a new CSR is generated and signed.
- Services are signaled to reload their SSL contexts.

## Verification
Test the handshake using `curl`:
```bash
curl -v --cacert ca.crt --cert edge.crt --key edge.key https://cluster-service:8443/api/v1/status
```

## Troubleshooting
- **Bad Certificate**: Ensure the client cert is sent and valid.
- **Expired Certificate**: Check cert validity dates (`openssl x509 -in cert.crt -text -noout`).
- **Wrong CA**: Ensure both client and server trust the same Root CA.
- **Hostname Mismatch**: Ensure the server certificate's SAN (Subject Alternative Name) includes the hostname used in the request.

## Security Notes
- **Storage**: Never commit private keys to version control.
- **Permissions**: Set key file permissions to `0600` (read/write by owner only).
- **Production**: Use an external PKI or Vault for production deployments rather than ad-hoc scripts.
