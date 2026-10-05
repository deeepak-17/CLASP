#!/usr/bin/env bash
# Throwaway CA + registry server cert + one client cert, for trying the registry
# behind mTLS locally (docker-compose.mtls.yml). DEV ONLY — production keys come
# from the security library's CA (P3) and never live in the repo.
#
#   scripts/make_dev_certs.sh [out_dir]      # default: .runtime/certs (gitignored)
set -euo pipefail

OUT="${1:-.runtime/certs}"
DAYS=30
mkdir -p "$OUT"
cd "$OUT"

ec_key() { openssl ecparam -name prime256v1 -genkey -noout -out "$1" 2>/dev/null; }

ec_key ca.key
openssl req -x509 -new -key ca.key -sha256 -days "$DAYS" -subj "/CN=clasp-dev-ca" \
  -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -out ca.pem

issue() {  # issue <name> <cn> <extendedKeyUsage> [subjectAltName]
  ec_key "$1.key"
  openssl req -new -key "$1.key" -subj "/CN=$2" -out "$1.csr"
  {
    echo "basicConstraints=critical,CA:FALSE"
    echo "keyUsage=critical,digitalSignature"
    echo "extendedKeyUsage=$3"
    [ -n "${4:-}" ] && echo "subjectAltName=$4"
  } > "$1.ext"
  openssl x509 -req -in "$1.csr" -CA ca.pem -CAkey ca.key -CAcreateserial -days "$DAYS" \
    -sha256 -extfile "$1.ext" -out "$1.pem" 2>/dev/null
  rm -f "$1.csr" "$1.ext"
}

issue registry registry serverAuth,clientAuth "DNS:registry,DNS:localhost,IP:127.0.0.1"
issue client clasp-dev-client clientAuth
chmod 600 ./*.key
echo "wrote $(pwd): ca.pem registry.pem/.key client.pem/.key"
