#!/usr/bin/env bash
# Throwaway CA + server certs (registry, cluster) + client certs, for running the
# stack behind mTLS locally or across laptops on one LAN (docker-compose.mtls.yml,
# docs/multi-laptop-demo.md). DEV ONLY — production keys come from the security
# library's CA (P3) and never live in the repo.
#
#   scripts/make_dev_certs.sh [out_dir]      # default: .runtime/certs (gitignored)
#
#   CLASP_LAN_IPS="192.168.43.10"            laptop A's LAN address(es), comma-separated,
#                                            added to the registry and cluster SANs so
#                                            edges can reach them by IP with hostname
#                                            checking on
#   CLASP_EDGE_CLIENTS="client-flask client-numpy"
#                                            one client cert per edge; the CN is the
#                                            client_id, which the cluster binds uploads to
set -euo pipefail

OUT="${1:-.runtime/certs}"
DAYS=30
LAN_IPS="${CLASP_LAN_IPS:-}"
EDGE_CLIENTS="${CLASP_EDGE_CLIENTS:-client-flask client-requests client-werkzeug client-numpy client-pandas client-scikit-learn}"
umask 077                      # keys are never group/world readable, even briefly
mkdir -p "$OUT/ca-private"
cd "$OUT"

ec_key() { openssl ecparam -name prime256v1 -genkey -noout -out "$1" 2>/dev/null; }

# Git Bash on Windows rewrites a leading "/" in arguments into a Windows path,
# which turns "/CN=x" into "C:/Program Files/Git/CN=x". Keep subjects literal.
export MSYS_NO_PATHCONV=1

ec_key ca-private/ca.key
openssl req -x509 -new -key ca-private/ca.key -sha256 -days "$DAYS" -subj "/CN=clasp-dev-ca" \
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
  openssl x509 -req -in "$1.csr" -CA ca.pem -CAkey ca-private/ca.key -CAcreateserial -CAserial ca-private/ca.srl -days "$DAYS" \
    -sha256 -extfile "$1.ext" -out "$1.pem" 2>/dev/null
  rm -f "$1.csr" "$1.ext"
}

lan_sans=""
IFS=',' read -ra ips <<< "$LAN_IPS"
for ip in "${ips[@]}"; do
  ip="$(echo "$ip" | tr -d '[:space:]')"
  [ -n "$ip" ] && lan_sans="$lan_sans,IP:$ip"
done

# Servers also get clientAuth: the cluster presents its cert to the registry
# (seam B), and each service's healthcheck presents its own.
issue registry registry serverAuth,clientAuth "DNS:registry,DNS:localhost,IP:127.0.0.1$lan_sans"
issue cluster cluster serverAuth,clientAuth "DNS:cluster,DNS:localhost,IP:127.0.0.1$lan_sans"
issue client clasp-dev-client clientAuth
issue demo-ui demo-ui clientAuth            # the panel's calls to cluster + registry
mkdir -p registry cluster demo-ui
cp ca.pem registry.pem registry.key registry/
cp ca.pem cluster.pem cluster.key cluster/
cp ca.pem demo-ui.pem demo-ui.key demo-ui/

# One directory per edge laptop: copy edges/<client_id>/ onto that laptop.
for cid in $EDGE_CLIENTS; do
  mkdir -p "edges/$cid"
  issue "edges/$cid/client" "$cid" clientAuth
  cp ca.pem "edges/$cid/ca.pem"
done

echo "wrote $(pwd):"
echo "  registry/  cluster/  demo-ui/   mounted into the containers on laptop A"
echo "  edges/<client_id>/              ca.pem client.pem client.key — copy to that edge laptop"
echo "  SANs: registry, cluster, localhost, 127.0.0.1${lan_sans//,/ }"
echo "ca-private/ca.key stays on this machine — no container or laptop ever needs the CA key"
