# Laptop A, terminal 1, on the day. Leave it running for the whole session.
# Starts the registry (:8004) and the cluster (:8002) behind mutual TLS and the
# live panel (https://localhost:8010) with its login. Ctrl-C stops all three.
#
#   .\scripts\demo3\06_A_T1_services.ps1
#
# The cluster keeps its rounds in memory: do not restart this between the
# pre-staged round and the live round.
. (Join-Path $PSScriptRoot "_common.ps1")
if (-not (Test-Path ".runtime\certs\ca.pem")) { throw "Run 01_A_install_and_certs.ps1 first (the day before)." }
if (-not (Test-Path ".runtime\users\panel.json")) { throw "Run 02_A_users.ps1 first: without it the panel has no login." }
python "scripts/run_services.py" "--ui" "--registry-data" ".runtime/registry-demo"
