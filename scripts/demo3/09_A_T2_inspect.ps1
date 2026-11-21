# Laptop A, terminal 2, on the day: look at the cluster and the registry while
# the round runs. Every call is over mutual TLS with the operator's certificate
# (clasp-admin), except the two that show a refusal.
#
#   .\scripts\demo3\09_A_T2_inspect.ps1                 # the list of views
#   .\scripts\demo3\09_A_T2_inspect.ps1 clusters        # round, who has sent, what is served
#   .\scripts\demo3\09_A_T2_inspect.ps1 versions web    # registry versions of cluster-web
#   .\scripts\demo3\09_A_T2_inspect.ps1 decisions web   # every promote / reject, with its reason
#   .\scripts\demo3\09_A_T2_inspect.ps1 ledger          # each laptop's privacy spend
#   .\scripts\demo3\09_A_T2_inspect.ps1 access          # who called the registry (last 20)
#   .\scripts\demo3\09_A_T2_inspect.ps1 audit           # last lines of the three audit files
#   .\scripts\demo3\09_A_T2_inspect.ps1 nocert          # a caller with no certificate is refused
#   .\scripts\demo3\09_A_T2_inspect.ps1 edge            # an edge certificate gets 403 from the registry
param(
    [Parameter(Position = 0)][string]$View = "",
    [Parameter(Position = 1)][string]$Cluster = "web"
)
. (Join-Path $PSScriptRoot "_common.ps1")
$admin = ".runtime/certs/admin"
$get = "scripts/demo3/clasp_get.py"
$registry = "https://localhost:8004"
$clusterUrl = "https://localhost:8002"

switch ($View) {
    "clusters"  { & python $get "$clusterUrl/clusters" --certs $admin }
    "versions"  { & python $get "$registry/adapters/cluster-$Cluster/versions" --certs $admin }
    "decisions" { & python $get "$registry/adapters/cluster-$Cluster/promotions" --certs $admin }
    "ledger"    { & python $get "$registry/privacy/ledger" --certs $admin }
    "access"    { & python $get "$registry/access-log?limit=20" --certs $admin }
    "audit" {
        foreach ($name in @("registry", "cluster", "demo-ui")) {
            $file = ".runtime\audit\$name.jsonl"
            Write-Host "== $file"
            if (Test-Path $file) { Get-Content $file -Tail 6 } else { Write-Host "   (nothing written yet)" }
        }
    }
    "nocert" {
        Write-Host "A caller with no certificate: the handshake is refused, no HTTP answer at all."
        & python $get "$registry/healthz" --certs $admin --no-cert
    }
    "edge" {
        Write-Host "client-flask's own certificate, asking the registry: edges talk to the cluster only."
        & python $get "$registry/adapters" --certs ".runtime/certs/edges/client-flask"
    }
    default {
        Write-Host "views: clusters | versions <web|scientific> | decisions <web|scientific> | ledger | access | audit | nocert | edge"
    }
}
# a refusal shown on purpose is not a script failure
$global:LASTEXITCODE = 0
