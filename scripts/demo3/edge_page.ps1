# One edge client's page, in its own terminal on laptop B or C. Used by the
# 07_B_T*.ps1 and 08_C_T*.ps1 scripts; can also be run directly:
#
#   .\scripts\demo3\edge_page.ps1 -Client client-flask -AIp 192.168.43.10 -Port 8020
#
# The page talks to laptop A's cluster over mutual TLS with this client's
# certificate, signs its uploads with this client's key, is served over HTTPS
# on this laptop only (127.0.0.1) and asks its developer to log in. Each client
# has its own adapters folder, which also holds its privacy ledger.
param(
    [Parameter(Mandatory = $true)][string]$Client,
    [Parameter(Mandatory = $true)][string]$AIp,
    [int]$Port = 8020,
    [string]$RunId = "run-1",
    # Only when laptop A's address changed after the certificates were made:
    # -ServerHostname cluster checks A's certificate against its name "cluster"
    # instead of the address dialled, so the old certificates keep working.
    [string]$ServerHostname = ""
)
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Address $AIp
$certs = "edge-certs/$Client"
foreach ($file in @("ca.pem", "client.pem", "client.key")) {
    if (-not (Test-Path "$certs/$file")) { throw "$certs/$file is missing: unpack this client's bundle (05_BC_setup.ps1)." }
}
$short = $Client -replace '^client-', ''
$arguments = @("-m", "edge.webui",
    "--cluster-url", "https://${AIp}:8002",
    "--client-id", $Client,
    "--adapters-root", "services/edge/artifacts/demo-$short",
    "--corpus-root", "datasets/materialized",
    "--mtls-dir", $certs,
    "--run-id", $RunId,
    "--port", "$Port")
if ($ServerHostname) { $arguments += @("--server-hostname", $ServerHostname) }
$scheme = "http"
if ((Test-Path "$certs/page.pem") -and (Test-Path "$certs/page.key")) {
    $arguments += @("--tls-cert", "$certs/page.pem", "--tls-key", "$certs/page.key")
    $scheme = "https"
}
if (Test-Path "edge-users/$Client.json") {
    $arguments += @("--users", "edge-users/$Client.json")
} else {
    Write-Host "WARNING edge-users/$Client.json is missing: this page runs without a login"
}
New-Item -ItemType Directory -Force "services/edge/artifacts/demo-$short" | Out-Null
$Host.UI.RawUI.WindowTitle = "$Client  ${scheme}://127.0.0.1:$Port"
Write-Host "$Client -> open ${scheme}://127.0.0.1:$Port and log in as dev-$short   (ctrl-c to stop)"
python @arguments
