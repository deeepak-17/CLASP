# Laptop A, the day before, step 3 of 4.
# One zip per edge client in .runtime\edge-bundles: its adapters and corpus
# (gitignored, so git does not carry them), its certificate, upload key, page
# certificate and users file. Copy the three web zips to laptop B and the three
# scientific zips to laptop C by hand (USB stick): each holds private keys.
#
#   .\scripts\demo3\03_A_bundles.ps1 -AIp 192.168.43.10
#
# Laptop A must have, for every client, a trained adapter under
# services\edge\artifacts\{round2_d3,dp_smoke,round1}\<client>\adapter and its
# corpus under datasets\materialized\<cluster>\<client>.
param([Parameter(Mandatory = $true)][string]$AIp)
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Address $AIp
if (-not (Test-Path ".runtime\certs\ca.pem")) { throw "Run 01_A_install_and_certs.ps1 first." }
if (-not (Test-Path ".runtime\users\client-flask.json")) { throw "Run 02_A_users.ps1 first." }

$url = "https://${AIp}:8002"
Write-Host "== web bundles (laptop B)"
Invoke-Checked python @("scripts/make_edge_bundle.py", "client-flask", "client-requests",
    "client-werkzeug", "--cluster-url", $url, "--with-corpus")
Write-Host "== scientific bundles (laptop C)"
Invoke-Checked python @("scripts/make_edge_bundle.py", "client-numpy", "client-pandas",
    "client-scikit-learn", "--cluster-url", $url, "--with-corpus")
Write-Host ""
Get-ChildItem ".runtime\edge-bundles\edge-client-*.zip" | ForEach-Object { "   $($_.Name)  $([math]::Round($_.Length / 1MB, 1)) MB" }
Write-Host ""
Write-Host "Copy edge-client-{flask,requests,werkzeug}.zip to laptop B and"
Write-Host "edge-client-{numpy,pandas,scikit-learn}.zip to laptop C."
Write-Host "Next (PowerShell as administrator): .\scripts\demo3\04_A_firewall.ps1"
