# Laptops B and C, the day before.
# Installs the packages, unpacks this laptop's three bundles into the repo, and
# checks that laptop A's cluster answers this laptop's certificate.
#
#   .\scripts\demo3\05_BC_setup.ps1 -AIp 192.168.43.10 -Bundles D:\      # folder with the zips
#
# Before it: git checkout integration/demo; git pull; python -m venv .venv;
# and a CUDA build of torch:  pip install torch --index-url https://download.pytorch.org/whl/cu126
# The health check at the end needs laptop A's services running (06 on A); if A
# is not up yet, run this again later with -CheckOnly.
param(
    [Parameter(Mandatory = $true)][string]$AIp,
    [string]$Bundles = ".",
    [switch]$CheckOnly
)
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Address $AIp

if (-not $CheckOnly) {
    Write-Host "== installing packages"
    Invoke-Checked python @("-m", "pip", "install", "-q", "-e", "contracts", "-e", "security",
        "-e", "services/evaluation", "-e", "services/edge")
    $zips = @(Get-ChildItem (Join-Path $Bundles "edge-client-*.zip") -ErrorAction SilentlyContinue)
    if ($zips.Count -eq 0) { throw "No edge-client-*.zip in '$Bundles' (copy this laptop's three bundles from laptop A)." }
    foreach ($zip in $zips) {
        Write-Host "== unpacking $($zip.Name)"
        Expand-Archive -Path $zip.FullName -DestinationPath $RepoRoot -Force
    }
    Write-Host "== the base model (several GB the first time)"
    $download = "from huggingface_hub import snapshot_download; " +
        "print('   ' + snapshot_download('deepseek-ai/deepseek-coder-1.3b-base'))"
    Invoke-Checked python @("-c", $download)
}

$clients = @(Get-ChildItem "edge-certs" -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
if ($clients.Count -eq 0) { throw "No edge-certs\<client> folders: the bundles were not unpacked here." }
Write-Host "== this laptop's clients: $($clients -join ', ')"
foreach ($client in $clients) {
    foreach ($file in @("ca.pem", "client.pem", "client.key", "hmac.key", "page.pem", "page.key")) {
        if (-not (Test-Path "edge-certs\$client\$file")) { throw "edge-certs\$client\$file is missing: remake the bundle on laptop A." }
    }
    if (-not (Test-Path "edge-users\$client.json")) { Write-Host "   WARNING edge-users\$client.json is missing: that page will run without a login" }
}
Write-Host "== does laptop A's cluster answer? (https://${AIp}:8002/healthz)"
& python "scripts/demo3/clasp_get.py" "https://${AIp}:8002/healthz" "--certs" "edge-certs/$($clients[0])"
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Not reachable. Is laptop A running 06_A_T1_services.ps1? Same hotspot, marked Private?"
    Write-Host "   Firewall rule on A (04)? If A's address changed since the certificates were made,"
    Write-Host "   see 'If something refuses' in the plan."
} else {
    Write-Host "   ok. On the day: one terminal per client, 07_B_T*.ps1 or 08_C_T*.ps1"
}
