# Laptops B and C, after the session: a fresh start for the next presentation.
# Stop the three edge pages first (ctrl-c). Removes each client's demo folder:
# the adapters trained in this session, the merged adapters it pulled, and its
# privacy ledger. Certificates, keys, users and the bundle's adapters are KEPT.
#
# Removing a ledger forgets what that client spent: only do this together with
# 90_A_reset.ps1 on laptop A, so that the next session really is a new run on a
# cluster that holds nothing from this one.
#
#   .\scripts\demo3\91_BC_reset.ps1          # says what it would remove
#   .\scripts\demo3\91_BC_reset.ps1 -Yes     # removes it
param([switch]$Yes)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
$targets = @(Get-ChildItem "services\edge\artifacts" -Directory -Filter "demo-*" -ErrorAction SilentlyContinue)
if ($targets.Count -eq 0) { Write-Host "nothing to remove"; return }
foreach ($target in $targets) {
    if ($Yes) { Remove-Item -Recurse -Force $target.FullName; Write-Host "removed $($target.FullName)" }
    else { Write-Host "would remove $($target.FullName)" }
}
if (-not $Yes) { Write-Host "Run again with -Yes to remove them. Kept: edge-certs, edge-users." }
