# Laptop A, after the session: a fresh start for the next presentation.
# Stop terminal 1 first (ctrl-c). Removes the registry's data (every version and
# decision), the evaluation results of this session and the audit files.
# Certificates, keys and users are KEPT, so the edge bundles stay valid.
#
#   .\scripts\demo3\90_A_reset.ps1          # says what it would remove
#   .\scripts\demo3\90_A_reset.ps1 -Yes     # removes it
param([switch]$Yes)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
$targets = @(".runtime\registry-demo", ".runtime\live-eval", ".runtime\audit") | Where-Object { Test-Path $_ }
if ($targets.Count -eq 0) { Write-Host "nothing to remove"; return }
foreach ($target in $targets) {
    if ($Yes) { Remove-Item -Recurse -Force $target; Write-Host "removed $target" }
    else { Write-Host "would remove $target" }
}
if (-not $Yes) { Write-Host "Run again with -Yes to remove them. Kept: .runtime\certs, .runtime\users." }
