# Shared by the demo scripts: go to the repo root and activate the .venv.
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
$activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw "No .venv in $RepoRoot. Create it first: python -m venv .venv"
}
. $activate

function Invoke-Checked {
    # Run a native command and stop the script if it fails.
    param([Parameter(Mandatory = $true)][string]$Exe, [string[]]$Arguments = @())
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') exited with $LASTEXITCODE" }
}

function Assert-Address {
    param([string]$AIp)
    if ($AIp -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
        throw "-AIp must be laptop A's IPv4 address on the hotspot (ipconfig on A), got '$AIp'"
    }
}
