# Laptop A, terminal 3, on the day (optional): the evaluation results dashboard
# on http://localhost:8005. Static pages built from services\evaluation\results;
# it is not part of the round. Needs Node.js.
#
#   .\scripts\demo3\10_A_T3_dashboard.ps1
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location (Join-Path $RepoRoot "services\evaluation\dashboard")
if (-not (Test-Path "node_modules")) { npm ci --no-audit --no-fund }
npm run build
if ($LASTEXITCODE -ne 0) { throw "the dashboard build failed" }
npx vite preview --host --port 8005
