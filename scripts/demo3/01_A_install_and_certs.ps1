# Laptop A, the day before, step 1 of 4.
# Installs the packages and writes the development CA, every certificate and the
# shared keys to .runtime\certs, with laptop A's hotspot address in them.
#
#   .\scripts\demo3\01_A_install_and_certs.ps1 -AIp 192.168.43.10
#
# Needs Git for Windows (Git Bash + openssl). Run it ONCE: making the
# certificates again replaces the CA, and every bundle already handed to laptops
# B and C stops working. Use -Force only if you will also remake the bundles (03).
param(
    [Parameter(Mandatory = $true)][string]$AIp,
    [switch]$Force
)
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Address $AIp

Write-Host "== installing packages (contracts first: the registry needs contracts 1.2)"
Invoke-Checked python @("-m", "pip", "install", "-q", "-e", "contracts", "-e", "security",
    "-e", "services/evaluation", "-e", "services/edge", "-e", "services/cluster",
    "-e", "services/registry")

$ca = Join-Path $RepoRoot ".runtime\certs\ca.pem"
if ((Test-Path $ca) -and -not $Force) {
    Write-Host "== certificates already exist in .runtime\certs: left as they are."
    Write-Host "   (-Force makes new ones; then remake and re-copy every edge bundle)"
} else {
    Write-Host "== making certificates and keys for $AIp"
    $code = "import sys; sys.path.insert(0, 'scripts'); import run_services as r; " +
            "r.generate_certs(r.DEFAULT_MTLS_DIR, sys.argv[1])"
    Invoke-Checked python @("-c", $code, $AIp)
}
# (the pieces are joined first: inside @( ) a comma binds tighter than +)
$check = "import pathlib, sys; d = pathlib.Path('.runtime/certs'); " +
    "need = ['ca.pem', 'registry/registry.pem', 'cluster/cluster.pem', 'cluster/upload_hmac.json', " +
    "'demo-ui/demo-ui.pem', 'admin/admin.pem', 'pages/pages.pem', 'secrets/eval_hmac.key', " +
    "'secrets/jwt.key', 'edges/client-flask/hmac.key', 'edges/client-numpy/page.pem']; " +
    "missing = [n for n in need if not (d / n).is_file()]; " +
    "print('   certificates and keys: ok' if not missing else '   MISSING: ' + str(missing)); " +
    "sys.exit(1 if missing else 0)"
Invoke-Checked python @("-c", $check)
Write-Host ""
Write-Host "Next: .\scripts\demo3\02_A_users.ps1"
