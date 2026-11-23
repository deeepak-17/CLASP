# Laptop A, the day before, step 2 of 4.
# Writes who may log in: the panel's operator and one developer per edge page.
# Each password is asked for twice at the prompt (at least 8 characters) and is
# never an argument. The files hold scrypt hashes only.
#
#   .\scripts\demo3\02_A_users.ps1
#
# Users written: "operator" for the panel; "dev-flask", "dev-requests",
# "dev-werkzeug", "dev-numpy", "dev-pandas", "dev-scikit-learn" for the edge pages.
. (Join-Path $PSScriptRoot "_common.ps1")

Write-Host "== the live panel's operator"
Invoke-Checked python @("scripts/make_users.py", "--out", ".runtime/users/panel.json",
    "--add", "operator:operator")

foreach ($repo in @("flask", "requests", "werkzeug", "numpy", "pandas", "scikit-learn")) {
    Write-Host "== the developer of client-$repo"
    Invoke-Checked python @("scripts/make_users.py", "--out", ".runtime/users/client-$repo.json",
        "--add", "dev-${repo}:developer:client-$repo")
}
Write-Host ""
Write-Host "Next: .\scripts\demo3\03_A_bundles.ps1 -AIp <A's address>"
