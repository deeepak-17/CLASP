# Laptop C, terminal 3, on the day: the page of client-scikit-learn on port 8022.
#
#   .\scripts\demo3\08_C_T3_client-scikit-learn.ps1 -AIp 192.168.43.10
#
# Open https://127.0.0.1:8022 on this laptop and log in as dev-scikit-learn.
param([Parameter(Mandatory = $true)][string]$AIp, [string]$ServerHostname = "")
& (Join-Path $PSScriptRoot "edge_page.ps1") -Client client-scikit-learn -AIp $AIp -Port 8022 -ServerHostname $ServerHostname
