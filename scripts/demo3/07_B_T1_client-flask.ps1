# Laptop B, terminal 1, on the day: the page of client-flask on port 8020.
#
#   .\scripts\demo3\07_B_T1_client-flask.ps1 -AIp 192.168.43.10
#
# Open https://127.0.0.1:8020 on this laptop and log in as dev-flask.
param([Parameter(Mandatory = $true)][string]$AIp, [string]$ServerHostname = "")
& (Join-Path $PSScriptRoot "edge_page.ps1") -Client client-flask -AIp $AIp -Port 8020 -ServerHostname $ServerHostname
