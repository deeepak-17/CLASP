# Laptop B, terminal 2, on the day: the page of client-requests on port 8021.
#
#   .\scripts\demo3\07_B_T2_client-requests.ps1 -AIp 192.168.43.10
#
# Open https://127.0.0.1:8021 on this laptop and log in as dev-requests.
param([Parameter(Mandatory = $true)][string]$AIp, [string]$ServerHostname = "")
& (Join-Path $PSScriptRoot "edge_page.ps1") -Client client-requests -AIp $AIp -Port 8021 -ServerHostname $ServerHostname
