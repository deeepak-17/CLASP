# Laptop C, terminal 2, on the day: the page of client-pandas on port 8021.
#
#   .\scripts\demo3\08_C_T2_client-pandas.ps1 -AIp 192.168.43.10
#
# Open https://127.0.0.1:8021 on this laptop and log in as dev-pandas.
param([Parameter(Mandatory = $true)][string]$AIp, [string]$ServerHostname = "")
& (Join-Path $PSScriptRoot "edge_page.ps1") -Client client-pandas -AIp $AIp -Port 8021 -ServerHostname $ServerHostname
