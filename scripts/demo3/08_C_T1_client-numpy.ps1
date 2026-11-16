# Laptop C, terminal 1, on the day: the page of client-numpy on port 8020.
#
#   .\scripts\demo3\08_C_T1_client-numpy.ps1 -AIp 192.168.43.10
#
# Open https://127.0.0.1:8020 on this laptop and log in as dev-numpy.
param([Parameter(Mandatory = $true)][string]$AIp, [string]$ServerHostname = "")
& (Join-Path $PSScriptRoot "edge_page.ps1") -Client client-numpy -AIp $AIp -Port 8020 -ServerHostname $ServerHostname
