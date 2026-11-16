# Laptop A, the day before, step 4 of 4. Run in PowerShell AS ADMINISTRATOR.
# Lets laptops B and C reach the cluster (8002), and a browser on the hotspot
# reach the panel (8010) and the dashboard (8005). The registry (8004) is only
# called from laptop A itself, so it is not opened.
#
#   .\scripts\demo3\04_A_firewall.ps1            # add the rule
#   .\scripts\demo3\04_A_firewall.ps1 -Remove    # after the demo
param([switch]$Remove)
$ErrorActionPreference = "Stop"
$name = "CLASP demo"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Open PowerShell as administrator and run this again." }
Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if ($Remove) { Write-Host "removed the '$name' rule"; return }
New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP `
    -LocalPort 8002, 8010, 8005 -Action Allow -Profile Any | Out-Null
Write-Host "added the '$name' rule for TCP 8002, 8010, 8005"
Write-Host "Also mark the hotspot as a Private network on this laptop."
