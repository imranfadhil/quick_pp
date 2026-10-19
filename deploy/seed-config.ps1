<#
  seed-config.ps1 - create deploy\config.bat from config.example.bat,
  pre-filling app-level values from the project .env file.

  Usage:
    powershell -File seed-config.ps1 [-Force]

  Only app-level settings are copied from .env. Infrastructure settings
  (droplet name, region, size, SSH key) keep the template defaults, and
  anything not present in .env is left blank for you to fill in.
#>
[CmdletBinding()]
param(
  [switch]$Force
)

$ErrorActionPreference = 'Stop'

$deployDir = $PSScriptRoot
$envFile   = Join-Path $deployDir '..\.env'
$template  = Join-Path $deployDir 'config.example.bat'
$outFile   = Join-Path $deployDir 'config.bat'

if (-not (Test-Path -LiteralPath $template)) {
  Write-Host "[seed] config.example.bat not found at $template" -ForegroundColor Red
  exit 1
}

if ((Test-Path -LiteralPath $outFile) -and -not $Force) {
  Write-Host '[seed] config.bat already exists - leaving it untouched (use -Force to regenerate).'
  exit 0
}

# ---- Parse .env -----------------------------------------------------------
$dotenv = @{}
if (Test-Path -LiteralPath $envFile) {
  foreach ($line in Get-Content -LiteralPath $envFile) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    $i = $t.IndexOf('=')
    if ($i -lt 1) { continue }
    $k = $t.Substring(0, $i).Trim()
    $v = $t.Substring($i + 1).Trim()
    # Strip inline comments if unquoted
    if (-not ($v.StartsWith('"') -or $v.StartsWith("'"))) {
      $v = ($v -split '\s+#')[0].Trim()
    } else {
      $v = $v.Trim('"', "'")
    }
    $dotenv[$k] = $v
  }
} else {
  Write-Host "[seed] No .env found at $envFile - using template defaults."
}

# ---- Map config.bat keys to .env keys -------------------------------------
$direct = @(
  'POSTGRES_USER',
  'POSTGRES_DB',
  'POSTGRES_PASSWORD',
  'TAVILY_API_KEY',
  'WEB_PORT'
)

$values = @{}
foreach ($key in $direct) {
  if ($dotenv.ContainsKey($key)) {
    $val = $dotenv[$key]
    if ($val -ne '' -and $val -ne 'xxx' -and -not $val.StartsWith('change_this')) {
      $values[$key] = $val
    }
  }
}

# Derive DOMAIN from PUBLIC_URL if present
if ($dotenv.ContainsKey('PUBLIC_URL') -and $dotenv['PUBLIC_URL'] -ne '') {
  $domain = ($dotenv['PUBLIC_URL'] -replace '^https?://', '') -replace '/.*$', ''
  if ($domain) { $values['DOMAIN'] = $domain }
}

# ---- Render config.bat ----------------------------------------------------
$lines = New-Object System.Collections.Generic.List[string]
$filled = New-Object System.Collections.Generic.List[string]

foreach ($line in Get-Content -LiteralPath $template) {
  if ($line -match '^\s*set\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
    $key = $Matches[1]
    if ($values.ContainsKey($key)) {
      $lines.Add("set $key=$($values[$key])")
      $filled.Add($key)
      continue
    }
  }
  $lines.Add($line)
}

# Mark the file as generated, right after the template's `@echo off`.
$lines.Insert(1, 'REM AUTO-GENERATED from .env by seed-config.ps1 - regenerate with: seed-config.bat -Force')

Set-Content -LiteralPath $outFile -Value $lines -Encoding ASCII

Write-Host "[seed] Wrote $outFile"
if ($filled.Count -gt 0) {
  Write-Host "[seed] Pre-filled from .env: $($filled -join ', ')"
} else {
  Write-Host '[seed] No matching values found in .env - template defaults used.'
}
