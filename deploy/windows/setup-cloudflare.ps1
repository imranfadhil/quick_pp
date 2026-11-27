<#
  setup-cloudflare.ps1 — Automate Cloudflare Tunnel routes, DNS CNAMEs,
  Zero Trust Access policy, and WAF Rate Limiting from local config.bat files.

  Reads Account ID, Tunnel IDs, DOMAIN, and ALLOWED_EMAIL directly from:
    - deploy\windows\config.bat       (gitignored private app settings)
    - deploy\windows\demo\config.bat  (gitignored demo app settings)

  Usage:
    powershell -ExecutionPolicy Bypass -File deploy\windows\setup-cloudflare.ps1 [-AllowedEmail you@example.com]
#>
[CmdletBinding()]
param(
  [string]$ZoneName = '',
  [string]$PrivateHostname = '',
  [string]$PrivateService = 'http://web:80',
  [string]$DemoHostname = '',
  [string]$DemoService = 'http://qpp-demo:7860',
  [string]$AllowedEmail = ''
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-BatVar {
  param([string]$ConfigPath, [string]$VarName)
  if (-not (Test-Path -LiteralPath $ConfigPath)) { return '' }
  $raw = Get-Content -LiteralPath $ConfigPath -Raw
  $m = [regex]::Match($raw, "(?m)^\s*set\s+`"?$([regex]::Escape($VarName))=([^`"\r\n]*)")
  if ($m.Success) { return $m.Groups[1].Value.Trim() }
  return ''
}

function Get-TunnelInfoFromConfig {
  param([string]$ConfigPath, [string]$Label)
  if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
  }
  $tok = Get-BatVar -ConfigPath $ConfigPath -VarName 'CF_TUNNEL_TOKEN'
  if ([string]::IsNullOrWhiteSpace($tok)) {
    throw "CF_TUNNEL_TOKEN is empty in $ConfigPath. Paste your $Label tunnel token first."
  }
  $padLen = (4 - ($tok.Length % 4)) % 4
  $padded = $tok + ('=' * $padLen)
  $jsonBytes = [Convert]::FromBase64String($padded)
  $jsonText = [System.Text.Encoding]::UTF8.GetString($jsonBytes)
  $parsed = $jsonText | ConvertFrom-Json
  if (-not $parsed.a -or -not $parsed.t) {
    throw "CF_TUNNEL_TOKEN in $ConfigPath is not a valid Cloudflare Tunnel token."
  }
  return @{
    AccountId = [string]$parsed.a
    TunnelId  = [string]$parsed.t
    Domain    = (Get-BatVar -ConfigPath $ConfigPath -VarName 'DOMAIN')
    Email     = (Get-BatVar -ConfigPath $ConfigPath -VarName 'ALLOWED_EMAIL')
  }
}

$deployDir   = $PSScriptRoot
$privateInfo = Get-TunnelInfoFromConfig -ConfigPath (Join-Path $deployDir 'config.bat') -Label 'private'
$demoInfo    = Get-TunnelInfoFromConfig -ConfigPath (Join-Path $deployDir 'demo\config.bat') -Label 'demo'

if ([string]::IsNullOrWhiteSpace($PrivateHostname)) { $PrivateHostname = $privateInfo.Domain }
if ([string]::IsNullOrWhiteSpace($DemoHostname))    { $DemoHostname    = $demoInfo.Domain }
if ([string]::IsNullOrWhiteSpace($AllowedEmail))    { $AllowedEmail    = $privateInfo.Email }
if ([string]::IsNullOrWhiteSpace($PrivateHostname)) { throw 'DOMAIN is empty in deploy\windows\config.bat.' }
if ([string]::IsNullOrWhiteSpace($DemoHostname))    { throw 'DOMAIN is empty in deploy\windows\demo\config.bat.' }
if ([string]::IsNullOrWhiteSpace($ZoneName)) {
  $parts = $PrivateHostname.Split('.')
  if ($parts.Length -ge 2) {
    $ZoneName = ($parts[($parts.Length - 2)..($parts.Length - 1)] -join '.')
  } else {
    throw 'Could not infer root ZoneName from DOMAIN; pass -ZoneName explicitly.'
  }
}
if ([string]::IsNullOrWhiteSpace($AllowedEmail)) {
  $AllowedEmail = (Read-Host -Prompt "Enter email address to allow in Zero Trust Access for $PrivateHostname").Trim()
}
if ([string]::IsNullOrWhiteSpace($AllowedEmail)) {
  throw 'AllowedEmail is required for Cloudflare Zero Trust Access policy.'
}

if ($privateInfo.AccountId -ne $demoInfo.AccountId) {
  throw "Account ID mismatch between deploy\windows\config.bat ($($privateInfo.AccountId)) and deploy\windows\demo\config.bat ($($demoInfo.AccountId))."
}

$accountId       = $privateInfo.AccountId
$privateTunnelId = $privateInfo.TunnelId
$demoTunnelId    = $demoInfo.TunnelId

Write-Host ''
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host "   CLOUDFLARE PROVISIONING - $ZoneName" -ForegroundColor Cyan
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host "  Account ID     : $accountId"
Write-Host "  Private Tunnel : $privateTunnelId -> $PrivateHostname ($PrivateService)"
Write-Host "  Demo Tunnel    : $demoTunnelId -> $DemoHostname ($DemoService)"
Write-Host "  Access Policy  : Allow $AllowedEmail on $PrivateHostname"
Write-Host ''

# Resolve Cloudflare API / OAuth Token (env var -> cf CLI login -> interactive prompt)
$apiToken = $env:CLOUDFLARE_API_TOKEN
if ([string]::IsNullOrWhiteSpace($apiToken)) {
  $apiToken = $env:CF_API_TOKEN
}
if ([string]::IsNullOrWhiteSpace($apiToken)) {
  $cfConfigPaths = @(
    (Join-Path $env:APPDATA 'xdg.config\cloudflare\config\default.json'),
    (Join-Path $env:USERPROFILE '.config\cloudflare\config\default.json')
  )
  foreach ($cfCfg in $cfConfigPaths) {
    if ($cfCfg -and (Test-Path -LiteralPath $cfCfg)) {
      try {
        $cfgObj = Get-Content -LiteralPath $cfCfg -Raw | ConvertFrom-Json
        if ($cfgObj.oauth_token) {
          $apiToken = [string]$cfgObj.oauth_token
          Write-Host '  [Auth] Using authenticated session from Cloudflare CLI (cf auth login).' -ForegroundColor DarkGray
          break
        }
      } catch {}
    }
  }
}
if ([string]::IsNullOrWhiteSpace($apiToken)) {
  Write-Host 'Create a Cloudflare API Token at: https://dash.cloudflare.com/profile/api-tokens' -ForegroundColor Yellow
  Write-Host 'Required permissions:' -ForegroundColor Yellow
  Write-Host '  - Account | Cloudflare Tunnel (or Cloudflare One Connectors) | Edit'
  Write-Host '  - Account | Access: Apps and Policies                        | Edit'
  Write-Host '  - Account | Access: Organizations, Identity Providers, Groups| Edit'
  Write-Host '  - Zone    | DNS                                              | Edit'
  Write-Host '  - Zone    | Zone WAF                                         | Edit'
  Write-Host '  - Zone    | Zone Settings                                    | Edit'
  Write-Host '  - Zone    | Zone                                             | Read'
  Write-Host ''
  $sec = Read-Host -Prompt 'Paste Cloudflare API Token (input hidden)' -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
  try {
    $apiToken = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr).Trim()
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
}

if ([string]::IsNullOrWhiteSpace($apiToken)) {
  throw 'No Cloudflare API Token provided.'
}

$headers = @{
  'Authorization' = "Bearer $apiToken"
  'Content-Type'  = 'application/json'
}

function Invoke-CfApi {
  param(
    [Parameter(Mandatory)][string]$Method,
    [Parameter(Mandatory)][string]$Path,
    [object]$Body = $null,
    [switch]$Allow404
  )
  $uri = "https://api.cloudflare.com/client/v4$Path"
  $params = @{
    Method      = $Method
    Uri         = $uri
    Headers     = $headers
    ErrorAction = 'Stop'
  }
  if ($null -ne $Body) {
    $params['Body'] = ($Body | ConvertTo-Json -Depth 20 -Compress)
  }
  try {
    $resp = Invoke-RestMethod @params
    if (-not $resp.success) {
      $errMsg = ($resp.errors | ForEach-Object { "[$($_.code)] $($_.message)" }) -join '; '
      throw "Cloudflare API error on $Method $Path : $errMsg"
    }
    return $resp
  } catch {
    $statusCode = $null
    if ($_.Exception.Response) {
      $statusCode = [int]$_.Exception.Response.StatusCode
    }
    if ($Allow404 -and $statusCode -eq 404) {
      return $null
    }
    $detail = $_.ErrorDetails.Message
    if ($detail) {
      throw "HTTP $statusCode on $Method $Path : $detail"
    }
    throw
  }
}

# ---- 1. Resolve Zone & Ensure SSL=Full -------------------------------------
Write-Host "[1/5] Looking up zone '$ZoneName'..."
$zonesResp = Invoke-CfApi -Method GET -Path "/zones?name=$([uri]::EscapeDataString($ZoneName))"
if (-not $zonesResp.result -or $zonesResp.result.Count -eq 0) {
  throw "Zone '$ZoneName' not found in Cloudflare account $accountId."
}
$zone = $zonesResp.result[0]
$zoneId = $zone.id
$planId = [string]$zone.plan.legacy_id
$isFreePlan = ($planId -eq 'free' -or $zone.plan.name -match 'Free')
Write-Host "  [OK] Zone ID: $zoneId (Plan: $($zone.plan.name), Status: $($zone.status))" -ForegroundColor Green

try {
  $sslResp = Invoke-CfApi -Method GET -Path "/zones/$zoneId/settings/ssl"
  if ($sslResp.result.value -eq 'flexible' -or $sslResp.result.value -eq 'off') {
    [void](Invoke-CfApi -Method PATCH -Path "/zones/$zoneId/settings/ssl" -Body @{ value = 'full' })
    Write-Host '  [OK] Updated Zone SSL/TLS encryption mode to Full.' -ForegroundColor Green
  }
} catch {
  Write-Host "  [SKIP] Could not inspect/update SSL setting (non-fatal): $($_.Exception.Message)" -ForegroundColor DarkYellow
}

# ---- 2. Configure Both Tunnel Ingress Routes -------------------------------
Write-Host '[2/5] Configuring Cloudflare Tunnel ingress routes...'

$privateTunnelBody = @{
  config = @{
    ingress = @(
      @{ hostname = $PrivateHostname; service = $PrivateService },
      @{ service = 'http_status:404' }
    )
  }
}
[void](Invoke-CfApi -Method PUT -Path "/accounts/$accountId/cfd_tunnel/$privateTunnelId/configurations" -Body $privateTunnelBody)
Write-Host "  [OK] Private tunnel ($privateTunnelId) -> $PrivateHostname -> $PrivateService" -ForegroundColor Green

$demoTunnelBody = @{
  config = @{
    ingress = @(
      @{ hostname = $DemoHostname; service = $DemoService },
      @{ service = 'http_status:404' }
    )
  }
}
[void](Invoke-CfApi -Method PUT -Path "/accounts/$accountId/cfd_tunnel/$demoTunnelId/configurations" -Body $demoTunnelBody)
Write-Host "  [OK] Demo tunnel ($demoTunnelId) -> $DemoHostname -> $DemoService" -ForegroundColor Green

# ---- 3. Upsert Proxied CNAME DNS Records -----------------------------------
Write-Host '[3/5] Upserting proxied CNAME DNS records...'

function Set-TunnelCname {
  param([string]$Hostname, [string]$TunnelId)
  $target = "$TunnelId.cfargotunnel.com"
  $existing = Invoke-CfApi -Method GET -Path "/zones/$zoneId/dns_records?name=$([uri]::EscapeDataString($Hostname))"
  $recordBody = @{
    type    = 'CNAME'
    name    = $Hostname
    content = $target
    proxied = $true
    ttl     = 1
    comment = "Managed by quick_pp Cloudflare Tunnel ($TunnelId)"
  }
  if ($existing.result -and $existing.result.Count -gt 0) {
    $recId = $existing.result[0].id
    [void](Invoke-CfApi -Method PUT -Path "/zones/$zoneId/dns_records/$recId" -Body $recordBody)
    Write-Host "  [OK] Updated DNS CNAME: $Hostname -> $target (Proxied)" -ForegroundColor Green
  } else {
    [void](Invoke-CfApi -Method POST -Path "/zones/$zoneId/dns_records" -Body $recordBody)
    Write-Host "  [OK] Created DNS CNAME: $Hostname -> $target (Proxied)" -ForegroundColor Green
  }
}

Set-TunnelCname -Hostname $PrivateHostname -TunnelId $privateTunnelId
Set-TunnelCname -Hostname $DemoHostname    -TunnelId $demoTunnelId

# ---- 4. Zero Trust Access Policy & Self-Hosted App -------------------------
Write-Host "[4/5] Configuring Cloudflare Zero Trust Access for $PrivateHostname..."

$orgResp = Invoke-CfApi -Method GET -Path "/accounts/$accountId/access/organizations" -Allow404
if (-not $orgResp -or -not $orgResp.result -or -not $orgResp.result.auth_domain) {
  throw "Zero Trust Organization is not initialized yet. Open https://one.dash.cloudflare.com/ once to pick your team name (Free plan), then re-run this script."
}
Write-Host "  [OK] Zero Trust Team Domain: $($orgResp.result.auth_domain)" -ForegroundColor Green

# Ensure One-Time PIN IdP exists
$idpsResp = Invoke-CfApi -Method GET -Path "/accounts/$accountId/access/identity_providers"
$otpIdp = $idpsResp.result | Where-Object { $_.type -eq 'onetimepin' } | Select-Object -First 1
if (-not $otpIdp) {
  $otpCreated = Invoke-CfApi -Method POST -Path "/accounts/$accountId/access/identity_providers" -Body @{
    name   = 'One-time PIN'
    type   = 'onetimepin'
    config = @{}
  }
  $otpIdp = $otpCreated.result
  Write-Host '  [OK] Enabled One-time PIN Identity Provider.' -ForegroundColor Green
} else {
  Write-Host '  [OK] One-time PIN Identity Provider already active.' -ForegroundColor Green
}

# Upsert Reusable Access Policy (cloudflare-one best practice)
$policyName = 'Allow Owner Only (quick_pp)'
$policiesResp = Invoke-CfApi -Method GET -Path "/accounts/$accountId/access/policies"
$existingPolicy = $policiesResp.result | Where-Object { $_.name -eq $policyName } | Select-Object -First 1
$policyBody = @{
  name             = $policyName
  decision         = 'allow'
  include          = @(
    @{ email = @{ email = $AllowedEmail } }
  )
  session_duration = '168h'
}
if ($existingPolicy) {
  $polUpdated = Invoke-CfApi -Method PUT -Path "/accounts/$accountId/access/policies/$($existingPolicy.id)" -Body $policyBody
  $policyId = $polUpdated.result.id
  Write-Host "  [OK] Updated reusable Access policy '$policyName' ($policyId) -> $AllowedEmail" -ForegroundColor Green
} else {
  $polCreated = Invoke-CfApi -Method POST -Path "/accounts/$accountId/access/policies" -Body $policyBody
  $policyId = $polCreated.result.id
  Write-Host "  [OK] Created reusable Access policy '$policyName' ($policyId) -> $AllowedEmail" -ForegroundColor Green
}

# Upsert Self-Hosted Access Application for private.quick-pp.com
$appsResp = Invoke-CfApi -Method GET -Path "/accounts/$accountId/access/apps"
$existingApp = $appsResp.result | Where-Object { $_.domain -eq $PrivateHostname } | Select-Object -First 1
$appBody = @{
  name                       = 'quick_pp Private'
  domain                     = $PrivateHostname
  type                       = 'self_hosted'
  session_duration           = '168h'
  app_launcher_visible       = $true
  http_only_cookie_attribute = $true
  same_site_cookie_attribute = 'lax'
  allowed_idps               = @([string]$otpIdp.id)
  auto_redirect_to_identity  = $true
  policies                   = @(
    @{ id = $policyId; precedence = 1 }
  )
}
if ($existingApp) {
  [void](Invoke-CfApi -Method PUT -Path "/accounts/$accountId/access/apps/$($existingApp.id)" -Body $appBody)
  Write-Host "  [OK] Updated Access Application 'quick_pp Private' ($PrivateHostname)" -ForegroundColor Green
} else {
  [void](Invoke-CfApi -Method POST -Path "/accounts/$accountId/access/apps" -Body $appBody)
  Write-Host "  [OK] Created Access Application 'quick_pp Private' ($PrivateHostname)" -ForegroundColor Green
}

# ---- 5. WAF Rate Limiting Rule for demo.quick-pp.com -----------------------
Write-Host "[5/5] Configuring WAF Rate Limiting rule on $DemoHostname..."

$rlPeriod     = if ($isFreePlan) { 10 } else { 60 }
$rlRequests   = if ($isFreePlan) { 20 } else { 60 }
$rlMitigation = if ($isFreePlan) { 10 } else { 60 }

$demoRuleDesc = 'quick_pp Demo Rate Limit'
$demoRule = @{
  description = $demoRuleDesc
  expression  = "(http.host eq `"$DemoHostname`" and not starts_with(http.request.uri.path, `"/_app/`"))"
  action      = 'block'
  enabled     = $true
  ratelimit   = @{
    characteristics     = @('cf.colo.id', 'ip.src')
    period              = $rlPeriod
    requests_per_period = $rlRequests
    mitigation_timeout  = $rlMitigation
  }
}

$existingRuleset = Invoke-CfApi -Method GET -Path "/zones/$zoneId/rulesets/phases/http_ratelimit/entrypoint" -Allow404
$rulesToPut = @()
if ($existingRuleset -and $existingRuleset.result -and $existingRuleset.result.rules -and -not $isFreePlan) {
  foreach ($r in $existingRuleset.result.rules) {
    if ($r.description -ne $demoRuleDesc) {
      $rulesToPut += $r
    }
  }
}
$rulesToPut += $demoRule

[void](Invoke-CfApi -Method PUT -Path "/zones/$zoneId/rulesets/phases/http_ratelimit/entrypoint" -Body @{
  rules = $rulesToPut
})
Write-Host "  [OK] Active WAF Rate Limit: $rlRequests req / ${rlPeriod}s (block ${rlMitigation}s) on $DemoHostname (+ 60 req/min sustained in Nginx)" -ForegroundColor Green

Write-Host ''
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host '   CLOUDFLARE CONFIGURATION COMPLETE!' -ForegroundColor Cyan
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host "  Private App : https://$PrivateHostname  (Zero Trust: $AllowedEmail)"
Write-Host "  Public Demo : https://$DemoHostname     (2 GB cap + WAF & Nginx rate limits)"
Write-Host ''
