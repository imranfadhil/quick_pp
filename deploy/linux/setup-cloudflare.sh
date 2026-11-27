#!/usr/bin/env bash
# ===================================================================
#  setup-cloudflare.sh - Automate Cloudflare Tunnel routes, DNS CNAMEs,
#  Zero Trust Access policy, and WAF Rate Limiting from local config.sh
#  (or config.bat) files on Linux / Ubuntu.
#
#  Reads Account ID, Tunnel IDs, DOMAIN, and ALLOWED_EMAIL directly from:
#    - deploy/linux/config.sh       (gitignored private app settings)
#    - deploy/linux/demo/config.sh  (gitignored demo app settings)
#
#  Usage:
#    ./setup-cloudflare.sh [--allowed-email you@example.com] [--zone-name example.com]
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$DEPLOY_DIR" "$@" <<'PY'
import argparse
import base64
import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

deploy_dir = sys.argv[1]
parser = argparse.ArgumentParser(description="Provision Cloudflare Tunnels, DNS, Zero Trust Access & WAF rules.")
parser.add_argument("--zone-name", "-ZoneName", default="")
parser.add_argument("--private-hostname", "-PrivateHostname", default="")
parser.add_argument("--private-service", "-PrivateService", default="http://web:80")
parser.add_argument("--demo-hostname", "-DemoHostname", default="")
parser.add_argument("--demo-service", "-DemoService", default="http://qpp-demo:7860")
parser.add_argument("--allowed-email", "-AllowedEmail", default="")
args = parser.parse_args(sys.argv[2:])


def parse_config_var(sh_path: str, bat_path: str, var_name: str) -> str:
    if os.path.isfile(sh_path):
        with open(sh_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                t = raw.strip()
                if not t or t.startswith("#"):
                    continue
                m = re.match(rf"^(?:export\s+)?{re.escape(var_name)}=(.*)$", t)
                if m:
                    val = m.group(1).strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    return val.strip()
    if os.path.isfile(bat_path):
        with open(bat_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                m = re.match(rf'^\s*set\s+"?{re.escape(var_name)}=([^"\r\n]*)', raw)
                if m:
                    return m.group(1).strip()
    return ""


def get_tunnel_info(sh_path: str, bat_path: str, label: str) -> dict[str, str]:
    if not os.path.isfile(sh_path) and not os.path.isfile(bat_path):
        raise SystemExit(f"[ERROR] Config file not found: {sh_path}")
    tok = parse_config_var(sh_path, bat_path, "CF_TUNNEL_TOKEN")
    if not tok:
        raise SystemExit(f"[ERROR] CF_TUNNEL_TOKEN is empty in {sh_path}. Paste your {label} tunnel token first.")
    padded = tok + ("=" * ((4 - len(tok) % 4) % 4))
    try:
        parsed = json.loads(base64.b64decode(padded).decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"[ERROR] CF_TUNNEL_TOKEN in {sh_path} could not be decoded: {exc}") from exc
    if not parsed.get("a") or not parsed.get("t"):
        raise SystemExit(f"[ERROR] CF_TUNNEL_TOKEN in {sh_path} is not a valid Cloudflare Tunnel token.")
    return {
        "account_id": str(parsed["a"]),
        "tunnel_id": str(parsed["t"]),
        "domain": parse_config_var(sh_path, bat_path, "DOMAIN"),
        "email": parse_config_var(sh_path, bat_path, "ALLOWED_EMAIL"),
    }


private_info = get_tunnel_info(
    os.path.join(deploy_dir, "config.sh"),
    os.path.join(deploy_dir, "..", "windows", "config.bat"),
    "private",
)
demo_info = get_tunnel_info(
    os.path.join(deploy_dir, "demo", "config.sh"),
    os.path.join(deploy_dir, "..", "windows", "demo", "config.bat"),
    "demo",
)

private_hostname = args.private_hostname or private_info["domain"]
demo_hostname = args.demo_hostname or demo_info["domain"]
allowed_email = args.allowed_email or private_info["email"]
zone_name = args.zone_name

if not private_hostname:
    raise SystemExit("[ERROR] DOMAIN is empty in deploy/linux/config.sh.")
if not demo_hostname:
    raise SystemExit("[ERROR] DOMAIN is empty in deploy/linux/demo/config.sh.")
if not zone_name:
    parts = private_hostname.split(".")
    if len(parts) >= 2:
        zone_name = ".".join(parts[-2:])
    else:
        raise SystemExit("[ERROR] Could not infer root ZoneName from DOMAIN; pass --zone-name explicitly.")

if not allowed_email:
    allowed_email = input(f"Enter email address to allow in Zero Trust Access for {private_hostname}: ").strip()
if not allowed_email:
    raise SystemExit("[ERROR] AllowedEmail is required for Cloudflare Zero Trust Access policy.")

if private_info["account_id"] != demo_info["account_id"]:
    raise SystemExit(
        f"[ERROR] Account ID mismatch between deploy/linux/config.sh ({private_info['account_id']}) "
        f"and deploy/linux/demo/config.sh ({demo_info['account_id']})."
    )

account_id = private_info["account_id"]
private_tunnel_id = private_info["tunnel_id"]
demo_tunnel_id = demo_info["tunnel_id"]

print()
print("===============================================================")
print(f"   CLOUDFLARE PROVISIONING - {zone_name}")
print("===============================================================")
print(f"  Account ID     : {account_id}")
print(f"  Private Tunnel : {private_tunnel_id} -> {private_hostname} ({args.private_service})")
print(f"  Demo Tunnel    : {demo_tunnel_id} -> {demo_hostname} ({args.demo_service})")
print(f"  Access Policy  : Allow {allowed_email} on {private_hostname}")
print()

api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip() or os.environ.get("CF_API_TOKEN", "").strip()
if not api_token:
    xdg_cfg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    cf_cfg_paths = [
        os.path.join(xdg_cfg, "cloudflare", "config", "default.json"),
        os.path.expanduser("~/.config/cloudflare/config/default.json"),
    ]
    for cfg_path in cf_cfg_paths:
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_obj = json.load(f)
                if cfg_obj.get("oauth_token"):
                    api_token = str(cfg_obj["oauth_token"]).strip()
                    print("  [Auth] Using authenticated session from Cloudflare CLI (cf auth login).")
                    break
            except Exception:
                pass

if not api_token:
    print("Create a Cloudflare API Token at: https://dash.cloudflare.com/profile/api-tokens")
    print("Required permissions:")
    print("  - Account | Cloudflare Tunnel (or Cloudflare One Connectors) | Edit")
    print("  - Account | Access: Apps and Policies                        | Edit")
    print("  - Account | Access: Organizations, Identity Providers, Groups| Edit")
    print("  - Zone    | DNS                                              | Edit")
    print("  - Zone    | Zone WAF                                         | Edit")
    print("  - Zone    | Zone Settings                                    | Edit")
    print("  - Zone    | Zone                                             | Read")
    print()
    api_token = getpass.getpass("Paste Cloudflare API Token (input hidden): ").strip()

if not api_token:
    raise SystemExit("[ERROR] No Cloudflare API Token provided.")


def cf_api(method: str, path: str, body: object = None, allow_404: bool = False):
    url = f"https://api.cloudflare.com/client/v4{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if allow_404 and err.code == 404:
            return None
        detail = err.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[ERROR] HTTP {err.code} on {method} {path}: {detail}") from err
    if not payload.get("success"):
        errs = "; ".join(f"[{e.get('code')}] {e.get('message')}" for e in payload.get("errors", []))
        raise SystemExit(f"[ERROR] Cloudflare API error on {method} {path}: {errs}")
    return payload


# ---- 1. Resolve Zone & Ensure SSL=Full -------------------------------------
print(f"[1/5] Looking up zone '{zone_name}'...")
zones_resp = cf_api("GET", f"/zones?name={urllib.parse.quote(zone_name)}")
if not zones_resp.get("result"):
    raise SystemExit(f"[ERROR] Zone '{zone_name}' not found in Cloudflare account {account_id}.")
zone = zones_resp["result"][0]
zone_id = zone["id"]
plan_name = str(zone.get("plan", {}).get("name", ""))
plan_legacy = str(zone.get("plan", {}).get("legacy_id", ""))
is_free_plan = plan_legacy == "free" or "free" in plan_name.lower()
print(f"  [OK] Zone ID: {zone_id} (Plan: {plan_name}, Status: {zone.get('status')})")

try:
    ssl_resp = cf_api("GET", f"/zones/{zone_id}/settings/ssl")
    if ssl_resp["result"]["value"] in ("flexible", "off"):
        cf_api("PATCH", f"/zones/{zone_id}/settings/ssl", {"value": "full"})
        print("  [OK] Updated Zone SSL/TLS encryption mode to Full.")
except SystemExit as exc:
    print(f"  [SKIP] Could not inspect/update SSL setting (non-fatal): {exc}")

# ---- 2. Configure Both Tunnel Ingress Routes -------------------------------
print("[2/5] Configuring Cloudflare Tunnel ingress routes...")
cf_api(
    "PUT",
    f"/accounts/{account_id}/cfd_tunnel/{private_tunnel_id}/configurations",
    {
        "config": {
            "ingress": [
                {"hostname": private_hostname, "service": args.private_service},
                {"service": "http_status:404"},
            ]
        }
    },
)
print(f"  [OK] Private tunnel ({private_tunnel_id}) -> {private_hostname} -> {args.private_service}")

cf_api(
    "PUT",
    f"/accounts/{account_id}/cfd_tunnel/{demo_tunnel_id}/configurations",
    {
        "config": {
            "ingress": [
                {"hostname": demo_hostname, "service": args.demo_service},
                {"service": "http_status:404"},
            ]
        }
    },
)
print(f"  [OK] Demo tunnel ({demo_tunnel_id}) -> {demo_hostname} -> {args.demo_service}")


# ---- 3. Upsert Proxied CNAME DNS Records -----------------------------------
print("[3/5] Upserting proxied CNAME DNS records...")


def upsert_tunnel_cname(hostname: str, tunnel_id: str) -> None:
    target = f"{tunnel_id}.cfargotunnel.com"
    existing = cf_api("GET", f"/zones/{zone_id}/dns_records?name={urllib.parse.quote(hostname)}")
    record_body = {
        "type": "CNAME",
        "name": hostname,
        "content": target,
        "proxied": True,
        "ttl": 1,
        "comment": f"Managed by quick_pp Cloudflare Tunnel ({tunnel_id})",
    }
    if existing.get("result"):
        rec_id = existing["result"][0]["id"]
        cf_api("PUT", f"/zones/{zone_id}/dns_records/{rec_id}", record_body)
        print(f"  [OK] Updated DNS CNAME: {hostname} -> {target} (Proxied)")
    else:
        cf_api("POST", f"/zones/{zone_id}/dns_records", record_body)
        print(f"  [OK] Created DNS CNAME: {hostname} -> {target} (Proxied)")


upsert_tunnel_cname(private_hostname, private_tunnel_id)
upsert_tunnel_cname(demo_hostname, demo_tunnel_id)

# ---- 4. Zero Trust Access Policy & Self-Hosted App -------------------------
print(f"[4/5] Configuring Cloudflare Zero Trust Access for {private_hostname}...")
org_resp = cf_api("GET", f"/accounts/{account_id}/access/organizations", allow_404=True)
if not org_resp or not org_resp.get("result") or not org_resp["result"].get("auth_domain"):
    raise SystemExit(
        "[ERROR] Zero Trust Organization is not initialized yet. "
        "Open https://one.dash.cloudflare.com/ once to pick your team name (Free plan), then re-run this script."
    )
print(f"  [OK] Zero Trust Team Domain: {org_resp['result']['auth_domain']}")

idps_resp = cf_api("GET", f"/accounts/{account_id}/access/identity_providers")
otp_idp = next((i for i in idps_resp.get("result", []) if i.get("type") == "onetimepin"), None)
if not otp_idp:
    otp_idp = cf_api(
        "POST",
        f"/accounts/{account_id}/access/identity_providers",
        {"name": "One-time PIN", "type": "onetimepin", "config": {}},
    )["result"]
    print("  [OK] Enabled One-time PIN Identity Provider.")
else:
    print("  [OK] One-time PIN Identity Provider already active.")

policy_name = "Allow Owner Only (quick_pp)"
policies_resp = cf_api("GET", f"/accounts/{account_id}/access/policies")
existing_policy = next((p for p in policies_resp.get("result", []) if p.get("name") == policy_name), None)
policy_body = {
    "name": policy_name,
    "decision": "allow",
    "include": [{"email": {"email": allowed_email}}],
    "session_duration": "168h",
}
if existing_policy:
    policy_id = cf_api("PUT", f"/accounts/{account_id}/access/policies/{existing_policy['id']}", policy_body)["result"]["id"]
    print(f"  [OK] Updated reusable Access policy '{policy_name}' ({policy_id}) -> {allowed_email}")
else:
    policy_id = cf_api("POST", f"/accounts/{account_id}/access/policies", policy_body)["result"]["id"]
    print(f"  [OK] Created reusable Access policy '{policy_name}' ({policy_id}) -> {allowed_email}")

apps_resp = cf_api("GET", f"/accounts/{account_id}/access/apps")
existing_app = next((a for a in apps_resp.get("result", []) if a.get("domain") == private_hostname), None)
app_body = {
    "name": "quick_pp Private",
    "domain": private_hostname,
    "type": "self_hosted",
    "session_duration": "168h",
    "app_launcher_visible": True,
    "http_only_cookie_attribute": True,
    "same_site_cookie_attribute": "lax",
    "allowed_idps": [str(otp_idp["id"])],
    "auto_redirect_to_identity": True,
    "policies": [{"id": policy_id, "precedence": 1}],
}
if existing_app:
    cf_api("PUT", f"/accounts/{account_id}/access/apps/{existing_app['id']}", app_body)
    print(f"  [OK] Updated Access Application 'quick_pp Private' ({private_hostname})")
else:
    cf_api("POST", f"/accounts/{account_id}/access/apps", app_body)
    print(f"  [OK] Created Access Application 'quick_pp Private' ({private_hostname})")

# ---- 5. WAF Rate Limiting Rule for demo.quick-pp.com -----------------------
print(f"[5/5] Configuring WAF Rate Limiting rule on {demo_hostname}...")
rl_period = 10 if is_free_plan else 60
rl_requests = 20 if is_free_plan else 60
rl_mitigation = 10 if is_free_plan else 60

demo_rule_desc = "quick_pp Demo Rate Limit"
demo_rule = {
    "description": demo_rule_desc,
    "expression": f'(http.host eq "{demo_hostname}" and not starts_with(http.request.uri.path, "/_app/"))',
    "action": "block",
    "enabled": True,
    "ratelimit": {
        "characteristics": ["cf.colo.id", "ip.src"],
        "period": rl_period,
        "requests_per_period": rl_requests,
        "mitigation_timeout": rl_mitigation,
    },
}

existing_ruleset = cf_api("GET", f"/zones/{zone_id}/rulesets/phases/http_ratelimit/entrypoint", allow_404=True)
rules_to_put = []
if existing_ruleset and existing_ruleset.get("result") and existing_ruleset["result"].get("rules") and not is_free_plan:
    for r in existing_ruleset["result"]["rules"]:
        if r.get("description") != demo_rule_desc:
            rules_to_put.append(r)
rules_to_put.append(demo_rule)

cf_api("PUT", f"/zones/{zone_id}/rulesets/phases/http_ratelimit/entrypoint", {"rules": rules_to_put})
print(
    f"  [OK] Active WAF Rate Limit: {rl_requests} req / {rl_period}s "
    f"(block {rl_mitigation}s) on {demo_hostname} (+ 60 req/min sustained in Nginx)"
)

print()
print("===============================================================")
print("   CLOUDFLARE CONFIGURATION COMPLETE!")
print("===============================================================")
print(f"  Private App : https://{private_hostname}  (Zero Trust: {allowed_email})")
print(f"  Public Demo : https://{demo_hostname}     (2 GB cap + WAF & Nginx rate limits)")
print()
PY
