#!/usr/bin/env bash
# ===================================================================
#  seed-config.sh - create deploy/linux/config.sh from config.example.sh,
#  pre-filling app-level values from the project .env file (and falling
#  back to deploy/windows/config.bat if migrating from Windows).
#
#  Usage:
#    ./seed-config.sh            create config.sh if missing
#    ./seed-config.sh --force    regenerate, overwriting config.sh
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    -f|--force|-Force) FORCE=1 ;;
    *)
      echo "[seed] Unknown option: $arg (use --force to overwrite)" >&2
      exit 1
      ;;
  esac
done

TEMPLATE="${DEPLOY_DIR}/config.example.sh"
OUT_FILE="${DEPLOY_DIR}/config.sh"
ENV_FILE="${DEPLOY_DIR}/../../.env"
BAT_FILE="${DEPLOY_DIR}/../windows/config.bat"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "[seed] config.example.sh not found at $TEMPLATE" >&2
  exit 1
fi

if [[ -f "$OUT_FILE" && "$FORCE" -eq 0 ]]; then
  echo "[seed] config.sh already exists - leaving it untouched (use --force to regenerate)."
  exit 0
fi

python3 - "$TEMPLATE" "$OUT_FILE" "$ENV_FILE" "$BAT_FILE" <<'PY'
import os
import re
import sys

template_path, out_path, env_path, bat_path = sys.argv[1:5]

def parse_dotenv(path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    if not os.path.isfile(path):
        print(f"[seed] No .env found at {path} - using template defaults.")
        return data
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            t = raw.strip()
            if not t or t.startswith("#") or "=" not in t:
                continue
            k, v = t.split("=", 1)
            k, v = k.strip(), v.strip()
            m_dq = re.match(r'^"([^"]*)"', v)
            m_sq = re.match(r"^'([^']*)'", v)
            if m_dq:
                v = m_dq.group(1).strip()
            elif m_sq:
                v = m_sq.group(1).strip()
            else:
                v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
            data[k] = v
    return data

def parse_bat(path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    if not os.path.isfile(path):
        return data
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = re.match(r'^\s*set\s+"?([A-Za-z_][A-Za-z0-9_]*)=([^"\r\n]*)', raw)
            if m:
                data[m.group(1)] = m.group(2).strip()
    return data

dotenv = parse_dotenv(env_path)
bat = parse_bat(bat_path)

direct_keys = [
    "POSTGRES_USER",
    "POSTGRES_DB",
    "POSTGRES_PASSWORD",
    "TAVILY_API_KEY",
    "WEB_PORT",
    "CF_TUNNEL_TOKEN",
    "ALLOWED_EMAIL",
]

values: dict[str, str] = {}
filled_env: list[str] = []
filled_bat: list[str] = []

def is_real_value(v: str) -> bool:
    return bool(v) and v != "xxx" and not v.startswith("change_this") and "%" not in v

for key in direct_keys:
    val = dotenv.get(key, "")
    if is_real_value(val):
        values[key] = val
        filled_env.append(key)
    elif is_real_value(bat.get(key, "")):
        values[key] = bat[key]
        filled_bat.append(key)

pub_url = dotenv.get("PUBLIC_URL", "")
if pub_url:
    domain = re.sub(r"/.*$", "", re.sub(r"^https?://", "", pub_url)).strip()
    if domain:
        values["DOMAIN"] = domain
        filled_env.append("DOMAIN")
elif is_real_value(bat.get("DOMAIN", "")):
    values["DOMAIN"] = bat["DOMAIN"]
    filled_bat.append("DOMAIN")

with open(template_path, "r", encoding="utf-8") as f:
    tpl_lines = [line.rstrip("\r\n") for line in f]

out_lines: list[str] = []
applied: list[str] = []
for idx, line in enumerate(tpl_lines):
    if idx == 1:
        out_lines.append("# AUTO-GENERATED from .env by seed-config.sh - regenerate with: ./seed-config.sh --force")
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
    if m and m.group(1) in values:
        key = m.group(1)
        escaped = values[key].replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        out_lines.append(f'{key}="{escaped}"')
        applied.append(key)
        continue
    out_lines.append(line)

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out_lines) + "\n")

os.chmod(out_path, 0o600)
print(f"[seed] Wrote {out_path}")
if applied:
    print(f"[seed] Pre-filled keys: {', '.join(applied)}")
else:
    print("[seed] No matching values found in .env - template defaults used.")
PY
