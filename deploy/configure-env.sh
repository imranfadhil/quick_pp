#!/bin/sh
# Configure the droplet's .env from deploy/config.bat.
#
# Piped over stdin by sync.bat (`ssh ... "bash -s -- <args>" < configure-env.sh`)
# so cmd.exe never has to quote the shell code. Values arrive as positional
# args, so tokens containing ':' or '+' are never re-parsed by cmd.exe.
#
#   $1  REMOTE_DIR
#   $2  POSTGRES_USER
#   $3  POSTGRES_DB
#   $4  POSTGRES_PASSWORD
#   $5  WEB_PORT
#   $6  TAVILY_API_KEY
#   $7  DOMAIN
set -e

REMOTE_DIR="$1"; shift
cd "$REMOTE_DIR"

PG_USER="${1:-qpp_user}"
PG_DB="${2:-quick_pp}"
PG_PASS="$3"
WEB_PORT="${4:-8080}"
TAVILY_KEY="$5"
DOMAIN="$6"

# Helper to set or append a key=value in .env
set_kv() {
  key="$1"
  val="$2"
  if grep -q "^${key}=" .env; then
    sed -i "s#^${key}=.*#${key}=${val}#" .env
  else
    echo "${key}=${val}" >> .env
  fi
}

set_kv POSTGRES_USER "$PG_USER"
set_kv POSTGRES_DB "$PG_DB"
set_kv WEB_PORT "$WEB_PORT"
set_kv FRONTEND_MODE "prod"

# Rotate or set the DB password ONLY before the postgres data directory exists.
# Postgres applies POSTGRES_PASSWORD on first init only, so changing it after
# ./data/postgres is populated would leave the DB expecting the old password.
if grep -q '^POSTGRES_PASSWORD=\(xxx\|change_this\)' .env || ! grep -q '^POSTGRES_PASSWORD=' .env; then
  if [ -n "$PG_PASS" ] && [ "$PG_PASS" != "xxx" ] && ! echo "$PG_PASS" | grep -q '^change_this'; then
    set_kv POSTGRES_PASSWORD "$PG_PASS"
  elif grep -q '^POSTGRES_PASSWORD=\(xxx\|change_this\)' .env || ! grep -q '^POSTGRES_PASSWORD=' .env; then
    set_kv POSTGRES_PASSWORD "$(openssl rand -hex 16)"
  fi
fi

# Read the effective password from .env so QPP_DATABASE_URL always matches
CURRENT_PASS=$(grep '^POSTGRES_PASSWORD=' .env | head -1 | cut -d= -f2-)
set_kv QPP_DATABASE_URL "postgresql+psycopg2://${PG_USER}:${CURRENT_PASS}@postgres:5432/${PG_DB}"

if grep -q '^QPP_AUTH_SECRET_KEY=change_this' .env || ! grep -q '^QPP_AUTH_SECRET_KEY=' .env; then
  set_kv QPP_AUTH_SECRET_KEY "$(openssl rand -hex 32)"
fi

# Set VITE_BACKEND_URL to empty string so browser requests use relative paths
# (/quick_pp/...) through the unified Nginx reverse proxy + Cloudflare Tunnel.
set_kv VITE_BACKEND_URL ""

if [ -n "$TAVILY_KEY" ] && [ "$TAVILY_KEY" != "xxx" ]; then
  set_kv TAVILY_API_KEY "$TAVILY_KEY"
fi

if [ -n "$DOMAIN" ]; then
  set_kv PUBLIC_URL "https://$DOMAIN"
fi

echo '--- applied ---'
grep -E '^(POSTGRES_USER|POSTGRES_DB|WEB_PORT|FRONTEND_MODE|VITE_BACKEND_URL|PUBLIC_URL)=' .env || true
