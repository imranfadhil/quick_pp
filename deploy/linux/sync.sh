#!/usr/bin/env bash
# ===================================================================
#  sync.sh - Push quick_pp code to the DigitalOcean droplet over SSH
#  Resolves the droplet IP automatically via doctl (no manual IP).
#
#  Usage:
#    ./sync.sh                push code + restart containers
#    ./sync.sh --restart      force docker rebuild + restart frontend
#    ./sync.sh --from-deploy  called by deploy.sh (first push)
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/../.." && pwd)"
cd "$DEPLOY_DIR"

if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
  echo "[sync] config.sh not found - seeding from .env via config.example.sh"
  bash "${DEPLOY_DIR}/seed-config.sh"
  echo "[sync] Review deploy/linux/config.sh, then re-run ./sync.sh."
  exit 1
fi

# shellcheck disable=SC1090
source <(tr -d '\r' < "${DEPLOY_DIR}/config.sh")

MODE="${1:-}"
BUILD=0
if [[ "$MODE" == "--restart" || "$MODE" == "--from-deploy" ]]; then
  BUILD=1
fi

echo ""
echo "=== SYNC - quick_pp UI (DigitalOcean) ==="
echo ""

# -- Resolve public IP ----------------------------------------------
IP_CACHE="${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"
PUBLIC_IP=""
if [[ -f "$IP_CACHE" ]]; then
  PUBLIC_IP="$(head -n1 "$IP_CACHE" | tr -d '[:space:]')"
fi
if [[ -z "$PUBLIC_IP" ]] && command -v doctl >/dev/null 2>&1; then
  PUBLIC_IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
fi
if [[ -z "$PUBLIC_IP" ]]; then
  echo "[ERROR] No droplet IP found. Run ./deploy.sh first." >&2
  exit 1
fi
echo "[INFO] Target: $PUBLIC_IP"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "[ERROR] SSH key not found: $KEY_FILE" >&2
  echo "        (If migrating from Windows, copy your ${KEY_NAME} key pair to ~/.ssh/)" >&2
  exit 1
fi
chmod 600 "$KEY_FILE" 2>/dev/null || true

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=15
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
  -i "$KEY_FILE"
)
SSH_RETRIES=5

ssh_run() {
  local cmd="$1"
  local attempt=1
  while true; do
    if ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "$cmd" </dev/null; then
      return 0
    fi
    if (( attempt >= SSH_RETRIES )); then
      return 1
    fi
    echo "[WARN] SSH attempt ${attempt}/${SSH_RETRIES} failed - retrying..."
    sleep 5
    ((attempt++))
  done
}

ssh_pipe_file() {
  local src_file="$1"
  local remote_cmd="$2"
  local attempt=1
  while true; do
    if ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "$remote_cmd" < "$src_file"; then
      return 0
    fi
    if (( attempt >= SSH_RETRIES )); then
      return 1
    fi
    echo "[WARN] SSH pipe attempt ${attempt}/${SSH_RETRIES} failed - retrying..."
    sleep 5
    ((attempt++))
  done
}

TAR_TMP="$(mktemp "${TMPDIR:-/tmp}/${PROJECT_NAME}-XXXXXX.tar")"
ENV_TMP="$(mktemp "${TMPDIR:-/tmp}/${PROJECT_NAME}-configure-env-XXXXXX.sh")"
cleanup() {
  rm -f "$TAR_TMP" "$ENV_TMP"
}
trap cleanup EXIT

# -- Package code (exclude heavy local folders, venvs, notebooks, data) --
echo "[1/5] Packaging project..."
tar -cf "$TAR_TMP" \
  --warning=no-file-changed \
  -C "$REPO_ROOT" \
  --exclude="./node_modules" \
  --exclude="node_modules" \
  --exclude=".svelte-kit" \
  --exclude="./quick_pp/app/frontend/build" \
  --exclude="./.git" \
  --exclude="./.venv*" \
  --exclude="./venv*" \
  --exclude="./.uv_tools" \
  --exclude="./.cloudflare" \
  --exclude="./.files" \
  --exclude="./.vite" \
  --exclude="__pycache__" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude=".ruff_cache" \
  --exclude="./dist" \
  --exclude="./build" \
  --exclude="*.egg-info" \
  --exclude="./notebooks" \
  --exclude="./docs/_build" \
  --exclude="./site" \
  --exclude="./data" \
  --exclude="./uploads" \
  --exclude="mlruns" \
  --exclude="./quick_pp/app/docker/data" \
  --exclude="./quick_pp.db*" \
  --exclude="./.env" \
  --exclude="./deploy" \
  --exclude="*.log" \
  .
echo "[OK] Packaged."

# -- Copy to droplet ------------------------------------------------
echo "[2/5] Ensuring remote directory exists..."
if ! ssh_run "mkdir -p '${REMOTE_DIR}/uploads'"; then
  echo "[ERROR] Could not create remote directory. Droplet may still be booting." >&2
  exit 1
fi
echo "[OK] Remote directory ready."

echo "[3/5] Copying to droplet..."
if ! ssh_pipe_file "$TAR_TMP" "cat > '${REMOTE_DIR}/app.tar'"; then
  echo "[ERROR] Upload failed. Droplet may still be booting, or SSH not ready yet." >&2
  exit 1
fi
rm -f "$TAR_TMP"
echo "[OK] Uploaded."

# -- Extract + configure .env ---------------------------------------
echo "[4/5] Extracting and configuring environment..."
if ! ssh_run "cd '${REMOTE_DIR}' && rm -rf quick_pp/app/frontend/build && tar -xf app.tar && rm -f app.tar"; then
  echo "[ERROR] Extraction failed." >&2
  exit 1
fi

if ! ssh_run "cd '${REMOTE_DIR}' && (test -f .env || cp .env.example .env)"; then
  echo "[ERROR] Could not create/verify remote .env. Droplet may be unreachable." >&2
  exit 1
fi

tr -d '\r' < "${DEPLOY_DIR}/../common/configure-env.sh" > "$ENV_TMP"
REMOTE_ENV_ARGS="$(printf '%q ' \
  "${REMOTE_DIR}" \
  "${POSTGRES_USER:-qpp_user}" \
  "${POSTGRES_DB:-quick_pp}" \
  "${POSTGRES_PASSWORD:-}" \
  "${WEB_PORT:-8080}" \
  "${TAVILY_API_KEY:-}" \
  "${DOMAIN:-}" \
  "${CF_TUNNEL_TOKEN:-}")"

if ! ssh_pipe_file "$ENV_TMP" "bash -s -- ${REMOTE_ENV_ARGS}"; then
  echo "[ERROR] Failed to configure remote .env - SSH error or script failure." >&2
  echo "        Nothing was started. Re-run ./sync.sh once the droplet is reachable." >&2
  exit 1
fi
rm -f "$ENV_TMP"
echo "[OK] Environment ready."

# -- Start containers -----------------------------------------------
echo "[5/5] Starting containers..."
COMPOSE_CMD="docker compose --env-file '${REMOTE_DIR}/.env' -f '${REMOTE_DIR}/quick_pp/app/docker/docker-compose.yaml'"

# Stop qpp-frontend before `up -d` so an in-progress `npm install` is never killed mid-extraction.
if [[ "$MODE" != "--from-deploy" ]]; then
  ssh_run "${COMPOSE_CMD} stop qpp-frontend" >/dev/null 2>&1 || true
fi

TUNNEL_UP_FLAGS="--no-recreate --no-deps"
if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then
  TUNNEL_UP_FLAGS="--no-deps"
fi

if [[ "$MODE" == "--from-deploy" ]]; then
  REMOTE_CMD="${COMPOSE_CMD} up -d --build"
elif [[ "$BUILD" -eq 1 ]]; then
  REMOTE_CMD="${COMPOSE_CMD} up -d --no-recreate --no-deps postgres && ${COMPOSE_CMD} up -d --build --no-deps qpp-backend qpp-frontend && ${COMPOSE_CMD} up -d --no-recreate --no-deps web && ${COMPOSE_CMD} up -d ${TUNNEL_UP_FLAGS} quicktunnel"
else
  REMOTE_CMD="${COMPOSE_CMD} up -d --no-recreate --no-deps postgres web qpp-backend qpp-frontend && ${COMPOSE_CMD} up -d ${TUNNEL_UP_FLAGS} quicktunnel && ${COMPOSE_CMD} restart qpp-backend"
fi

if ! ssh_run "$REMOTE_CMD"; then
  echo "[ERROR] Remote deploy failed. Check the droplet:" >&2
  echo "        ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP}" >&2
  exit 1
fi

echo ""
echo "[OK] Sync complete."
[[ "$BUILD" -eq 1 ]] && echo "[OK] Containers rebuilt."

echo ""
echo "[INFO] Waiting for frontend build to complete (takes ~1-2 min on first deploy)..."
FE_READY=0
for ((n=1; n<=40; n++)); do
  if ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "curl -sf -o /dev/null http://127.0.0.1:${WEB_PORT:-8080}/" </dev/null >/dev/null 2>&1; then
    FE_READY=1
    break
  fi
  sleep 4
done

if [[ "$FE_READY" -eq 1 ]]; then
  echo "[OK] Frontend is built and serving HTTP 200."
else
  echo "[WARN] Frontend build is still running in the background. Check logs with:"
  echo "       ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP} \"${COMPOSE_CMD} logs -f qpp-frontend\""
fi

echo ""
echo "[INFO] Container status:"
ssh_run "${COMPOSE_CMD} ps"

# -- Tunnel URL (Named Cloudflare Tunnel or random *.trycloudflare.com) ---
if [[ -n "${CF_TUNNEL_TOKEN:-}" ]]; then
  echo ""
  if [[ -n "${DOMAIN:-}" ]]; then
    echo "==============================================================="
    echo "   PRIVATE APP URL (Cloudflare Tunnel): https://${DOMAIN}"
    echo "==============================================================="
  else
    echo "[OK] Cloudflare Named Tunnel active (CF_TUNNEL_TOKEN configured)."
  fi
  exit 0
fi

QUICK_URL=""
echo ""
echo "[INFO] Waiting for the Cloudflare Quick Tunnel URL..."
for ((n=1; n<=15; n++)); do
  QUICK_URL="$(ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "${COMPOSE_CMD} logs quicktunnel 2>/dev/null | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1" </dev/null 2>/dev/null || true)"
  if [[ -n "$QUICK_URL" ]]; then
    break
  fi
  sleep 2
done

echo ""
if [[ -n "$QUICK_URL" ]]; then
  echo "[OK] App (HTTPS Tunnel): $QUICK_URL"
else
  echo "[WARN] Quick tunnel URL not ready yet. Check:"
  echo "       ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP} \"${COMPOSE_CMD} logs quicktunnel\""
fi
