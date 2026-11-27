#!/usr/bin/env bash
# ===================================================================
#  sync.sh - Push quick_pp Live Demo to a Shared or Small Droplet (Linux)
#
#  Packages the app + Volve/Costa raw LAS/CSV seed files (~2.5 MB),
#  ensures 2 GB swap is active on the droplet, builds/starts the
#  isolated quick_pp_demo stack on port 7860, and prints the
#  Cloudflare Tunnel HTTPS URL.
#
#  Usage:
#    ./sync.sh                push code + restart demo container
#    ./sync.sh --restart      force Docker image rebuild + restart
#    ./sync.sh --from-deploy  called by deploy.sh
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/../../.." && pwd)"
cd "$DEPLOY_DIR"

if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
  echo "[sync-demo] config.sh not found - generating from config.example.sh"
  bash "${DEPLOY_DIR}/seed-config.sh"
  echo "[sync-demo] Review deploy/linux/demo/config.sh, then re-run ./sync.sh."
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
echo "=== SYNC - quick_pp Interactive Live Demo ==="
echo ""

# -- Resolve public IP ----------------------------------------------
IP_CACHE="${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"
PUBLIC_IP="${SHARED_DROPLET_IP:-}"
if [[ -z "$PUBLIC_IP" && -f "$IP_CACHE" ]]; then
  PUBLIC_IP="$(head -n1 "$IP_CACHE" | tr -d '[:space:]')"
fi
if [[ -z "$PUBLIC_IP" ]] && command -v doctl >/dev/null 2>&1; then
  PUBLIC_IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
fi
if [[ -z "$PUBLIC_IP" ]]; then
  echo "[ERROR] No droplet IP found. Set SHARED_DROPLET_IP in config.sh or run ./deploy.sh first." >&2
  exit 1
fi
echo "[INFO] Target Droplet IP: ${PUBLIC_IP}  (Remote path: ${REMOTE_DIR})"

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
DEMO_ENV_TMP="$(mktemp "${TMPDIR:-/tmp}/${PROJECT_NAME}-demo-XXXXXX.env")"
cleanup() {
  rm -f "$TAR_TMP" "$DEMO_ENV_TMP"
}
trap cleanup EXIT

# -- Package code + minimal Volve/Costa seed files -------------------
echo "[1/5] Packaging demo project and Volve/Costa seed files..."
tar -cf "$TAR_TMP" \
  --warning=no-file-changed \
  -C "$REPO_ROOT" \
  --exclude="node_modules" \
  --exclude=".svelte-kit" \
  --exclude="quick_pp/app/frontend/build" \
  --exclude="__pycache__" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude=".ruff_cache" \
  --exclude="*.egg-info" \
  --exclude="quick_pp/app/docker/data" \
  --exclude="*.log" \
  pyproject.toml README.md quick_pp \
  deploy/demo/Dockerfile \
  deploy/demo/docker-compose.yaml \
  deploy/demo/nginx.conf \
  deploy/demo/start_demo.sh \
  deploy/demo/seed_demo.py \
  notebooks/data/01_raw/VOLVE/15-9-19-A.las \
  notebooks/data/01_raw/VOLVE/VOLVE_15-9_Surveys.csv \
  notebooks/data/01_raw/COSTA/HW-3.las \
  notebooks/data/01_raw/COSTA/HW-4.las \
  notebooks/data/01_raw/COSTA/HW-5.las \
  notebooks/data/01_raw/COSTA/HW_core_data_all.csv \
  notebooks/data/01_raw/36_7-3.las \
  notebooks/data/01_raw/formation_pressure_data.csv
echo "[OK] Packaged."

# -- Ensure remote directory + 2 GB swap on small/shared droplet -----
echo "[2/5] Ensuring remote directory and swap file on droplet..."
if ! ssh_run "mkdir -p '${REMOTE_DIR}' && (swapon --show | grep -q '/swapfile' || (fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile)) || true"; then
  echo "[ERROR] Could not reach droplet over SSH." >&2
  exit 1
fi
echo "[OK] Remote directory and swap verified."

# -- Upload archive --------------------------------------------------
echo "[3/5] Uploading archive to ${REMOTE_DIR}..."
if ! ssh_pipe_file "$TAR_TMP" "cat > '${REMOTE_DIR}/demo_app.tar'"; then
  echo "[ERROR] Upload failed." >&2
  exit 1
fi
rm -f "$TAR_TMP"
echo "[OK] Uploaded."

# -- Extract and write demo .env -------------------------------------
echo "[4/5] Extracting and configuring ${REMOTE_DIR}/.env..."
if ! ssh_run "cd '${REMOTE_DIR}' && tar -xf demo_app.tar && rm -f demo_app.tar"; then
  echo "[ERROR] Extraction failed." >&2
  exit 1
fi

PUBLIC_URL_VAL=""
if [[ -n "${DOMAIN:-}" ]]; then
  PUBLIC_URL_VAL="https://${DOMAIN}"
fi
DEMO_MEM_LIMIT="${DEMO_MEM_LIMIT:-2g}"
CF_TOKEN_TRIMMED="$(printf '%s' "${CF_TUNNEL_TOKEN:-}" | tr -d '[:space:]')"
if [[ -n "$CF_TOKEN_TRIMMED" ]]; then
  CF_TUNNEL_ARG="run"
else
  CF_TUNNEL_ARG="--url=http://qpp-demo:7860"
fi

cat > "$DEMO_ENV_TMP" <<EOF
QPP_DEMO_MODE=true
QPP_DISABLE_TRAINING=true
DEMO_PORT=${DEMO_PORT:-7860}
DEMO_BIND=${DEMO_BIND:-127.0.0.1}
DEMO_MEM_LIMIT=${DEMO_MEM_LIMIT}
DEMO_CPU_LIMIT=${DEMO_CPU_LIMIT:-0.85}
DEMO_MAX_USERS=${DEMO_MAX_USERS:-6}
DEMO_LAG_WARNING_USERS=${DEMO_LAG_WARNING_USERS:-3}
DEMO_IDLE_TIMEOUT_SECONDS=${DEMO_IDLE_TIMEOUT_SECONDS:-600}
DEMO_MAX_CONCURRENT_COMPUTE=${DEMO_MAX_CONCURRENT_COMPUTE:-2}
PUBLIC_URL=${PUBLIC_URL_VAL}
CF_TUNNEL_TOKEN=${CF_TOKEN_TRIMMED}
CF_TUNNEL_ARG=${CF_TUNNEL_ARG}
EOF

if ! ssh_pipe_file "$DEMO_ENV_TMP" "cat > '${REMOTE_DIR}/.env'"; then
  echo "[ERROR] Failed to write remote .env." >&2
  exit 1
fi
rm -f "$DEMO_ENV_TMP"
echo "[OK] Demo environment configured (Memory limit: ${DEMO_MEM_LIMIT})."

# -- Build & Start isolated demo compose stack -----------------------
echo "[5/5] Starting isolated demo stack (project: ${COMPOSE_PROJECT})..."
COMPOSE_CMD="docker compose -p '${COMPOSE_PROJECT}' --env-file '${REMOTE_DIR}/.env' -f '${REMOTE_DIR}/deploy/demo/docker-compose.yaml'"

DEMO_TUNNEL_FLAGS="--no-recreate --no-deps"
if [[ -n "$CF_TOKEN_TRIMMED" ]]; then
  DEMO_TUNNEL_FLAGS="--no-deps"
fi

if [[ "$MODE" == "--from-deploy" ]]; then
  REMOTE_CMD="${COMPOSE_CMD} up -d --build"
elif [[ "$BUILD" -eq 1 ]]; then
  REMOTE_CMD="${COMPOSE_CMD} up -d --build --no-deps qpp-demo && ${COMPOSE_CMD} up -d ${DEMO_TUNNEL_FLAGS} demo-quicktunnel"
else
  REMOTE_CMD="${COMPOSE_CMD} up -d --no-deps qpp-demo && ${COMPOSE_CMD} up -d ${DEMO_TUNNEL_FLAGS} demo-quicktunnel"
fi

if ! ssh_run "$REMOTE_CMD"; then
  echo "[ERROR] Demo container startup failed. Inspect logs via:" >&2
  echo "        ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP} \"${COMPOSE_CMD} logs\"" >&2
  exit 1
fi

if [[ "$BUILD" -eq 0 ]]; then
  ssh_run "${COMPOSE_CMD} restart qpp-demo" >/dev/null 2>&1 || true
fi

echo ""
echo "[OK] Demo stack running:"
ssh_run "${COMPOSE_CMD} ps"

# -- Retrieve Cloudflare Tunnel URL ----------------------------------
if [[ -n "$CF_TOKEN_TRIMMED" ]]; then
  echo ""
  if [[ -n "${DOMAIN:-}" ]]; then
    echo "==============================================================="
    echo "   LIVE DEMO URL (Cloudflare Tunnel): https://${DOMAIN}"
    echo "==============================================================="
  else
    echo "[OK] Cloudflare Named Tunnel active (CF_TUNNEL_TOKEN configured)."
  fi
  exit 0
fi

QUICK_URL=""
echo ""
echo "[INFO] Resolving Cloudflare Quick Tunnel HTTPS URL..."
for ((n=1; n<=15; n++)); do
  QUICK_URL="$(ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "${COMPOSE_CMD} logs demo-quicktunnel 2>/dev/null | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1" </dev/null 2>/dev/null || true)"
  if [[ -n "$QUICK_URL" ]]; then
    break
  fi
  sleep 2
done

echo ""
if [[ -n "$QUICK_URL" ]]; then
  echo "==============================================================="
  echo "   LIVE DEMO URL (HTTPS): ${QUICK_URL}"
  echo "==============================================================="
else
  echo "[WARN] Quick tunnel URL not ready yet. Check logs with:"
  echo "       ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP} \"${COMPOSE_CMD} logs demo-quicktunnel\""
fi
