#!/usr/bin/env bash
# ===================================================================
#  destroy.sh - Tear down the quick_pp Live Demo (Linux)
#
#  Safe for shared droplets! Lets you choose between:
#    1. APP ONLY (Recommended for shared droplets): Stops and removes
#       only the quick_pp_demo containers and /opt/quick_pp_demo
#       directory, leaving the droplet and other apps untouched.
#    2. ENTIRE DROPLET: Deletes the DigitalOcean droplet itself.
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
  echo "[destroy-demo] config.sh not found." >&2
  exit 1
fi

# shellcheck disable=SC1090
source <(tr -d '\r' < "${DEPLOY_DIR}/config.sh")

echo ""
echo "==============================================================="
echo "   TEARDOWN - quick_pp Interactive Live Demo"
echo "==============================================================="
echo ""
echo "Choose teardown mode:"
echo "  [1] Remove DEMO APP ONLY (safe for shared droplets - keeps droplet)"
echo "  [2] Destroy ENTIRE DROPLET (${DROPLET_NAME})"
echo ""
read -r -p "Select 1 or 2 (or N to cancel): " MODE

resolve_ip() {
  local ip="${SHARED_DROPLET_IP:-}"
  local ip_cache="${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"
  if [[ -z "$ip" && -f "$ip_cache" ]]; then
    ip="$(head -n1 "$ip_cache" | tr -d '[:space:]')"
  fi
  if [[ -z "$ip" ]] && command -v doctl >/dev/null 2>&1; then
    ip="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  fi
  printf '%s' "$ip"
}

if [[ "$MODE" == "1" ]]; then
  PUBLIC_IP="$(resolve_ip)"
  if [[ -z "$PUBLIC_IP" ]]; then
    echo "[ERROR] Could not resolve droplet IP." >&2
    exit 1
  fi
  chmod 600 "$KEY_FILE" 2>/dev/null || true
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i "$KEY_FILE")
  echo ""
  echo "[1/2] Stopping and removing ${COMPOSE_PROJECT} containers & images on ${PUBLIC_IP}..."
  ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker compose -p '${COMPOSE_PROJECT}' -f '${REMOTE_DIR}/deploy/demo/docker-compose.yaml' down --rmi local -v 2>/dev/null || true" </dev/null
  echo "[2/2] Removing ${REMOTE_DIR}..."
  ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "rm -rf '${REMOTE_DIR}'" </dev/null
  echo ""
  echo "[OK] Demo app removed from ${PUBLIC_IP}. Shared droplet and other apps are untouched."
  exit 0
elif [[ "$MODE" == "2" ]]; then
  read -r -p "Type DESTROY to permanently delete droplet ${DROPLET_NAME}: " CONFIRM
  if [[ "$CONFIRM" != "DESTROY" ]]; then
    echo "Aborted."
    exit 0
  fi

  PUBLIC_IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  if [[ -n "$PUBLIC_IP" && -f "$KEY_FILE" ]]; then
    chmod 600 "$KEY_FILE" 2>/dev/null || true
    SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes -i "$KEY_FILE")
    TOTAL_COUNT="$(ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker ps -qa | wc -l" </dev/null 2>/dev/null | tr -d '[:space:]' || echo 0)"
    OUR_COUNT="$(ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker ps -qa --filter label=com.docker.compose.project=${COMPOSE_PROJECT} | wc -l" </dev/null 2>/dev/null | tr -d '[:space:]' || echo 0)"
    TOTAL_COUNT="${TOTAL_COUNT:-0}"
    OUR_COUNT="${OUR_COUNT:-0}"

    if [[ "$TOTAL_COUNT" =~ ^[0-9]+$ && "$OUR_COUNT" =~ ^[0-9]+$ ]] && (( TOTAL_COUNT > OUR_COUNT )); then
      echo ""
      echo "[WARN] ${DROPLET_NAME} is also running containers from OTHER projects:"
      ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker ps -a --format '{{.Names}}'" </dev/null 2>/dev/null || true
      echo "[WARN] Deleting ${DROPLET_NAME} deletes them and all their data too!"
      echo ""
      read -r -p "Type the droplet name (${DROPLET_NAME}) to delete it anyway: " CONFIRM2
      if [[ "${CONFIRM2,,}" != "${DROPLET_NAME,,}" ]]; then
        echo "Aborted - nothing was deleted."
        exit 0
      fi
    fi
  fi

  echo "Deleting droplet ${DROPLET_NAME}..."
  doctl compute droplet delete "$DROPLET_NAME" --force
  rm -f "${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"
  echo "[OK] Droplet ${DROPLET_NAME} deleted."
else
  echo "Aborted."
  exit 0
fi
