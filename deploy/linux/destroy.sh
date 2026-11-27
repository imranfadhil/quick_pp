#!/usr/bin/env bash
# ===================================================================
#  destroy.sh - Tear down the DigitalOcean droplet (+ SSH key)
#  WARNING: permanently deletes the droplet and all data on it, including
#           any OTHER app you host on that same droplet.
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
  echo "[destroy] config.sh not found. Run ./deploy.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source <(tr -d '\r' < "${DEPLOY_DIR}/config.sh")

echo ""
echo "==============================================================="
echo "   DESTROY - quick_pp UI (DigitalOcean)"
echo "==============================================================="
echo ""

read -r -p "Type DESTROY to confirm: " CONFIRM
if [[ "$CONFIRM" != "DESTROY" ]]; then
  echo "Aborted."
  exit 0
fi

# -- Shared-droplet guard --------------------------------------------
# Compare the host's total container count with the count belonging to
# THIS compose project (docker) and require the droplet name to be typed
# as a second confirmation when foreign containers are found.
PUBLIC_IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
if [[ -n "$PUBLIC_IP" && -f "$KEY_FILE" ]]; then
  chmod 600 "$KEY_FILE" 2>/dev/null || true
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes -i "$KEY_FILE")
  PROJECT_SLUG="docker"
  TOTAL_COUNT="$(ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker ps -qa | wc -l" </dev/null 2>/dev/null | tr -d '[:space:]' || echo 0)"
  OUR_COUNT="$(ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker ps -qa --filter label=com.docker.compose.project=${PROJECT_SLUG} | wc -l" </dev/null 2>/dev/null | tr -d '[:space:]' || echo 0)"
  TOTAL_COUNT="${TOTAL_COUNT:-0}"
  OUR_COUNT="${OUR_COUNT:-0}"

  if [[ "$TOTAL_COUNT" =~ ^[0-9]+$ && "$OUR_COUNT" =~ ^[0-9]+$ ]] && (( TOTAL_COUNT > OUR_COUNT )); then
    echo ""
    echo "[WARN] ${DROPLET_NAME} is also running containers from OTHER projects:"
    ssh "${SSH_OPTS[@]}" "root@${PUBLIC_IP}" "docker ps -a --format '{{.Names}}'" </dev/null 2>/dev/null || true
    echo "[WARN] Deleting ${DROPLET_NAME} deletes them and all their data too."
    echo ""
    read -r -p "Type the droplet name (${DROPLET_NAME}) to delete it anyway: " CONFIRM2
    if [[ "${CONFIRM2,,}" != "${DROPLET_NAME,,}" ]]; then
      echo "Aborted - nothing was deleted."
      exit 0
    fi
  fi
fi

# -- Delete droplet -------------------------------------------------
echo "[1/2] Deleting droplet ${DROPLET_NAME}..."
if doctl compute droplet get "$DROPLET_NAME" >/dev/null 2>&1; then
  if doctl compute droplet delete "$DROPLET_NAME" --force >/dev/null 2>&1; then
    echo "[OK] Droplet deleted."
  else
    echo "[WARN] Failed to delete droplet - check the DO console." >&2
  fi
else
  echo "[SKIP] No droplet named ${DROPLET_NAME} found."
fi

# -- Delete SSH key from DO (keeps the local key file) --------------
echo "[2/2] Removing SSH key from DO..."
KEY_ID="$(doctl compute ssh-key list --format ID,Name --no-header 2>/dev/null | awk -v name="$KEY_NAME" 'tolower($2)==tolower(name) {print $1; exit}')"
if [[ -n "$KEY_ID" ]]; then
  if doctl compute ssh-key delete "$KEY_ID" --force >/dev/null 2>&1; then
    echo "[OK] SSH key removed from DO (${KEY_NAME}, ID: ${KEY_ID}). Local key file kept at ${KEY_FILE}."
  else
    echo "[WARN] Failed to remove SSH key from DO (ID: ${KEY_ID})." >&2
    echo "       Remove manually: doctl compute ssh-key delete ${KEY_ID}" >&2
  fi
else
  echo "[SKIP] No SSH key named ${KEY_NAME} on DO."
fi

# -- Clean cached state ---------------------------------------------
rm -f "${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"

echo ""
echo "==============================================================="
echo "   DESTROY COMPLETE"
echo "==============================================================="
