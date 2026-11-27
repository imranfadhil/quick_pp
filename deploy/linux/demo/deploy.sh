#!/usr/bin/env bash
# ===================================================================
#  deploy.sh - Deploy quick_pp Interactive Live Demo to a Shared or
#              Small DigitalOcean Droplet (Linux)
#
#  If SHARED_DROPLET_IP is set or DROPLET_NAME already exists on DO,
#  it reuses that droplet without creating a new one and deploys into
#  /opt/quick_pp_demo on port 7860.
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
  echo "[deploy-demo] config.sh not found - generating from config.example.sh"
  bash "${DEPLOY_DIR}/seed-config.sh"
  if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
    echo "[deploy-demo] Could not create config.sh. Edit deploy/linux/demo/config.sh, then re-run." >&2
    exit 1
  fi
  echo "[deploy-demo] Review deploy/linux/demo/config.sh (set DROPLET_NAME or SHARED_DROPLET_IP), then re-run ./deploy.sh."
  exit 1
fi

# shellcheck disable=SC1090
source <(tr -d '\r' < "${DEPLOY_DIR}/config.sh")

echo ""
echo "==============================================================="
echo "   QUICK_PP INTERACTIVE LIVE DEMO - DROPLET DEPLOY"
echo "==============================================================="
echo ""

# -- Check if SHARED_DROPLET_IP is explicitly provided --------------
if [[ -n "${SHARED_DROPLET_IP:-}" ]]; then
  PUBLIC_IP="$SHARED_DROPLET_IP"
  echo "[OK] Using configured SHARED_DROPLET_IP: ${PUBLIC_IP}"
  printf '%s\n' "$PUBLIC_IP" > "${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"
else
  # -- Prerequisites for doctl --------------------------------------
  if ! command -v doctl >/dev/null 2>&1; then
    echo "[ERROR] doctl not found. Install doctl or set SHARED_DROPLET_IP in deploy/linux/demo/config.sh." >&2
    echo "        Ubuntu install: sudo snap install doctl && sudo snap connect doctl:ssh-keys" >&2
    exit 1
  fi

  if ! doctl account get >/dev/null 2>&1; then
    echo "[ERROR] doctl not authenticated. Run: doctl auth init" >&2
    exit 1
  fi
  echo "[OK] doctl authenticated."

  # -- Step 1: Ensure a local SSH key + register it with DO ---------
  echo ""
  echo "[1/4] Ensuring SSH key..."
  mkdir -p "$(dirname "$KEY_FILE")"
  chmod 700 "$(dirname "$KEY_FILE")" 2>/dev/null || true

  if [[ ! -f "$KEY_FILE" ]]; then
    ssh-keygen -t ed25519 -N "" -f "$KEY_FILE" -C "$KEY_NAME"
    echo "[OK] Key generated: $KEY_FILE"
  else
    chmod 600 "$KEY_FILE" 2>/dev/null || true
    [[ -f "$PUB_FILE" ]] && chmod 644 "$PUB_FILE" 2>/dev/null || true
    echo "[OK] Local key exists: $KEY_FILE"
  fi

  if [[ ! -f "$PUB_FILE" ]]; then
    ssh-keygen -y -f "$KEY_FILE" > "$PUB_FILE"
    chmod 644 "$PUB_FILE" 2>/dev/null || true
  fi

  KEY_FP="$(doctl compute ssh-key list --format Name,FingerPrint --no-header | awk -v name="$KEY_NAME" 'tolower($1)==tolower(name) {print $2; exit}')"
  if [[ -n "$KEY_FP" ]]; then
    echo "[OK] SSH key already registered with DO (by name)."
  else
    LOCAL_FP="$(ssh-keygen -lf "$PUB_FILE" -E md5 | awk '{print $2}' | sed 's/^MD5://')"
    KEY_FP="$(doctl compute ssh-key list --format Name,FingerPrint --no-header | awk -v fp="$LOCAL_FP" 'tolower($2)==tolower(fp) {print $2; exit}')"
    if [[ -n "$KEY_FP" ]]; then
      echo "[OK] SSH key already registered with DO (matched by fingerprint)."
    else
      if ! doctl compute ssh-key import "$KEY_NAME" --public-key-file "$PUB_FILE" >/dev/null; then
        echo "[ERROR] Failed to import SSH key to DO." >&2
        exit 1
      fi
      echo "[OK] SSH key registered with DO."
      KEY_FP="$(doctl compute ssh-key list --format Name,FingerPrint --no-header | awk -v name="$KEY_NAME" 'tolower($1)==tolower(name) {print $2; exit}')"
    fi
  fi

  # -- Step 2: Check existing shared droplet or create small droplet -
  echo ""
  echo "[2/4] Checking droplet \"${DROPLET_NAME}\"..."
  if doctl compute droplet get "$DROPLET_NAME" >/dev/null 2>&1; then
    echo "[OK] Reusing existing droplet: ${DROPLET_NAME}"
  else
    USERDATA_TMP="$(mktemp "${TMPDIR:-/tmp}/do-demo-userdata-XXXXXX.sh")"
    tr -d '\r' < "${DEPLOY_DIR}/../../common/userdata-demo.sh" > "$USERDATA_TMP"

    if ! doctl compute droplet create "$DROPLET_NAME" \
        --region "$REGION" \
        --size "$SIZE" \
        --image "$IMAGE" \
        --ssh-keys "$KEY_FP" \
        --user-data-file "$USERDATA_TMP" \
        --wait; then
      rm -f "$USERDATA_TMP"
      echo "[ERROR] Droplet creation failed." >&2
      exit 1
    fi
    rm -f "$USERDATA_TMP"
    echo "[OK] Small droplet created: ${DROPLET_NAME} (${SIZE})"
    echo ""
    echo "[INFO] Waiting 40s for new droplet cloud-init & Docker daemon..."
    sleep 40
  fi

  # -- Step 3: Resolve public IP ------------------------------------
  echo ""
  echo "[3/4] Resolving public IP..."
  PUBLIC_IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header | head -n1 | tr -d '[:space:]')"
  if [[ -z "$PUBLIC_IP" ]]; then
    echo "[ERROR] Could not resolve droplet public IP." >&2
    exit 1
  fi
  echo "[OK] Public IP: ${PUBLIC_IP}"
  printf '%s\n' "$PUBLIC_IP" > "${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"
fi

# -- Step 4: Sync and start isolated demo container -----------------
echo ""
echo "[4/4] Syncing and starting isolated demo container..."
if ! bash "${DEPLOY_DIR}/sync.sh" --from-deploy; then
  echo "[WARN] Initial sync failed - if droplet is newly created, wait 30s and run:" >&2
  echo "       deploy/linux/demo/sync.sh --restart" >&2
  exit 1
fi

echo ""
echo "==============================================================="
echo "   QUICK_PP LIVE DEMO DEPLOY COMPLETE"
echo "==============================================================="
echo "   Droplet:     ${DROPLET_NAME}"
echo "   Public IP:   ${PUBLIC_IP}"
echo "   Remote Dir:  ${REMOTE_DIR}"
echo "   Host Port:   ${DEMO_BIND:-127.0.0.1}:${DEMO_PORT:-7860} -> container :7860"
echo "   SSH:         ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP}"
echo "==============================================================="
