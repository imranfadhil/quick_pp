#!/usr/bin/env bash
# ===================================================================
#  deploy.sh - Provision quick_pp UI on a DigitalOcean Droplet (Linux)
#  Creates the droplet with doctl (no manual IP needed), then pushes
#  the code and starts the Docker stack.
#
#  Prereqs: doctl (authenticated), ssh + ssh-keygen (OpenSSH), tar
# ===================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
  echo "[deploy] config.sh not found - seeding from .env via config.example.sh"
  bash "${DEPLOY_DIR}/seed-config.sh"
  if [[ ! -f "${DEPLOY_DIR}/config.sh" ]]; then
    echo "[deploy] Could not create config.sh. Edit deploy/linux/config.sh, then re-run." >&2
    exit 1
  fi
  echo "[deploy] Review deploy/linux/config.sh (check SIZE, REGION, DOMAIN), then re-run ./deploy.sh."
  exit 1
fi

# shellcheck disable=SC1090
source <(tr -d '\r' < "${DEPLOY_DIR}/config.sh")

echo ""
echo "==============================================================="
echo "   QUICK_PP UI - DIGITALOCEAN DEPLOY"
echo "==============================================================="
echo ""

# -- Prerequisites --------------------------------------------------
if ! command -v doctl >/dev/null 2>&1; then
  echo "[ERROR] doctl not found. Install on Ubuntu:" >&2
  echo "        sudo snap install doctl && sudo snap connect doctl:ssh-keys" >&2
  echo "        (or see https://docs.digitalocean.com/reference/doctl/how-to/install/)" >&2
  exit 1
fi

if ! doctl account get >/dev/null 2>&1; then
  echo "[ERROR] doctl not authenticated. Run: doctl auth init" >&2
  exit 1
fi
echo "[OK] doctl authenticated."

# -- Step 1: Ensure a local SSH key + register it with DO -----------
echo ""
echo "[1/5] Ensuring SSH key..."
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

# Register (import) the public key with DO if not already there.
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
      echo "        The key may already exist under a different name." >&2
      echo "        Check: doctl compute ssh-key list" >&2
      exit 1
    fi
    echo "[OK] SSH key registered with DO."
    KEY_FP="$(doctl compute ssh-key list --format Name,FingerPrint --no-header | awk -v name="$KEY_NAME" 'tolower($1)==tolower(name) {print $2; exit}')"
  fi
fi

if [[ -z "$KEY_FP" ]]; then
  echo "[ERROR] Could not resolve SSH key fingerprint." >&2
  exit 1
fi

# -- Step 2: Create the droplet ------------------------------------
echo ""
echo "[2/5] Creating droplet..."
CREATED_NEW=0

if doctl compute droplet get "$DROPLET_NAME" >/dev/null 2>&1; then
  echo "[OK] Droplet already exists: $DROPLET_NAME"
else
  USERDATA_TMP="$(mktemp "${TMPDIR:-/tmp}/do-userdata-XXXXXX.sh")"
  tr -d '\r' < "${DEPLOY_DIR}/../common/userdata.sh" > "$USERDATA_TMP"

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
  CREATED_NEW=1
  echo "[OK] Droplet created: $DROPLET_NAME"
fi

# -- Step 3: Resolve public IP -------------------------------------
echo ""
echo "[3/5] Resolving public IP..."
PUBLIC_IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header | head -n1 | tr -d '[:space:]')"
if [[ -z "$PUBLIC_IP" ]]; then
  echo "[ERROR] Could not resolve droplet public IP." >&2
  exit 1
fi
echo "[OK] Public IP: $PUBLIC_IP"
printf '%s\n' "$PUBLIC_IP" > "${TMPDIR:-/tmp}/${PROJECT_NAME}-do-ip.txt"

# -- Step 4: Wait for Docker/boot to settle ------------------------
echo ""
if [[ "$CREATED_NEW" -eq 1 ]]; then
  echo "[4/5] Waiting 45s for newly created droplet to finish cloud-init & boot..."
  sleep 45
else
  echo "[4/5] Existing droplet detected - skipping initial boot wait."
fi

# -- Step 5: Push code and start the app ---------------------------
echo ""
echo "[5/5] Deploying code..."
if ! bash "${DEPLOY_DIR}/sync.sh" --from-deploy; then
  echo "[WARN] Initial sync failed - droplet may still be booting." >&2
  echo "       Wait a minute and run:  ./sync.sh --restart" >&2
fi

echo ""
echo "==============================================================="
echo "   DIGITALOCEAN DEPLOY COMPLETE"
echo "==============================================================="
echo "   Droplet:    ${DROPLET_NAME}  (${SIZE}, ${REGION})"
echo "   Public IP:  ${PUBLIC_IP}"
echo "   Key File:   ${KEY_FILE}"
echo ""
if [[ -n "${DOMAIN:-}" ]]; then
  echo "   App:        https://${DOMAIN}/"
  echo ""
fi
echo "   SSH:        ssh -i \"${KEY_FILE}\" root@${PUBLIC_IP}"
echo "==============================================================="
