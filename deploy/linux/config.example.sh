#!/usr/bin/env bash
# ===================================================================
#  DigitalOcean deployment config for quick_pp UI (Linux / Ubuntu)
#  Copy this file to  config.sh  (or run ./seed-config.sh) and fill in values.
#  config.sh is gitignored so your secrets stay local.
#
#  Prereq: doctl installed & authenticated  ->  doctl auth init
# ===================================================================

# ---- Project ----
PROJECT_NAME="quick-pp"
# Directory on the droplet where the app lives
REMOTE_DIR="/opt/quick_pp"

# ---- Droplet ----
DROPLET_NAME="${PROJECT_NAME}-server"
# Region: sgp1 = Singapore (lowest DO latency to Malaysia).
REGION="sgp1"
# Size: s-2vcpu-4gb (~$24/mo, hourly-billed). 4 GB RAM + 2 GB swap gives
# smooth headroom for Postgres + FastAPI + SvelteKit build.
# Use s-1vcpu-2gb (~$12/mo) for light single-well usage (relies on 2 GB swap).
SIZE="s-2vcpu-4gb"
# Docker 1-click marketplace image on Ubuntu (Docker + Compose preinstalled).
IMAGE="docker-20-04"

# ---- SSH Key ----
# Stored in ~/.ssh (ext4) so OpenSSH strict 0600 permissions work even if the repo is on NTFS.
KEY_NAME="${PROJECT_NAME}-do-key"
KEY_FILE="${HOME}/.ssh/${KEY_NAME}"
PUB_FILE="${HOME}/.ssh/${KEY_NAME}.pub"

# ---- App & Cloudflare Tunnel config ----
# Your custom domain in Cloudflare (e.g. app.example.com). Leave blank for Quick Tunnel.
DOMAIN=""
# Cloudflare Zero Trust Named Tunnel token (Networks -> Tunnels -> <tunnel> -> Token).
# Leave blank to fall back to a random *.trycloudflare.com Quick Tunnel.
CF_TUNNEL_TOKEN=""
# Optional email allowed by Cloudflare Zero Trust Access (used by setup-cloudflare.sh)
ALLOWED_EMAIL=""

# Host port where Nginx unified reverse proxy listens on the droplet
WEB_PORT="8080"

# Database settings (auto-generates strong password on first deploy if left blank/xxx)
POSTGRES_USER="qpp_user"
POSTGRES_DB="quick_pp"
POSTGRES_PASSWORD=""

TAVILY_API_KEY=""
