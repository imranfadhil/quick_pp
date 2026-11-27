#!/usr/bin/env bash
# ===================================================================
#  DigitalOcean / Shared Droplet deployment config for quick_pp DEMO
#  Copy this file to  config.sh  (or run ./seed-config.sh).
#  config.sh is gitignored so your settings stay local.
#
#  Supports BOTH:
#    1. Deploying onto an EXISTING shared droplet (set DROPLET_NAME to
#       your existing droplet name, or set SHARED_DROPLET_IP directly).
#    2. Creating a dedicated small droplet (s-1vcpu-1gb or s-1vcpu-2gb).
# ===================================================================

# ---- Project & Isolated Remote Path ----
PROJECT_NAME="quick-pp-demo"
COMPOSE_PROJECT="quick_pp_demo"
# Isolated directory so it never collides with /opt/quick_pp or other apps
REMOTE_DIR="/opt/quick_pp_demo"

# ---- Droplet ----
# Set DROPLET_NAME to an existing droplet to share it, or a new name to create one.
DROPLET_NAME="quick-pp-server"
# Optional: If deploying to a non-DO server or by IP directly, set SHARED_DROPLET_IP.
SHARED_DROPLET_IP=""

# Region: sgp1 = Singapore
REGION="sgp1"
# Size: Because the demo uses read-only SQLite + pre-trained joblib models
# (no Postgres or Celery), it runs comfortably on a small droplet with 2 GB swap:
#   s-1vcpu-1gb  (~$6/mo)  - minimum budget / shared micro droplet
#   s-1vcpu-2gb  (~$12/mo) - recommended small droplet for fast Docker builds
SIZE="s-1vcpu-2gb"
IMAGE="docker-20-04"

# ---- SSH Key ----
# Reuses your main quick-pp-do-key by default in ~/.ssh (ext4) so OpenSSH 0600 perms work.
KEY_NAME="quick-pp-do-key"
KEY_FILE="${HOME}/.ssh/${KEY_NAME}"
PUB_FILE="${HOME}/.ssh/${KEY_NAME}.pub"

# ---- Demo Networking, Resource Limits & Cloudflare Tunnel ----
# Host port on the droplet (7860 avoids colliding with 80, 443, or 8080)
DEMO_PORT="7860"
# Bind to 127.0.0.1 when using Cloudflare Tunnel or host Nginx reverse proxy.
# Set to 0.0.0.0 if you want http://<PUBLIC_IP>:7860 accessible directly.
DEMO_BIND="127.0.0.1"
# Hard container memory & CPU limits for qpp-demo-app (protects shared droplets from OOM/starvation)
DEMO_MEM_LIMIT="2g"
DEMO_CPU_LIMIT="0.85"
# Concurrent active visitor cap, lag-indicator threshold, and browser idle pause (seconds)
DEMO_MAX_USERS="6"
DEMO_LAG_WARNING_USERS="3"
DEMO_IDLE_TIMEOUT_SECONDS="600"
DEMO_MAX_CONCURRENT_COMPUTE="2"
# Custom domain for the public demo in Cloudflare (e.g. demo.example.com)
DOMAIN=""
# Cloudflare Zero Trust Named Tunnel token (Networks -> Tunnels -> <tunnel> -> Token).
# Leave blank to fall back to a random *.trycloudflare.com Quick Tunnel.
CF_TUNNEL_TOKEN=""
