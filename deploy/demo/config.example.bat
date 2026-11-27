@echo off
REM ===================================================================
REM  DigitalOcean / Shared Droplet deployment config for quick_pp DEMO
REM  Copy this file to  config.bat  (or run seed-config.bat).
REM  config.bat is gitignored so your settings stay local.
REM
REM  Supports BOTH:
REM    1. Deploying onto an EXISTING shared droplet (set DROPLET_NAME to
REM       your existing droplet name, or set SHARED_DROPLET_IP directly).
REM    2. Creating a dedicated small droplet (s-1vcpu-1gb or s-1vcpu-2gb).
REM ===================================================================

REM ---- Project & Isolated Remote Path ----
set PROJECT_NAME=quick-pp-demo
set COMPOSE_PROJECT=quick_pp_demo
REM Isolated directory so it never collides with /opt/quick_pp or other apps
set REMOTE_DIR=/opt/quick_pp_demo

REM ---- Droplet ----
REM Set DROPLET_NAME to an existing droplet to share it, or a new name to create one.
set DROPLET_NAME=quick-pp-server
REM Optional: If deploying to a non-DO server or by IP directly, set SHARED_DROPLET_IP.
set SHARED_DROPLET_IP=

REM Region: sgp1 = Singapore
set REGION=sgp1
REM Size: Because the demo uses read-only SQLite + pre-trained joblib models
REM (no Postgres or Celery), it runs comfortably on a small droplet with 2 GB swap:
REM   s-1vcpu-1gb  (~$6/mo)  - minimum budget / shared micro droplet
REM   s-1vcpu-2gb  (~$12/mo) - recommended small droplet for fast Docker builds
set SIZE=s-1vcpu-2gb
set IMAGE=docker-20-04

REM ---- SSH Key ----
REM Reuses your main quick-pp-do-key by default so you don't need a second key.
set KEY_NAME=quick-pp-do-key
set KEY_FILE=%USERPROFILE%\.ssh\%KEY_NAME%
set PUB_FILE=%USERPROFILE%\.ssh\%KEY_NAME%.pub

REM ---- Demo Networking, Resource Limits & Cloudflare Tunnel ----
REM Host port on the droplet (7860 avoids colliding with 80, 443, or 8080)
set DEMO_PORT=7860
REM Bind to 127.0.0.1 when using Cloudflare Tunnel or host Nginx reverse proxy.
REM Set to 0.0.0.0 if you want http://<PUBLIC_IP>:7860 accessible directly.
set DEMO_BIND=127.0.0.1
REM Hard container memory & CPU limits for qpp-demo-app (protects shared droplets from OOM/starvation)
set DEMO_MEM_LIMIT=2g
set DEMO_CPU_LIMIT=0.85
REM Concurrent active visitor cap, lag-indicator threshold, and browser idle pause (seconds)
set DEMO_MAX_USERS=6
set DEMO_LAG_WARNING_USERS=3
set DEMO_IDLE_TIMEOUT_SECONDS=600
set DEMO_MAX_CONCURRENT_COMPUTE=2
REM Custom domain for the public demo in Cloudflare (e.g. demo.example.com)
set DOMAIN=
REM Cloudflare Zero Trust Named Tunnel token (Networks -> Tunnels -> <tunnel> -> Token).
REM Leave blank to fall back to a random *.trycloudflare.com Quick Tunnel.
set CF_TUNNEL_TOKEN=

