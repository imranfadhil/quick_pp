@echo off
REM ===================================================================
REM  DigitalOcean deployment config for quick_pp UI
REM  Copy this file to  config.bat  and fill in your values.
REM  config.bat is gitignored so your secrets stay local.
REM
REM  Prereq: doctl installed & authenticated  ->  doctl auth init
REM ===================================================================

REM ---- Project ----
set PROJECT_NAME=quick-pp
REM Directory on the droplet where the app lives
set REMOTE_DIR=/opt/quick_pp

REM ---- Droplet ----
set DROPLET_NAME=%PROJECT_NAME%-server
REM Region: sgp1 = Singapore (lowest DO latency to Malaysia).
set REGION=sgp1
REM Size: s-2vcpu-4gb (~$24/mo, hourly-billed). 4 GB RAM + 2 GB swap gives
REM smooth headroom for Postgres + FastAPI + SvelteKit build.
REM Use s-1vcpu-2gb (~$12/mo) for light single-well usage (relies on 2 GB swap).
set SIZE=s-2vcpu-4gb
REM Docker 1-click marketplace image on Ubuntu (Docker + Compose preinstalled).
set IMAGE=docker-20-04

REM ---- SSH Key ----
REM A key of this name is registered with DO (from your local public key).
set KEY_NAME=%PROJECT_NAME%-do-key
set KEY_FILE=%USERPROFILE%\.ssh\%KEY_NAME%
set PUB_FILE=%USERPROFILE%\.ssh\%KEY_NAME%.pub

REM ---- App config ----
REM Your public domain (e.g. qpp.example.com). Leave blank if using Quick Tunnel / IP.
set DOMAIN=

REM Host port where Nginx unified reverse proxy listens on the droplet
set WEB_PORT=8080

REM Database settings (auto-generates strong password on first deploy if left blank/xxx)
set POSTGRES_USER=qpp_user
set POSTGRES_DB=quick_pp
set POSTGRES_PASSWORD=

set TAVILY_API_KEY=
