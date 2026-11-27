# Deploying `quick_pp` Interactive Live Demo to a Shared or Small Droplet

Because Hugging Face Spaces now requires a paid plan for Docker SDK Spaces, this folder (`deploy/demo/`) provides one-command batch scripts to host the **Interactive Live Demo** on either:
1. **An existing shared DigitalOcean Droplet** (alongside other apps or your main stack, with zero port or container collisions), or
2. **A dedicated small DigitalOcean Droplet** (`s-1vcpu-1gb` ~$6/mo or `s-1vcpu-2gb` ~$12/mo).

---

## Why This Works on a Small / Shared Droplet

Unlike the full production stack (which runs PostgreSQL 17, persistent volumes, and live `npm run build` on container start), the **Live Demo stack** ([`deploy/demo/docker-compose.yaml`](./docker-compose.yaml)) is self-contained and lightweight:
- **No PostgreSQL or Celery containers**: Uses the pre-seeded read-only SQLite database (`quick_pp.db`, `chmod 444`) + pre-trained Conformal PIML bundle (`pretrained_models.joblib`).
- **Single Application Container (`qpp-demo-app`)**: Bundles Nginx (`:7860`), FastAPI (`127.0.0.1:6312`), and the pre-built SvelteKit Node server (`127.0.0.1:5469`). Steady-state RAM usage is **~220–280 MB**, with hard Docker resource ceilings (`DEMO_MEM_LIMIT=2g`, `DEMO_CPU_LIMIT=0.85`, `pids_limit=128`, `cap_drop=ALL`, 256 MB `/tmp` tmpfs, rotated `3×10m` container logs) and tiered Nginx rate limiting (`60 req/min` general, `10 req/min` heavy compute/LLM/PDF).
- **Live Occupancy, Idle Pause, & 3-Tier Storage Cleanup**:
  - Displays a real-time **online users badge** in the top bar (amber lag warning at `3+` concurrent visitors) and enforces a soft active-session gate (`DEMO_MAX_USERS=6`) plus a `2`-job heavy-compute semaphore.
  - Pauses inactive browser tabs (`10 min` idle / `5 min` hidden tab) to release capacity for other visitors.
  - Automatically purges committed log uploads immediately after parsing, removes session reports/staged files on tab exit, and runs a 5-minute TTL + ring-buffer sweep so disk usage stays bounded.
- **Complete Isolation on Shared Droplets**:
  - Remote folder: `/opt/quick_pp_demo` (never touches `/opt/quick_pp`)
  - Compose project: `quick_pp_demo` (`qpp-demo-app` & `qpp-demo-tunnel`)
  - Host port: `127.0.0.1:7860` (never collides with `:80`, `:443`, or `:8080`)
  - Automatic 2 GB `/swapfile` provisioning so multi-stage Docker builds never hit OOM on 1 GB / 2 GB droplets.
- **Custom Domain or Instant HTTPS Public Link**: Includes `demo-quicktunnel` (`cloudflared`) which connects to your permanent Cloudflare Named Tunnel (`https://<DOMAIN>`) when `CF_TUNNEL_TOKEN` is set in `deploy\demo\config.bat`, or prints a free `https://<words>.trycloudflare.com` Quick Tunnel URL when left blank.

---

## Quick Start

### 1. Generate `deploy\demo\config.bat`
```bat
cd deploy\demo
seed-config.bat
```

### 2. Configure Target Droplet in `deploy\demo\config.bat`
- **Option A — Share an existing droplet**:
  - Set `DROPLET_NAME=<your-existing-droplet-name>` (if managed via `doctl`), **OR**
  - Set `SHARED_DROPLET_IP=<your-droplet-ip>` and `KEY_FILE=<path-to-your-ssh-key>`.
- **Option B — Create a new small droplet**:
  - Leave `DROPLET_NAME=quick-pp-demo` and set `SIZE=s-1vcpu-2gb` (or `s-1vcpu-1gb`).

### 3. Deploy & Sync
```bat
REM First deploy (reuses existing droplet or creates one, builds demo image, and prints HTTPS URL)
deploy.bat

REM Push code updates and restart the demo container
sync.bat

REM Force a full Docker image & seed rebuild
sync.bat --restart
```

### 4. Safe Teardown (`destroy.bat`)
```bat
destroy.bat
```
Prompts you to choose:
- **`[1] Remove DEMO APP ONLY`** *(Recommended for shared droplets)* — Stops and removes only `quick_pp_demo` containers and `/opt/quick_pp_demo`, leaving your shared droplet and all other hosted apps untouched.
- **`[2] Destroy ENTIRE DROPLET`** — Deletes the droplet on DigitalOcean (with a foreign-container safety check).

