# Deploying `quick_pp` UI & Live Demo to DigitalOcean (`doctl`)

Provisioning and deployment scripts for hosting both the **Private `quick_pp` UI** and the **Public Interactive Live Demo** on DigitalOcean Droplets with Cloudflare Tunnels.

## Directory Layout

```text
deploy/
├── common/                  # Remote helper scripts executed on the Droplet (cloud-init & .env config)
│   ├── configure-env.sh
│   ├── userdata.sh
│   └── userdata-demo.sh
├── demo/                    # Shared Live Demo Docker stack, Nginx config & Volve/Costa seeder
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── nginx.conf
│   ├── seed_demo.py
│   └── start_demo.sh
├── linux/                   # Ubuntu / Linux host runner scripts (.sh)
│   ├── config.example.sh / seed-config.sh
│   ├── deploy.sh / sync.sh / destroy.sh / setup-cloudflare.sh
│   └── demo/                # Demo host runner scripts (.sh)
└── windows/                 # Windows host runner scripts (.bat / .ps1)
    ├── config.example.bat / seed-config.bat / seed-config.ps1
    ├── deploy.bat / sync.bat / destroy.bat / setup-cloudflare.bat / setup-cloudflare.ps1
    └── demo/                # Demo host runner scripts (.bat / .ps1)
```

## Architecture on the Droplet (Main Stack)

The Docker Compose stack ([`quick_pp/app/docker/docker-compose.yaml`](../quick_pp/app/docker/docker-compose.yaml)) runs:
- **`web` (Nginx)**: Unified reverse proxy on port `8080` (and internal `:80`). Routes `/quick_pp/*` to `qpp-backend:6312` and `/` to `qpp-frontend:5469` with a `100 MB` upload limit for LAS files.
- **`quicktunnel` (Cloudflare Tunnel)**: Runs `cloudflared` outbound to Cloudflare's edge. Supports a permanent **Named Tunnel** (`https://<DOMAIN>` + Cloudflare Zero Trust Access when `CF_TUNNEL_TOKEN` is set in `config.sh` / `config.bat`), or falls back to a free `https://<words>.trycloudflare.com` Quick Tunnel when `CF_TUNNEL_TOKEN` is blank.
- **`qpp-frontend` (SvelteKit / Node 20)**: Builds and serves the SvelteKit UI on `:5469` using relative API paths (`VITE_BACKEND_URL=""`).
- **`qpp-backend` (FastAPI / Gunicorn)**: Serves the petrophysical API on `:6312` with in-memory caching for merged well logs.
- **`postgres` (pgvector/pg17)**: PostgreSQL 17 database.

*(For the isolated Live Demo stack architecture, see [`deploy/demo/README.md`](./demo/README.md).)*

## Prerequisites

- `doctl` installed and authenticated: `doctl auth init`
  - **Ubuntu (Snap)**:
    ```bash
    sudo snap install doctl
    sudo snap connect doctl:ssh-keys
    doctl auth init
    ```
  - **Other platforms**: https://docs.digitalocean.com/reference/doctl/how-to/install/
- OpenSSH (`ssh`, `ssh-keygen`) and `tar` on `PATH` (`python3` on Ubuntu; PowerShell on Windows)

## Recommended Droplet Size

| Tier | Slug | Specs | Cost | Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **Recommended** | `s-2vcpu-4gb` | 2 vCPU, 4 GB RAM, 80 GB SSD | ~\$24/mo | **Smooth & responsive UI** for multi-well projects, concurrent LAS parsing, and hosting the Live Demo alongside the main stack. |
| **Minimum / Budget** | `s-1vcpu-2gb` | 1 vCPU, 2 GB RAM, 50 GB SSD | ~\$12/mo | Works for single-user / light well evaluation (relies on the 2 GB swap file created by `common/userdata.sh`). |

## Setup & Typical Workflow

### Ubuntu / Linux (`deploy/linux/`)

1. **Generate `deploy/linux/config.sh`** (pre-filled from `../../.env` if present):
   ```bash
   cd deploy/linux
   ./seed-config.sh
   ```
2. **Review `deploy/linux/config.sh`** (`config.sh` is gitignored).
3. **Provision & Deploy**:
   ```bash
   # First time (creates or reuses the droplet + pushes code + starts containers)
   ./deploy.sh

   # After changing frontend or backend code
   ./sync.sh

   # Force a full Docker image rebuild (e.g. after changing pyproject.toml or Dockerfile)
   ./sync.sh --restart

   # Provision/update Cloudflare Tunnel routes, DNS, Zero Trust Access & WAF rules
   ./setup-cloudflare.sh

   # Tear everything down when done
   ./destroy.sh
   ```

### Windows (`deploy\windows\`)

1. **Generate `deploy\windows\config.bat`** (pre-filled from `..\..\.env` if present):
   ```bat
   cd deploy\windows
   seed-config.bat
   ```
2. **Review `deploy\windows\config.bat`** (`config.bat` is gitignored).
3. **Provision & Deploy**:
   ```bat
   deploy.bat
   sync.bat
   sync.bat --restart
   setup-cloudflare.bat
   destroy.bat
   ```
