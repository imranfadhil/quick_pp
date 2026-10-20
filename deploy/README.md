# Deploying `quick_pp` UI to DigitalOcean (`doctl`)

Batch scripts that provision and manage the `quick_pp` UI & Backend stack on a DigitalOcean Droplet entirely through **`doctl`** — no need to look up or paste an IP address. The scripts create the droplet, configure swap, resolve its public IP automatically, push the code, and can destroy everything when done.

## Architecture on the Droplet

The Docker Compose stack ([`quick_pp/app/docker/docker-compose.yaml`](../quick_pp/app/docker/docker-compose.yaml)) runs:
- **`web` (Nginx)**: Unified reverse proxy on port `8080` (and internal `:80`). Routes `/quick_pp/*` to `qpp-backend:6312` and `/` to `qpp-frontend:5469` with a `100 MB` upload limit for LAS files.
- **`quicktunnel` (Cloudflare)**: Runs `cloudflared tunnel --url http://web:80` and prints a free `https://<words>.trycloudflare.com` URL at the end of `sync.bat`.
- **`qpp-frontend` (SvelteKit / Node 20)**: Builds and serves the SvelteKit UI on `:5469` using relative API paths (`VITE_BACKEND_URL=""`).
- **`qpp-backend` (FastAPI / Gunicorn)**: Serves the petrophysical API on `:6312` with in-memory caching for merged well logs.
- **`postgres` (pgvector/pg17)**: PostgreSQL 17 database.

## Prerequisites

- `doctl` installed and authenticated: `doctl auth init`
  (install: https://docs.digitalocean.com/reference/doctl/how-to/install/)
- OpenSSH `ssh`, `scp`, `ssh-keygen` on PATH (built into Windows 10/11)
- `tar` (built into Windows 10/11)

## Recommended Droplet Size

| Tier | Slug | Specs | Cost | Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **Recommended** | `s-2vcpu-4gb` | 2 vCPU, 4 GB RAM, 80 GB SSD | ~\$24/mo | **Smooth & responsive UI** for multi-well projects, concurrent LAS parsing, and fast Vite builds. |
| **Minimum / Budget** | `s-1vcpu-2gb` | 1 vCPU, 2 GB RAM, 50 GB SSD | ~\$12/mo | Works for single-user / light well evaluation (relies on the 2 GB swap file created by `userdata.sh`). |
| **With Langflow** | `s-4vcpu-8gb` | 4 vCPU, 8 GB RAM, 160 GB SSD | ~\$48/mo | Recommended if you set `ENABLE_LANGFLOW=1`. |

## Setup & Typical Workflow

1. **Generate `config.bat`** (pre-filled from `..\.env` if present):
   ```bat
   cd deploy
   seed-config.bat
   ```
2. **Review `deploy\config.bat`** (adjust `SIZE`, `REGION`, or `DOMAIN` if desired). `config.bat` is gitignored.
3. **Provision & Deploy**:
   ```bat
   REM First time (creates the droplet + pushes code + starts containers)
   deploy.bat

   REM After changing frontend or backend code
   sync.bat

   REM Force a full Docker image rebuild (e.g. after changing pyproject.toml or Dockerfile)
   sync.bat --restart

   REM Tear everything down when done
   destroy.bat
   ```
