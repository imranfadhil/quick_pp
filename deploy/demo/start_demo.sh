#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo "  Starting quick_pp Interactive Live Demo"
echo "  Port: 7860 | Mode: QPP_DEMO_MODE=true (Read-Only Seed DB + ML)"
echo "=================================================================="

mkdir -p /code/data /tmp/uploads

# Restore clean pre-seeded Volve + Costa SQLite database on every container start
if [ -f /code/data/seed/quick_pp_seed.db ]; then
    rm -f /code/data/quick_pp.db
    cp /code/data/seed/quick_pp_seed.db /code/data/quick_pp.db
    chmod 444 /code/data/quick_pp.db
    echo "[seed] Restored read-only pre-seeded Volve & Costa database (/code/data/quick_pp.db)"
fi

export QPP_DEMO_MODE=true
export QPP_DISABLE_TRAINING=true
export QPP_DATABASE_URL="sqlite:////code/data/quick_pp.db"
export QPP_PRETRAINED_MODELS_PATH="/code/data/seed/pretrained_models.joblib"

# 1. Launch FastAPI backend on 127.0.0.1:6312
uvicorn quick_pp.app.backend.main:app --host 127.0.0.1 --port 6312 --workers 1 &
BACKEND_PID=$!

# 2. Launch SvelteKit Node adapter build on 127.0.0.1:5469
HOST=127.0.0.1 PORT=5469 BODY_SIZE_LIMIT=5242880 node /code/quick_pp/app/frontend/build &
FRONTEND_PID=$!

# 3. Launch Nginx reverse proxy in foreground on port 7860
nginx -c /etc/nginx/nginx.demo.conf -g "daemon off;" &
NGINX_PID=$!

trap "kill -TERM $BACKEND_PID $FRONTEND_PID $NGINX_PID 2>/dev/null || true" SIGINT SIGTERM

wait -n $BACKEND_PID $FRONTEND_PID $NGINX_PID
exit $?

