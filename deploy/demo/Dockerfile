# ==============================================================================
#  quick_pp Interactive Live Demo — Self-Contained Full-Stack Dockerfile
# ==============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY quick_pp/app/frontend/package*.json ./
RUN npm install --no-audit --no-fund

COPY quick_pp/app/frontend/ ./
ENV VITE_BACKEND_URL=""
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        nginx \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /code

COPY pyproject.toml README.md ./
COPY quick_pp ./quick_pp
COPY deploy/demo/seed_demo.py ./deploy/demo/seed_demo.py
COPY notebooks/data/01_raw/VOLVE/15-9-19-A.las ./notebooks/data/01_raw/VOLVE/15-9-19-A.las
COPY notebooks/data/01_raw/VOLVE/VOLVE_15-9_Surveys.csv ./notebooks/data/01_raw/VOLVE/VOLVE_15-9_Surveys.csv
COPY notebooks/data/01_raw/COSTA/HW-3.las ./notebooks/data/01_raw/COSTA/HW-3.las
COPY notebooks/data/01_raw/COSTA/HW-4.las ./notebooks/data/01_raw/COSTA/HW-4.las
COPY notebooks/data/01_raw/COSTA/HW-5.las ./notebooks/data/01_raw/COSTA/HW-5.las
COPY notebooks/data/01_raw/COSTA/HW_core_data_all.csv ./notebooks/data/01_raw/COSTA/HW_core_data_all.csv
COPY notebooks/data/01_raw/36_7-3.las ./notebooks/data/01_raw/36_7-3.las
COPY notebooks/data/01_raw/formation_pressure_data.csv ./notebooks/data/01_raw/formation_pressure_data.csv

RUN uv venv --python 3.11 /venv && \
    . /venv/bin/activate && \
    uv pip install --no-cache-dir ".[app,ml]"

ENV PATH="/venv/bin:$PATH"

COPY --from=frontend-builder /app/frontend/build /code/quick_pp/app/frontend/build
COPY --from=frontend-builder /app/frontend/package.json /code/quick_pp/app/frontend/package.json
COPY --from=frontend-builder /app/frontend/node_modules /code/quick_pp/app/frontend/node_modules

COPY deploy/demo/nginx.conf /etc/nginx/nginx.demo.conf
COPY deploy/demo/start_demo.sh /usr/local/bin/start_demo.sh
RUN sed -i 's/\r$//' /usr/local/bin/start_demo.sh && chmod +x /usr/local/bin/start_demo.sh

RUN mkdir -p /code/data/seed /code/uploads && \
    QPP_DATABASE_URL="sqlite:////code/data/seed/quick_pp_seed.db" \
    QPP_PRETRAINED_MODELS_PATH="/code/data/seed/pretrained_models.joblib" \
    python /code/deploy/demo/seed_demo.py && \
    cp /code/data/seed/quick_pp_seed.db /code/data/quick_pp.db && \
    chmod 444 /code/data/seed/quick_pp_seed.db /code/data/quick_pp.db && \
    chown -R user:user /code /tmp

USER user

ENV HOME=/home/user \
    QPP_DEMO_MODE=true \
    QPP_DISABLE_TRAINING=true \
    QPP_DATABASE_URL="sqlite:////code/data/quick_pp.db" \
    QPP_PRETRAINED_MODELS_PATH="/code/data/seed/pretrained_models.joblib" \
    PORT=7860

EXPOSE 7860

CMD ["/usr/local/bin/start_demo.sh"]
