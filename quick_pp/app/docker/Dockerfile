# Use an official Python runtime as a parent image
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache-dir quick_pp gunicorn

# Copy the rest of the repository into the image
COPY . /code

# Make the backend entrypoint executable (if present)
# The repository is copied with a top-level `quick_pp` folder when building from
# the workspace root, so the script lives at /code/quick_pp/app/docker/backend-entrypoint.sh
RUN chmod +x /code/quick_pp/app/docker/backend-entrypoint.sh || true

# Expose the backend API port (application listens on 6312)
EXPOSE 6312