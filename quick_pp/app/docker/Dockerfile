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

# Copy only requirements first to leverage Docker layer caching and ensure
# the file is present for pip install even if the full context copy has issues.
COPY requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN uv pip install --system --no-cache-dir -r /code/requirements.txt

# Copy the rest of the repository into the image
COPY . /code

# Make the backend entrypoint executable (if present)
# The repository is copied with a top-level `quick_pp` folder when building from
# the workspace root, so the script lives at /code/quick_pp/app/docker/backend-entrypoint.sh
RUN chmod +x /code/quick_pp/app/docker/backend-entrypoint.sh || true

# Expose the backend API port (application listens on 6312)
EXPOSE 6312

# Entrypoint will wait for Postgres, run DB init, then start the app
ENTRYPOINT ["/bin/sh", "/code/quick_pp/app/docker/backend-entrypoint.sh"]

# Fallback command (kept for clarity)
CMD ["python", "main.py", "app"]