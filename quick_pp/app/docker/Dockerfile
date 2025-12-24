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

COPY pyproject.toml .

# Copy the rest of the repository into the image
COPY . /code

# Create a virtual environment using uv and install dependencies inside it
RUN uv venv --python 3.11 /venv \
    && . /venv/bin/activate \
    && uv pip install --no-cache-dir . gunicorn

# Ensure the venv is used for all subsequent commands and at runtime
ENV PATH="/venv/bin:$PATH"

# Set the default command to run the backend
CMD ["quick_pp", "backend", "--no-open"]

# Expose the backend API port (application listens on 6312)
EXPOSE 6312