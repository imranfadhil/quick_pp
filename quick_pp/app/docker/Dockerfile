# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install quick_pp and gunicorn, then verify installation
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir --upgrade quick_pp gunicorn

# Set the default command to run the backend
CMD ["quick_pp", "backend", "--no-open"]

# Expose the backend API port (application listens on 6312)
EXPOSE 6312