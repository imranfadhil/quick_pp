# Use an official Python runtime as a parent image
FROM python:3.11-slim

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

# Install the package itself
RUN pip install --no-cache-dir . gunicorn

# Set the default command to run the backend
CMD ["quick_pp", "backend", "--no-open"]

# Expose the backend API port (application listens on 6312)
EXPOSE 6312