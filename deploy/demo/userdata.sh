#!/bin/bash
# DigitalOcean cloud-init - quick_pp Live Demo
# Prepares /opt/quick_pp_demo and ensures a 2 GB swap file is active so
# small 1 GB / 2 GB droplets never hit OOM during multi-stage Docker builds.

set -e -x

mkdir -p /opt/quick_pp_demo

# Create a 2 GB swap file if none exists (critical for s-1vcpu-1gb and s-1vcpu-2gb)
if ! swapon --show | grep -q '/swapfile'; then
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  if ! grep -q '/swapfile' /etc/fstab; then
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
  fi
fi

# Ensure the compose plugin is available as `docker compose`.
if ! docker compose version >/dev/null 2>&1; then
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

echo "Droplet ready for quick_pp_demo sync."

