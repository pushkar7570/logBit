# Force Ubuntu base
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Minimal extras (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git jq less unzip \
 && rm -rf /var/lib/apt/lists/*
