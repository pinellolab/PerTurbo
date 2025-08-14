# Use official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="logan-blaine"
LABEL description="Docker image for PerTurbo with PyTorch and CUDA 12.4"

WORKDIR /app

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Use build-time ARG with GitHub access token (only needed while private)
ARG GITHUB_TOKEN
RUN pip install git+https://logan-blaine:${GITHUB_TOKEN}@github.com/pinellolab/PerTurbo.git

# Default to Python shell
CMD ["python3"]
