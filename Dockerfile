# JAX's CUDA 12 extra carries the CUDA/cuDNN user-space libraries as Python
# wheels. The host's NVIDIA driver is supplied at runtime by the NVIDIA
# Container Toolkit, so a PyTorch or system-CUDA base image is unnecessary.
FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install uv==0.6.3

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --locked --no-dev --extra cuda

ENV PATH="/app/.venv/bin:$PATH"

CMD ["perturbo", "--help"]
