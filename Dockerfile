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

# Expose the venv's interpreter and console scripts on the default system PATH
# (/usr/local/bin) rather than overriding PATH with `ENV PATH=/app/.venv/bin:...`.
# Docker bakes such an override into a fixed absolute PATH in the image config,
# which Singularity/Apptainer then uses to REPLACE the environment Nextflow
# injects at runtime. Nextflow makes a pipeline's bin/ scripts available by
# prepending bin/ to PATH, so the override silently drops bin/ and wrapper
# scripts (e.g. perturbo_v2_pipeline_adapter.py) fail with "command not found".
# Symlinking into /usr/local/bin keeps bin/ injection intact while still making
# `perturbo`/`python` resolve to the venv (which owns all the dependencies).
RUN ln -sf /app/.venv/bin/perturbo /usr/local/bin/perturbo \
    && ln -sf /app/.venv/bin/python /usr/local/bin/python \
    && ln -sf /app/.venv/bin/python3 /usr/local/bin/python3

CMD ["perturbo", "--help"]
