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
#
# ORDER MATTERS. uv points the venv at the base interpreter's *unversioned*
# name -- .venv/bin/python -> /usr/local/bin/python3, .venv/bin/python3 ->
# ./python -- so symlinking /usr/local/bin/python3 back at the venv closes a
# cycle:
#   /usr/local/bin/python3 -> .venv/bin/python3 -> .venv/bin/python -> /usr/local/bin/python3
# Every entry point shebanged `#!/app/.venv/bin/python` then dies at exec() with
# ELOOP ("too many levels of symbolic links") before Python ever starts: no
# output, no traceback. Re-point the venv at the VERSIONED real binary
# (/usr/local/bin/python3.11, which nothing below rewrites) so the unversioned
# names can be repointed at the venv without forming a ring.
RUN BASE_PYTHON="$(readlink -f /usr/local/bin/python3)" \
    && echo "base interpreter: ${BASE_PYTHON}" \
    && ln -sfn "${BASE_PYTHON}" /app/.venv/bin/python \
    && ln -sfn python /app/.venv/bin/python3 \
    && ln -sfn python /app/.venv/bin/python3.11 \
    && ln -sfn /app/.venv/bin/perturbo /usr/local/bin/perturbo \
    && ln -sfn /app/.venv/bin/python /usr/local/bin/python \
    && ln -sfn /app/.venv/bin/python3 /usr/local/bin/python3

# Fail the BUILD, not a downstream pipeline run, if the links above ever loop
# again or the venv's site-packages stop being visible. `sys.prefix` must be the
# venv (not the base prefix) -- that is what proves the /usr/local/bin hop kept
# venv semantics rather than silently falling back to the bare interpreter.
RUN set -eu \
    && for exe in /usr/local/bin/python /usr/local/bin/python3 /app/.venv/bin/python; do \
         readlink -f "$exe" >/dev/null || { echo "FATAL: $exe does not resolve (symlink loop?)"; exit 1; }; \
       done \
    && python -c 'import sys; assert sys.prefix == "/app/.venv", f"venv not active: sys.prefix={sys.prefix}"' \
    && python -c 'import perturbo; print("perturbo", getattr(perturbo, "__version__", "?"))' \
    && perturbo --help >/dev/null \
    && echo "OK: entry points resolve and the venv is active"

CMD ["perturbo", "--help"]
