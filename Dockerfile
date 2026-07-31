# JAX's CUDA 12 extra carries the CUDA/cuDNN user-space libraries as Python
# wheels. The host's NVIDIA driver is supplied at runtime by the NVIDIA
# Container Toolkit, so a PyTorch or system-CUDA base image is unnecessary.
FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

# procps supplies `ps`, which Nextflow's task wrapper shells out to in order to
# collect task metrics. Without it the wrapper reports "Command 'ps' required by
# nextflow to collect task metrics cannot be found" and the task dies BEFORE the
# process script runs -- exit 1, empty stdout, no traceback, which reads exactly
# like a silent crash in the payload. nf-core requires procps in every container
# for this reason. python:3.11-slim omits it; the pytorch base image this
# Dockerfile replaced happened to include it, so the loss went unnoticed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# uv lives OUTSIDE /usr/local so the sync below (which prunes its target
# environment to match the lock) cannot delete the tool doing the pruning.
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /usr/bin/uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# Install into the image's SYSTEM prefix (/usr/local) rather than a project
# venv. A container is already an isolation boundary, so a venv only adds a
# second prefix that has to be *activated* -- and every way of activating it
# here is broken under Singularity/Nextflow:
#
#   * ENV PATH=/app/.venv/bin:...  Docker bakes an absolute PATH into the image
#     config, which Singularity uses to REPLACE the environment Nextflow
#     injects. Nextflow exposes a pipeline's bin/ scripts by prepending bin/ to
#     PATH, so the override drops them and they fail with "command not found".
#   * symlink /usr/local/bin/python -> /app/.venv/bin/python.  CPython locates
#     pyvenv.cfg relative to the path it was INVOKED as, so it searches
#     /usr/local, finds nothing, and silently runs as the BASE interpreter with
#     none of the dependencies (sys.prefix=/usr/local). It also loops if
#     /usr/local/bin/python3 is repointed, since uv aims the venv at that name.
#   * wrapper script at /usr/local/bin/python.  Linux does not allow a script
#     to serve as a shebang interpreter, so the pipeline's ~20 bin/*.py scripts
#     (`#!/usr/bin/env python`) would die with ENOEXEC.
#
# With a system install there is nothing to activate: `python`, `python3`,
# `#!/usr/bin/env python`, and `perturbo` all resolve to the one interpreter
# that owns the dependencies, whatever PATH the runtime hands us.
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --locked --no-dev --no-editable --extra cuda

# Fail the BUILD, not a pipeline run six hours into a cluster job, if the
# install ever stops being reachable from a bare PATH. The `env python` check
# is the one that matters most: that is how the pipeline's bin/*.py scripts
# actually enter Python, and it is what the venv layouts kept breaking.
RUN set -eu \
    && test ! -e /app/.venv || { echo "FATAL: a venv exists; deps must be in the system prefix"; exit 1; } \
    && command -v ps >/dev/null || { echo "FATAL: no ps; Nextflow's task wrapper needs it"; exit 1; } \
    && python  -c 'import sys, perturbo; print("python  ->", sys.executable, "| perturbo", getattr(perturbo, "__version__", "?"))' \
    && python3 -c 'import perturbo' \
    && /usr/bin/env python  -c 'import perturbo' \
    && /usr/bin/env python3 -c 'import perturbo' \
    && perturbo --help >/dev/null \
    && echo "OK: system install reachable as python / python3 / env python / perturbo"

CMD ["perturbo", "--help"]
