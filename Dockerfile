# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# The workflow will place PerTurbo/ alongside this Dockerfile
COPY . /app/PerTurbo
# (optional) avoid copying .git via .dockerignore
RUN pip install /app/PerTurbo

CMD ["python3"]
