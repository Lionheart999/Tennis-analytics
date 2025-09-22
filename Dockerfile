# Choose base with --build-arg FLAVOR=cuda or FLAVOR=cpu
ARG FLAVOR=cuda

# ------- CUDA (GPU) base -------
FROM nvcr.io/nvidia/pytorch:24.02-py3 AS cuda

# ------- CPU base -------
FROM python:3.10-slim AS cpu

# ------- Final stage switches by FLAVOR -------
FROM ${FLAVOR}
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget unzip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
ARG FLAVOR
# If CPU build, install CPU torch explicitly
RUN if [ "$FLAVOR" = "cpu" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ; \
    fi && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

