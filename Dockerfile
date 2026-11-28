# Multi-target Trackflow container (CPU, NVIDIA CUDA, NVIDIA TensorRT FP16).
# Build:
#   docker build --target cpu -t trackflow:cpu .
#   docker build --target cuda -t trackflow:cuda .
#   docker build --target tensorrt -t trackflow:tensorrt .
# Run (NVIDIA GPU, persisting Hugging Face weights and TensorRT engine caches):
#   docker run --rm --gpus all \
#     -v "$PWD:/work" \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     -v "$HOME/.cache/trackflow:/root/.cache/trackflow" \
#     trackflow:tensorrt run /work/street.mp4

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts/build_variants.py ./scripts/build_variants.py

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

FROM base AS cpu
RUN uv pip install --no-cache .
WORKDIR /work
ENTRYPOINT ["trackflow"]

FROM base AS cuda
RUN python scripts/build_variants.py --variants cuda --out-dir /tmp/dist \
    && uv pip install --no-cache /tmp/dist/trackflow_python_cuda-*.whl \
    && rm -rf /tmp/dist
WORKDIR /work
ENTRYPOINT ["trackflow"]

FROM base AS tensorrt
RUN python scripts/build_variants.py --variants tensorrt --out-dir /tmp/dist \
    && uv pip install --no-cache /tmp/dist/trackflow_python_tensorrt-*.whl \
    && rm -rf /tmp/dist
ENV TRACKFLOW_DEFAULT_DEVICE=tensorrt
WORKDIR /work
ENTRYPOINT ["trackflow"]
