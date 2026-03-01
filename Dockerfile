FROM nvcr.io/nvidia/pytorch:25.09-py3

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_BREAK_SYSTEM_PACKAGES=1

# Copy only the package manifest first so that the expensive flash-attn
# compilation layer is cached independently of source file changes.
COPY pyproject.toml README.md ./

# flash-attn: compile against base image's torch (--no-build-isolation).
# bitsandbytes: binary wheel, CUDA <= 13.0 required (25.09 ships 12.6).
RUN uv pip install --system --no-cache --no-build-isolation 'flash-attn==2.7.4.post1' && \
    uv pip install --system --no-cache 'bitsandbytes==0.49.1'

COPY . .

RUN uv pip install --system --no-cache -e .

ENTRYPOINT ["python", "/app/docker-entrypoint.py"]
