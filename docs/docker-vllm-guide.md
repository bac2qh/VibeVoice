# VibeVoice ASR: Docker Guide (vLLM)

Run VibeVoice ASR via vLLM for memory-efficient, high-throughput inference. Uses PagedAttention for dynamic GPU memory management.

**Note:** The 7B model in bf16 requires ~14GB for weights alone. On 16GB GPUs (e.g. RTX 4080), this leaves very little room for KV cache. A GPU with >= 24GB VRAM (e.g. RTX 3090/4090) is recommended. See the [16GB GPU section](#16gb-gpu-low-memory-setup) for constrained setups.

## Prerequisites

- Ubuntu with NVIDIA GPU drivers installed
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NAS mounted at `/mnt/NAS_1/`
- GPU with >= 24GB VRAM recommended (16GB possible with constraints)

## Quick Start

### 1. Clone the Fork

```bash
git clone https://github.com/bac2qh/VibeVoice.git
cd VibeVoice
```

### 2. Launch the vLLM Server

```bash
docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -v /mnt/NAS_1:/NAS_1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"
```

Volume mounts:
- `$(pwd)` → `/app` inside the container (VibeVoice code + vLLM plugin)
- `/mnt/NAS_1` → `/NAS_1` inside the container (audio files)
- `~/.cache/huggingface` → persists downloaded models across container restarts

### 3. Check Server Logs

```bash
docker logs -f vibevoice-vllm
```

Wait until you see the server is ready before running inference.

### 4. Run Inference

**Basic transcription:**
```bash
docker exec -it vibevoice-vllm \
  python3 vllm_plugin/tests/test_api.py /NAS_1/audio/recording.mp3
```

**With hotwords:**
```bash
docker exec -it vibevoice-vllm \
  python3 vllm_plugin/tests/test_api.py /NAS_1/audio/recording.mp3 \
  --hotwords "Alice,Bob,machine learning"
```

**With auto-recovery from repetition loops (recommended for long audio):**
```bash
docker exec -it vibevoice-vllm \
  python3 vllm_plugin/tests/test_api_auto_recover.py /NAS_1/audio/recording.mp3

# With hotwords
docker exec -it vibevoice-vllm \
  python3 vllm_plugin/tests/test_api_auto_recover.py /NAS_1/audio/recording.mp3 \
  --hotwords "Alice,Bob"
```

## Managing the Server

```bash
# Stop
docker stop vibevoice-vllm

# Restart
docker start vibevoice-vllm

# Remove container
docker rm vibevoice-vllm
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | Max FFmpeg processes for audio decoding | `64` |
| `PYTORCH_ALLOC_CONF` | PyTorch memory allocator config | `expandable_segments:True` |

## 16GB GPU (Low-Memory Setup)

The default `start_server.py` uses `--max-model-len 65536` and `--max-num-seqs 64`, which requires far more VRAM than 16GB. To run on a 16GB GPU (e.g. RTX 4080), bypass the default script and launch vLLM directly with constrained settings:

```bash
docker rm -f vibevoice-vllm 2>/dev/null

docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -v /mnt/NAS_1:/NAS_1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "pip install -e /app && vllm serve microsoft/VibeVoice-ASR \
    --served-model-name vibevoice \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --enforce-eager \
    --chat-template-content-format openai \
    --tensor-parallel-size 1 \
    --allowed-local-media-path /app \
    --allowed-local-media-path /NAS_1 \
    --port 8000"
```

Key differences from default:
- `--max-model-len 4096` (down from 65536) — limits KV cache memory
- `--max-num-seqs 1` (down from 64) — one request at a time
- `--gpu-memory-utilization 0.95` (up from 0.8) — use nearly all VRAM
- `--enforce-eager` — skips CUDA graph compilation to save memory

**Limitations:** Short context window (4096 tokens) limits transcription length. For long audio, a 24GB+ GPU is needed.

## Troubleshooting

**CUDA out of memory / model fails to load:**
- If using default `start_server.py`: switch to the [16GB GPU setup](#16gb-gpu-low-memory-setup)
- Lower `--max-model-len` further (e.g. 2048)
- Ensure no other processes are using the GPU: `nvidia-smi`

**Audio decoding failed:**
- Ensure audio file is inside a mounted directory (`/NAS_1` or `/app`)
- Check format is supported: `ffmpeg -i /NAS_1/audio/file.mp3`

**Server won't start:**
- Check logs: `docker logs vibevoice-vllm`
- Ensure no other process is using port 8000 or the GPU

## vLLM Version Compatibility

The vLLM plugin uses internal vLLM APIs (`multimodal.profiling`, `multimodal.processing`, etc.) that change across releases. The plugin was written against vLLM ~v0.14.x (Jan 2026).

**Recommended:** `v0.14.1`

**Known issues with other versions:**

| Version | Status | Issue |
|---------|--------|-------|
| `v0.14.1` | Recommended | Compatible with the plugin |
| `v0.15.0` | May work | Try if v0.14.1 has issues |
| `v0.15.1` / `latest` | Broken | `AudioMediaIO` removed, `multimodal.profiling` moved |

If you hit import errors like `ModuleNotFoundError: No module named 'vllm.multimodal.profiling'`, pin to an older version:

```bash
docker rm -f vibevoice-vllm 2>/dev/null

docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -v /mnt/NAS_1:/NAS_1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"
```
