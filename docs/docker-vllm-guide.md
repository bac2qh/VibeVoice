# VibeVoice ASR: Docker Guide (vLLM)

Run VibeVoice ASR via vLLM for memory-efficient, high-throughput inference. Uses PagedAttention to fit the 7B model on GPUs with 16GB VRAM.

Tested on RTX 4080 (16GB VRAM).

## Prerequisites

- Ubuntu with NVIDIA GPU drivers installed
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NAS mounted at `/mnt/NAS_1/`
- GPU with >= 16GB VRAM

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
  vllm/vllm-openai:latest \
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

## Troubleshooting

**CUDA out of memory:**
- Reduce GPU memory utilization: add `-e VLLM_GPU_MEMORY_UTILIZATION=0.85` to docker run
- Reduce max sequences: add `-e VLLM_MAX_NUM_SEQS=1` to docker run

**Audio decoding failed:**
- Ensure audio file is inside a mounted directory (`/NAS_1` or `/app`)
- Check format is supported: `ffmpeg -i /NAS_1/audio/file.mp3`

**Server won't start:**
- Check logs: `docker logs vibevoice-vllm`
- Ensure no other process is using port 8000 or the GPU
