# VibeVoice ASR: Docker Guide with 8-bit Quantization

Run VibeVoice ASR inference with 8-bit quantization on Ubuntu with NVIDIA GPUs. 8-bit quantization reduces the model from ~14GB to ~7GB VRAM, making it fit comfortably on 16GB GPUs like the RTX 4080.

**This is the recommended approach for 16GB GPUs.** bf16 (14GB) is too tight, and vLLM bf16 also OOMs due to KV cache overhead.

## Prerequisites

- Ubuntu with NVIDIA GPU drivers installed
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NAS mounted at `/mnt/NAS_1/`
- GPU with >= 16GB VRAM

## Container Version

bitsandbytes requires a container with CUDA <= 13.0 (pre-compiled binaries not available for newer CUDA).

| Container | CUDA | bitsandbytes |
|-----------|------|-------------|
| `25.09` | 13.0 | Works (recommended) |
| `25.06` | 12.9 | Works |
| `25.03` | 12.8 | Works |
| `25.12` | 13.1 | Broken (no pre-compiled binary) |

## First-Time Setup

### 1. Create a Persistent Docker Container

Use `--name` without `--rm` so the container survives reboots:

```bash
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all -it --name vibevoice \
  -v /mnt/NAS_1:/NAS_1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.09-py3
```

Volume mounts:
- `/mnt/NAS_1` → `/NAS_1` inside the container (audio files)
- `~/.cache/huggingface` → persists downloaded models across container restarts

### 2. Install VibeVoice (inside the container)

```bash
apt update && apt install ffmpeg -y
git clone https://github.com/bac2qh/VibeVoice.git
cd VibeVoice
pip install -e .
pip install bitsandbytes flash-attn --no-build-isolation
```

### 3. Run Inference

```bash
cd /root/VibeVoice
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --load_in_8bit \
  --attn_implementation flash_attention_2 \
  --output /NAS_1/results/output.json
```

The model downloads on first run and is cached on the host at `~/.cache/huggingface`.

## After Reboot

The container persists across reboots. Restart and reattach:

```bash
sudo docker start vibevoice
sudo docker exec -it vibevoice bash
cd /root/VibeVoice
```

Then run inference as usual:

```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --load_in_8bit \
  --attn_implementation flash_attention_2 \
  --output /NAS_1/results/output.json
```

All pip packages and cloned repos inside the container are preserved — no reinstall needed.

## Updating VibeVoice

To pull the latest code inside the container:

```bash
cd /root/VibeVoice
git pull
pip install -e .
```

## Inference Examples

**Single file with hotwords:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --load_in_8bit \
  --attn_implementation flash_attention_2 \
  --context "Alice, Bob, machine learning" \
  --output /NAS_1/results/output.json
```

**Single file with hotwords from file:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --load_in_8bit \
  --attn_implementation flash_attention_2 \
  --hotwords_file /NAS_1/hotwords.txt \
  --output /NAS_1/results/output.json
```

**Batch process a directory:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_dir /NAS_1/audio/ \
  --load_in_8bit \
  --attn_implementation flash_attention_2 \
  --output /NAS_1/results/batch_output.json
```

## Hotwords File Format

One term per line:

```
Alice
Bob
machine learning
OpenAI
```

Or comma-separated:

```
Alice, Bob, machine learning, OpenAI
```

## Memory Usage

| Mode | VRAM (approx) | Fits on 4080 (16GB) |
|------|---------------|---------------------|
| bfloat16 (default) | ~14 GB | No (OOM with any overhead) |
| 8-bit quantized | ~8-9 GB | Yes (recommended) |

## Technical Details: Selective 8-bit Quantization

The `--load_in_8bit` flag uses `BitsAndBytesConfig` with selective module skipping to prevent audio representation corruption:

**Quantized to INT8 (using LLM.int8()):**
- Qwen2 LLM backbone layers (`model.language_model.layers.*`): ~7B parameters
- Reduces these layers from 14GB to ~7GB VRAM

**Kept in bfloat16 (full precision):**
- `acoustic_tokenizer`: 52 FFN linears processing raw audio (small dims, high sensitivity)
- `semantic_tokenizer`: Same architecture, processes semantic audio features
- `acoustic_connector`: 64→3584 projection bridging audio latents to LLM space
- `semantic_connector`: 64→3584 projection added element-wise with acoustic features
- `lm_head`: 3584→152064 vocabulary projection (affects token probabilities directly)

**Why skip these modules?**

Quantization errors in the tokenizers and connectors corrupt the audio representation **before** the LLM sees it. Even small errors at the 64-dimensional audio latent level get amplified through the connector's asymmetric projection (64→3584), causing the LLM to receive garbled input → gibberish output.

The skipped modules total ~800M parameters (kept in bf16), adding ~1.5GB VRAM compared to full quantization, but are essential for maintaining transcription quality. The trade-off is well worth it: you get ~40% memory savings (14GB → 8-9GB) while preserving accuracy.

## CLI Reference

```
--model_path          Model path or HuggingFace name (required)
--audio               Single audio file path
--audio_dir           Directory of audio files for batch processing
--load_in_8bit        Enable 8-bit quantization
--attn_implementation flash_attention_2 | sdpa | eager
--context             Inline context string (hotwords, speaker names, topics)
--hotwords_file       Path to hotwords file
--output              Save results as JSON
--max_new_tokens      Max tokens to generate (default: 32768)
--temperature         Sampling temperature, 0 = greedy (default: 0.0)
--repetition_penalty  Repetition penalty (default: 1.0)
--verbose             Print detailed output
```
