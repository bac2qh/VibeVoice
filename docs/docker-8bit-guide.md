# VibeVoice ASR: Docker Guide with 8-bit Quantization

Run VibeVoice ASR inference with 8-bit quantization on Ubuntu with NVIDIA GPUs (tested on RTX 4080 16GB).

## Prerequisites

- Ubuntu with NVIDIA GPU drivers installed
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NAS mounted at `/mnt/NAS_1/`

## Quick Start

### 1. Launch Docker Container

```bash
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all --rm -it \
  -v /mnt/NAS_1:/NAS_1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:24.12-py3
```

Volume mounts:
- `/mnt/NAS_1` → `/NAS_1` inside the container (audio files)
- `~/.cache/huggingface` → persists downloaded models across container restarts

### 2. Install VibeVoice

```bash
git clone https://github.com/bac2qh/VibeVoice.git
cd VibeVoice
pip install -e .
pip install bitsandbytes flash-attn --no-build-isolation
```

### 3. Run Inference

**Single file:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --load_in_8bit \
  --attn_implementation flash_attention_2 \
  --output /NAS_1/results/output.json
```

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

Create a text file with one term per line:

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
| bfloat16 (default) | ~14 GB | Tight |
| 8-bit quantized | ~7 GB | Yes |

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
