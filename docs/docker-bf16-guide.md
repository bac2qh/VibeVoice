# VibeVoice ASR: Docker Guide (bf16)

Run VibeVoice ASR inference in bfloat16 on Ubuntu with NVIDIA GPUs. No extra quantization libraries needed.

Tested on RTX 4080 (16GB VRAM). The 7B model uses ~14GB in bf16.

## Prerequisites

- Ubuntu with NVIDIA GPU drivers installed
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NAS mounted at `/mnt/NAS_1/`
- GPU with >= 16GB VRAM

## Quick Start

### 1. Launch Docker Container

```bash
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all --rm -it \
  -v /mnt/NAS_1:/NAS_1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.12-py3
```

Volume mounts:
- `/mnt/NAS_1` → `/NAS_1` inside the container (audio files)
- `~/.cache/huggingface` → persists downloaded models across container restarts

### 2. Install VibeVoice

```bash
git clone https://github.com/bac2qh/VibeVoice.git
cd VibeVoice
pip install -e .
pip install flash-attn --no-build-isolation
```

### 3. Run Inference

**Single file:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --attn_implementation flash_attention_2 \
  --output /NAS_1/results/output.json
```

**With inline context/hotwords:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --attn_implementation flash_attention_2 \
  --context "Alice, Bob, machine learning" \
  --output /NAS_1/results/output.json
```

**With hotwords from file:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio /NAS_1/audio/recording.mp3 \
  --attn_implementation flash_attention_2 \
  --hotwords_file /NAS_1/hotwords.txt \
  --output /NAS_1/results/output.json
```

**Batch process a directory:**
```bash
python demo/vibevoice_asr_inference_with_context.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_dir /NAS_1/audio/ \
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

## Troubleshooting

**OOM (Out of Memory):**
- Switch attention: `--attn_implementation sdpa` uses slightly less memory than `flash_attention_2`
- Lower max tokens: `--max_new_tokens 8192` (default is 32768)
- Ensure no other processes are using the GPU: `nvidia-smi`

**flash-attn build fails:**
- Fall back to sdpa (no extra install needed): `--attn_implementation sdpa`

## CLI Reference

```
--model_path          Model path or HuggingFace name (required)
--audio               Single audio file path
--audio_dir           Directory of audio files for batch processing
--attn_implementation flash_attention_2 | sdpa | eager
--context             Inline context string (hotwords, speaker names, topics)
--hotwords_file       Path to hotwords file
--output              Save results as JSON
--max_new_tokens      Max tokens to generate (default: 32768)
--temperature         Sampling temperature, 0 = greedy (default: 0.0)
--repetition_penalty  Repetition penalty (default: 1.0)
--verbose             Print detailed output
```
