# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VibeVoice is a family of open-source frontier voice AI models that includes:
- **VibeVoice-ASR**: Unified speech-to-text model handling 60-minute long-form audio with speaker diarization and timestamping
- **VibeVoice-TTS**: Long-form multi-speaker text-to-speech (code removed, models still available on HuggingFace)
- **VibeVoice-Streaming (Realtime)**: Real-time TTS with streaming text input

Core innovation: Continuous speech tokenizers (Acoustic and Semantic) operating at ultra-low frame rate of 7.5 Hz, using next-token diffusion framework with LLM for text understanding and diffusion head for acoustic generation.

## Installation & Setup

### Standard Installation
```bash
# Clone repository
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

# Install package in editable mode
pip install -e .

# For streaming TTS specifically
pip install -e ".[streamingtts]"
```

### Docker Installation (Recommended for ASR)
```bash
# NVIDIA PyTorch Container 24.07 ~ 25.12 verified
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all --rm -it nvcr.io/nvidia/pytorch:25.12-py3

# If flash attention not included:
pip install flash-attn --no-build-isolation
```

## Common Commands

### ASR (Automatic Speech Recognition)

**Gradio Demo:**
```bash
apt update && apt install ffmpeg -y  # Required for demo
python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share
```

**Inference from File:**
```bash
python demo/vibevoice_asr_inference_from_file.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_files path/to/audio.mp3
```

**vLLM High-Performance Serving:**
```bash
# Launch server in background
docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app -w /app --entrypoint bash \
  vllm/vllm-openai:latest \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"

# Test API
docker exec -it vibevoice-vllm python3 vllm_plugin/tests/test_api.py /app/audio.wav
docker exec -it vibevoice-vllm python3 vllm_plugin/tests/test_api.py /app/audio.wav --hotwords "term1,term2"
```

### Streaming TTS (Realtime)

**Basic Inference:**
```bash
python demo/realtime_model_inference_from_file.py
```

**Colab Demo:**
Available at: [demo/vibevoice_realtime_colab.ipynb](demo/vibevoice_realtime_colab.ipynb)

### LoRA Fine-tuning (ASR)

**Install Dependencies:**
```bash
pip install peft
```

**Single GPU:**
```bash
cd finetuning-asr
torchrun --nproc_per_node=1 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --data_dir ./toy_dataset \
  --output_dir ./output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --bf16 \
  --report_to none
```

**Multi-GPU:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --data_dir ./toy_dataset \
  --output_dir ./output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --bf16 \
  --report_to none
```

**Inference with Fine-tuned Model:**
```bash
python inference_lora.py \
  --base_model microsoft/VibeVoice-ASR \
  --lora_path ./output \
  --audio_file ./toy_dataset/0.mp3 \
  --context_info "term1, term2"
```

## Architecture

### Package Structure

```
vibevoice/                          # Main Python package
├── modular/                        # Core model implementations
│   ├── modeling_vibevoice_asr.py   # ASR model architecture
│   ├── modeling_vibevoice_streaming_inference.py  # Streaming TTS inference
│   ├── modeling_vibevoice_streaming.py            # Streaming TTS model
│   ├── modular_vibevoice_diffusion_head.py       # Diffusion head component
│   ├── modular_vibevoice_tokenizer.py            # Speech tokenizer
│   └── streamer.py                                # Streaming utilities
├── processor/                      # Audio processing pipeline
│   ├── vibevoice_asr_processor.py  # ASR audio preprocessing
│   ├── vibevoice_streaming_processor.py  # Streaming TTS processor
│   └── audio_utils.py              # Audio utilities
├── schedule/                       # Scheduling logic
└── configs/                        # Model configurations

vllm_plugin/                        # vLLM integration for high-perf serving
├── model.py                        # vLLM model adapter
├── inputs.py                       # Input processing for vLLM
├── scripts/                        # Server launch scripts
└── tests/                          # API test scripts

demo/                               # Demo scripts and examples
├── vibevoice_asr_gradio_demo.py    # Web UI for ASR
├── vibevoice_asr_inference_from_file.py  # CLI inference
├── realtime_model_inference_from_file.py  # Streaming TTS demo
└── web/                            # Web assets

finetuning-asr/                     # LoRA fine-tuning for ASR
├── lora_finetune.py                # Training script
├── inference_lora.py               # Inference with LoRA weights
└── toy_dataset/                    # Example dataset structure
```

### Key Architectural Concepts

**Speech Tokenizers:**
- Operates at 7.5 Hz frame rate (extremely efficient for long sequences)
- Two types: Acoustic tokenizer and Semantic tokenizer
- Preserves audio fidelity while reducing computational cost

**Next-Token Diffusion Framework:**
- LLM component: Understands textual context and dialogue flow
- Diffusion head: Generates high-fidelity acoustic details
- Enables long-form generation (up to 90 minutes for TTS, 60 minutes for ASR)

**ASR Architecture (VibeVoice-ASR):**
- Unified model for ASR + diarization + timestamping
- Supports customized hotwords for domain-specific terms
- Single-pass processing for up to 60 minutes of audio
- Multilingual: 50+ languages with code-switching support

**Streaming TTS Architecture (VibeVoice-Realtime):**
- Lightweight: 0.5B parameters
- Real-time: ~300ms first audible latency
- Streaming text input support
- Robust long-form generation (~10 minutes)

**vLLM Plugin:**
- Implements vLLM's plugin architecture via entry point: `vllm.general_plugins`
- Registered in pyproject.toml: `vibevoice = "vllm_plugin:register_vibevoice"`
- Provides OpenAI-compatible API (`/v1/chat/completions`) with streaming
- Uses continuous batching for high-throughput inference

### Data Formats

**ASR Fine-tuning Dataset:**
Audio files paired with JSON labels containing:
- `audio_duration`: Duration in seconds
- `audio_path`: Path to audio file
- `segments`: Array of speaker segments with:
  - `speaker`: Speaker ID (0, 1, 2, ...)
  - `text`: Transcription
  - `start`: Start timestamp
  - `end`: End timestamp
- `customized_context` (optional): Domain-specific terms or context sentences

## Contributing Philosophy

This project prioritizes:
1. **Code Minimalism**: Concise, clear, minimal code
2. **High Readability**: Research-oriented, not enterprise-grade
3. **Functional Purity**: Avoid over-engineering and excessive abstraction

**Rejected Patterns:**
- Over-engineering (unnecessary encapsulation, excessive abstraction)
- Style-only PRs (formatting, beautification without functional changes)
- Large chunks of unverified AI-generated code

**Review Process:**
- Line-by-line manual review by maintainers
- Every line must have absolute necessity to exist
- Documentation must be precise, concise, and information-dense

## Model Availability

| Model | HuggingFace | Quick Try |
|-------|-------------|-----------|
| VibeVoice-ASR-7B | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | [Playground](https://aka.ms/vibevoice-asr) |
| VibeVoice-TTS-1.5B | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | Disabled |
| VibeVoice-Realtime-0.5B | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/vibevoice_realtime_colab.ipynb) |

## Important Notes

- **VibeVoice-TTS code was removed** from the repository due to misuse concerns (2025-09-05), but models remain available on HuggingFace
- **Docker is strongly recommended** for ASR work (consistent CUDA environment)
- **vLLM plugin** requires no vLLM source modification - uses plugin entry point system
- **Fine-tuning toy dataset** uses synthetic audio from VibeVoice TTS - not for production use
- **Audio file requirements** for vLLM: Files must be in mounted directory accessible to container
