#!/usr/bin/env python3
"""
Convention-based Docker entrypoint for VibeVoice ASR.

Scans /input for audio files, writes {stem}.json per file to /output.
No CLI args needed — all config comes from environment variables.

Environment variables:
  MODEL_PATH           HuggingFace model path (default: microsoft/VibeVoice-ASR)
  LOAD_IN_8BIT         Enable 8-bit quantization (default: true)
  ATTN_IMPLEMENTATION  Attention backend (default: flash_attention_2)
  HOTWORDS             Comma-separated hotwords (overridden by /input/hotwords.txt)
"""

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm", ".mp4"}
INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
RECORDINGS_DIR = Path("/recordings")


def collect_audio_files() -> list[Path]:
    return sorted(
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_hotwords() -> str | None:
    hotwords_file = INPUT_DIR / "hotwords.txt"
    if hotwords_file.exists():
        content = hotwords_file.read_text(encoding="utf-8").strip()
        if content:
            return content
    env_hotwords = os.environ.get("HOTWORDS", "").strip()
    return env_hotwords if env_hotwords else None


def load_model(
    model_path: str,
    load_in_8bit: bool,
    attn_implementation: str,
) -> tuple:
    print(f"Loading model: {model_path}")
    print(f"8-bit quantization: {load_in_8bit}")
    print(f"Attention implementation: {attn_implementation}")

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
        local_files_only=True,
    )

    model_kwargs = {
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
        "device_map": "auto",
        "local_files_only": True,
    }

    if load_in_8bit:
        # Only the Qwen2 LLM backbone gets quantized to INT8.
        # Audio-sensitive modules are skipped: quantization errors at the
        # 64-dim audio latent level get amplified through the 64→3584
        # connector projection, causing gibberish output.
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=[
                "acoustic_tokenizer",
                "semantic_tokenizer",
                "acoustic_connector",
                "semantic_connector",
                "lm_head",
            ],
        )
        print("Loading with 8-bit quantization (skipping audio modules)")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e9:.2f}B params on {next(model.parameters()).device}")
    return processor, model


def transcribe(processor, model, audio_path: Path, context_info: str | None) -> dict:
    device = next(model.parameters()).device

    inputs = processor(
        audio=str(audio_path),
        sampling_rate=None,
        return_tensors="pt",
        add_generation_prompt=True,
        context_info=context_info,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32768,
            pad_token_id=processor.pad_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.0,
        )
    generation_time = time.time() - start_time

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    try:
        segments = processor.post_process_transcription(generated_text)
    except Exception as e:
        print(f"  Warning: failed to parse structured output: {e}")
        segments = []

    return {
        "file": str(audio_path),
        "raw_text": generated_text,
        "segments": segments,
        "generation_time": generation_time,
        "context_used": context_info,
    }


def main() -> int:
    audio_files = collect_audio_files()
    if not audio_files:
        print("No audio files found in /input — exiting")
        return 0

    print(f"Found {len(audio_files)} audio file(s) to process")

    hotwords = load_hotwords()
    if hotwords:
        print(f"Using hotwords: {hotwords[:100]}{'...' if len(hotwords) > 100 else ''}")

    model_path = os.environ.get("MODEL_PATH", "microsoft/VibeVoice-ASR")
    load_in_8bit = os.environ.get("LOAD_IN_8BIT", "true").lower() in ("true", "1", "yes")
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")

    try:
        processor, model = load_model(model_path, load_in_8bit, attn_implementation)
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return 1

    failed = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
        output_path = OUTPUT_DIR / f"{audio_file.stem}.json"
        try:
            result = transcribe(processor, model, audio_file, hotwords)
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  -> {output_path.name} ({result['generation_time']:.1f}s, {len(result['segments'])} segments)")
            try:
                dest = RECORDINGS_DIR / audio_file.name
                shutil.move(str(audio_file), str(dest))
                print(f"  -> archived to /recordings/{audio_file.name}")
            except Exception as move_err:
                print(f"  Warning: could not archive audio file: {move_err}")
        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            try:
                output_path.write_text(
                    json.dumps({
                        "file": str(audio_file),
                        "error": str(e),
                        "raw_text": "",
                        "segments": [],
                        "generation_time": 0.0,
                        "context_used": hotwords,
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as write_err:
                print(f"  Warning: could not write error JSON: {write_err}")
            failed += 1

    print(f"\nDone: {len(audio_files) - failed}/{len(audio_files)} succeeded")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
