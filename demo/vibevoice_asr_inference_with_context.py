#!/usr/bin/env python
"""
VibeVoice ASR Inference with 8-bit Quantization and Context Support

This script supports:
- 8-bit quantization via bitsandbytes for reduced GPU memory usage
- Context/hotwords support (from CLI string or file) for improved transcription accuracy
- Single file or directory batch processing
- JSON output for structured results
"""

import os
import sys
import torch
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import BitsAndBytesConfig

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


class VibeVoiceASRInferenceWithContext:
    """Inference wrapper for VibeVoice ASR with 8-bit quantization and context support."""

    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa"
    ):
        """
        Initialize the ASR inference pipeline.

        Args:
            model_path: Path to the pretrained model
            load_in_8bit: Whether to use 8-bit quantization (requires bitsandbytes)
            device: Device to run inference on (ignored if load_in_8bit=True)
            dtype: Data type for model weights (ignored if load_in_8bit=True)
            attn_implementation: Attention implementation ('flash_attention_2', 'sdpa', 'eager')
        """
        print(f"Loading VibeVoice ASR model from {model_path}")

        if load_in_8bit:
            try:
                import bitsandbytes
                print("✅ bitsandbytes available, enabling 8-bit quantization")
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for 8-bit quantization. "
                    "Install with: pip install bitsandbytes"
                )

        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        # Load model with optional 8-bit quantization
        print(f"Using attention implementation: {attn_implementation}")

        model_kwargs = {
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }

        if load_in_8bit:
            # 8-bit quantization with selective module skipping
            # Skip audio-related layers (tokenizers, connectors, lm_head) to preserve audio representation quality
            # Only the Qwen2 LLM backbone (model.language_model.layers.*) gets quantized to INT8
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=[
                    "acoustic_tokenizer",
                    "semantic_tokenizer",
                    "acoustic_connector",
                    "semantic_connector",
                    "lm_head",
                ]
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            print("Loading model with 8-bit quantization (device_map=auto)")
            print("  Skipping quantization for: acoustic_tokenizer, semantic_tokenizer, acoustic_connector, semantic_connector, lm_head")
        else:
            # Standard loading
            model_kwargs["dtype"] = dtype
            model_kwargs["device_map"] = device if device == "auto" else None

        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Set device
        if load_in_8bit or device == "auto":
            self.device = next(self.model.parameters()).device
        else:
            self.model = self.model.to(device)
            self.device = device

        self.model.eval()

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        if load_in_8bit:
            print(f"💾 8-bit quantization enabled (memory savings: ~50%)")

    def transcribe(
        self,
        audio_path: str,
        context_info: Optional[str] = None,
        max_new_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text with optional context.

        Args:
            audio_path: Path to audio file
            context_info: Optional context (hotwords, speaker names, topics)
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0 for greedy)
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty (1.0 for no penalty)

        Returns:
            Dictionary with transcription results
        """
        # Process audio with context
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=None,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info if context_info and context_info.strip() else None
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Prepare generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }

        # Add sampling parameters only if do_sample is True
        if do_sample:
            generation_config["temperature"] = temperature if temperature > 0 else 0.01
            generation_config["top_p"] = top_p

        start_time = time.time()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )

        generation_time = time.time() - start_time

        # Decode output
        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Parse structured output
        try:
            transcription_segments = self.processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"Warning: Failed to parse structured output: {e}")
            transcription_segments = []

        return {
            "file": audio_path,
            "raw_text": generated_text,
            "segments": transcription_segments,
            "generation_time": generation_time,
            "context_used": context_info if context_info and context_info.strip() else None,
        }


def load_hotwords_from_file(file_path: str) -> str:
    """
    Load hotwords/context from file.

    File format:
    - One term per line, OR
    - Comma-separated terms, OR
    - Free-form text

    Args:
        file_path: Path to hotwords file

    Returns:
        Concatenated context string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            return ""

        # If content has commas but few newlines, treat as comma-separated
        if ',' in content and content.count('\n') < 3:
            # Already comma-separated, just clean up
            return ', '.join([term.strip() for term in content.split(',') if term.strip()])
        else:
            # Treat as newline-separated or free text
            return content

    except Exception as e:
        print(f"❌ Error reading hotwords file {file_path}: {e}")
        return ""


def collect_audio_files(
    audio_path: Optional[str] = None,
    audio_dir: Optional[str] = None
) -> List[str]:
    """
    Collect audio files from single path or directory.

    Args:
        audio_path: Single audio file path
        audio_dir: Directory containing audio files

    Returns:
        List of audio file paths
    """
    audio_files = []

    if audio_path:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_files.append(str(path))

    if audio_dir:
        dir_path = Path(audio_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Invalid directory: {audio_dir}")

        # Common audio extensions
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.webm', '.mp4']

        for ext in audio_extensions:
            audio_files.extend([str(f) for f in dir_path.glob(f"*{ext}")])
            audio_files.extend([str(f) for f in dir_path.glob(f"*{ext.upper()}")])

        # Remove duplicates and sort
        audio_files = sorted(list(set(audio_files)))

    return audio_files


def print_result(result: Dict[str, Any], verbose: bool = True):
    """Pretty print a transcription result."""
    print(f"\n{'='*80}")
    print(f"File: {result['file']}")
    print(f"Generation Time: {result['generation_time']:.2f}s")

    if result.get('context_used'):
        print(f"Context Used: {result['context_used']}")

    if verbose:
        print(f"\n--- Raw Output ---")
        raw_text = result['raw_text']
        print(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

    if result['segments']:
        print(f"\n--- Structured Output ({len(result['segments'])} segments) ---")
        for i, seg in enumerate(result['segments'][:10]):  # Show first 10 segments
            start = seg.get('start_time', 'N/A')
            end = seg.get('end_time', 'N/A')
            speaker = seg.get('speaker_id', 'N/A')
            text = seg.get('text', '')

            # Format times
            start_str = f"{start:.2f}" if isinstance(start, (int, float)) else str(start)
            end_str = f"{end:.2f}" if isinstance(end, (int, float)) else str(end)

            print(f"  [{start_str}s - {end_str}s] Speaker {speaker}: {text[:100]}{'...' if len(text) > 100 else ''}")

        if len(result['segments']) > 10:
            print(f"  ... and {len(result['segments']) - 10} more segments")
    else:
        print("\n⚠️  No structured segments parsed")


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice ASR Inference with 8-bit Quantization and Context Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single file
  python %(prog)s --model_path microsoft/VibeVoice-ASR --audio audio.mp3

  # With 8-bit quantization
  python %(prog)s --model_path microsoft/VibeVoice-ASR --audio audio.mp3 --load_in_8bit

  # With hotwords from file
  python %(prog)s --model_path microsoft/VibeVoice-ASR --audio audio.mp3 --hotwords_file terms.txt

  # With inline context
  python %(prog)s --model_path microsoft/VibeVoice-ASR --audio audio.mp3 --context "John, Mary, AI, Python"

  # Process directory and save results to JSON
  python %(prog)s --model_path microsoft/VibeVoice-ASR --audio_dir ./audios --output results.json
        """
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or HuggingFace model name"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Enable 8-bit quantization (requires bitsandbytes, saves ~50%% memory)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps", "xpu", "auto"],
        help="Device to run inference on (ignored if --load_in_8bit is used)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: sdpa)"
    )

    # Audio input arguments
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to single audio file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files for batch processing"
    )

    # Context arguments
    parser.add_argument(
        "--context",
        type=str,
        help="Context information (hotwords, speaker names, topics) as string"
    )
    parser.add_argument(
        "--hotwords_file",
        type=str,
        help="Path to file containing hotwords/context (one per line or comma-separated)"
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate (default: 32768)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy, default: 0.0)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling (default: 1.0)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 = no penalty, default: 1.0)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results as JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output including raw text"
    )

    args = parser.parse_args()

    # Validate input
    if not args.audio and not args.audio_dir:
        parser.error("Must specify either --audio or --audio_dir")

    # Collect audio files
    try:
        audio_files = collect_audio_files(args.audio, args.audio_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    if not audio_files:
        print("❌ No audio files found")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) to process")
    if args.verbose:
        for f in audio_files:
            print(f"  - {f}")

    # Build context info
    context_info = None

    if args.hotwords_file:
        print(f"Loading hotwords from: {args.hotwords_file}")
        file_context = load_hotwords_from_file(args.hotwords_file)
        if file_context:
            context_info = file_context
            print(f"✅ Loaded context from file: {context_info[:100]}{'...' if len(context_info) > 100 else ''}")

    if args.context:
        if context_info:
            # Combine file context and CLI context
            context_info = f"{context_info}, {args.context}"
        else:
            context_info = args.context
        print(f"✅ Using context: {context_info[:100]}{'...' if len(context_info) > 100 else ''}")

    # Initialize model
    print("\n" + "="*80)
    print("Initializing model...")
    print("="*80)

    # Handle dtype for non-8bit mode
    if not args.load_in_8bit:
        if args.device in ["mps", "xpu", "cpu"]:
            dtype = torch.float32
        else:
            dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16  # Ignored when load_in_8bit=True, but needed for signature

    try:
        asr = VibeVoiceASRInferenceWithContext(
            model_path=args.model_path,
            load_in_8bit=args.load_in_8bit,
            device=args.device,
            dtype=dtype,
            attn_implementation=args.attn_implementation
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Determine if sampling should be enabled
    do_sample = args.temperature > 0

    # Process all audio files
    print("\n" + "="*80)
    print(f"Processing {len(audio_files)} audio file(s)...")
    print("="*80)

    all_results = []

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file}")

        try:
            result = asr.transcribe(
                audio_path=audio_file,
                context_info=context_info,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=do_sample,
                repetition_penalty=args.repetition_penalty,
            )

            all_results.append(result)
            print_result(result, verbose=args.verbose)

        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()

            # Add error result
            all_results.append({
                "file": audio_file,
                "error": str(e),
                "raw_text": "",
                "segments": [],
                "generation_time": 0.0,
                "context_used": context_info,
            })

    # Save results to JSON if requested
    if args.output:
        print(f"\n{'='*80}")
        print(f"Saving results to: {args.output}")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved successfully")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print("="*80)
    print(f"Total files processed: {len(audio_files)}")
    print(f"Successful: {sum(1 for r in all_results if not r.get('error'))}")
    print(f"Failed: {sum(1 for r in all_results if r.get('error'))}")

    total_time = sum(r.get('generation_time', 0) for r in all_results)
    print(f"Total generation time: {total_time:.2f}s")

    if len(audio_files) > 0:
        avg_time = total_time / len(audio_files)
        print(f"Average time per file: {avg_time:.2f}s")

    print("="*80)


if __name__ == "__main__":
    main()
