#!/usr/bin/env python3
"""Transcribe audio with VibeVoice-CSK1 using the official VibeVoice runtime."""
import argparse
import json
from pathlib import Path
import sys

import torch

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def load_model(model_path, device="cuda:0", dtype=torch.bfloat16,
               revision=None, local_files_only=False):
    options = {"revision": revision, "local_files_only": local_files_only}
    processor = VibeVoiceASRProcessor.from_pretrained(str(model_path), **options)
    model, loading = VibeVoiceASRForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True, output_loading_info=True, **options,
    )
    problems = {key: loading[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if problems:
        raise RuntimeError(f"Incomplete model checkpoint: {problems}")
    model = model.to(device).eval()
    return model, processor


@torch.inference_mode()
def transcribe(model, processor, audio, max_new_tokens=4096, context=None):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    inputs = processor(
        audio=str(audio), return_tensors="pt", padding=True,
        add_generation_prompt=True, context_info=context,
    )
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=processor.pad_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generated = output[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    complete = bool(len(generated) and generated[-1].item() == processor.tokenizer.eos_token_id)
    segments = processor.post_process_transcription(text)
    return {
        "text": " ".join(segment.get("text", "") for segment in segments),
        "segments": segments, "raw_text": text,
        "complete": complete, "generated_tokens": len(generated),
    }


def main():
    here = Path(__file__).resolve().parent
    default_model = str(here) if (here / "config.json").is_file() else "lofcz/VibeVoice-CSK1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=default_model, help="Hugging Face repo ID or local model directory")
    parser.add_argument("--revision", help="Optional Hugging Face commit or tag")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--context", help="Optional vocabulary or domain context")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    model, processor = load_model(
        args.model, args.device, getattr(torch, args.dtype),
        args.revision, args.local_files_only,
    )
    result = transcribe(model, processor, args.audio, args.max_new_tokens, args.context)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    if not result["complete"]:
        print("Token limit reached; the output is incomplete. Increase --max-new-tokens or split the audio.", file=sys.stderr)
        return 2
    if not result["segments"] and result["raw_text"].strip() != "[]":
        print("Could not parse transcription segments; raw_text contains the original output.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
