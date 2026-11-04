#!/usr/bin/env python3
"""Merge a VibeVoice ASR adapter using Microsoft's official PEFT recipe."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from peft import PeftModel

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

BASE_REVISION = "d0c9efdb8d614685062c04425d91e01b6f37d944"
TOKENIZER_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
VIBEVOICE_REVISION = "16fb2cb1217c9934a886e1948ffb06120caa2df5"
ADAPTER_SHA256 = "6a43720eec9a7b75ceb1bc80d709ed539f45c1a8f1dd5ba7ef7168581bdf1b1b"
ADAPTER_CONFIG_SHA256 = "85911cc9ffabefbf1dd4f7f8e7c8339167223774617cf9e50ae538bf327e9add"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")
    if sha256(args.adapter / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise ValueError("Adapter does not match the selected 450h checkpoint")
    if sha256(args.adapter / "adapter_config.json") != ADAPTER_CONFIG_SHA256:
        raise ValueError("Adapter configuration does not match the selected checkpoint")
    torch.set_num_threads(8)
    print("Loading the pinned full BF16 base on CPU", flush=True)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        "microsoft/VibeVoice-ASR", revision=BASE_REVISION,
        cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="sdpa",
    ).eval()
    shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    frozen = {}
    for name in ("acoustic_tokenizer", "semantic_tokenizer", "acoustic_connector", "semantic_connector"):
        module = getattr(model.model, name)
        key, tensor = next(iter(module.state_dict().items()))
        frozen[f"model.{name}.{key}"] = tensor_sha256(tensor)

    # Official recipe: load the adapter, merge it into the unquantized base,
    # and save_pretrained. Use the same BF16 addition as the evaluated model;
    # PEFT 0.17.1's optional safe_merge rounds the delta to BF16 before adding.
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    adapted = [(name, layer) for name, layer in model.named_modules() if hasattr(layer, "lora_A")]
    if not adapted:
        raise ValueError("No LoRA layers were loaded")
    name, layer = adapted[0]
    before = tensor_sha256(layer.get_base_layer().weight)
    expected = layer.get_base_layer().weight.detach().clone()
    expected += layer.get_delta_weight("default")
    print(f"Merging {len(adapted)} LoRA layers", flush=True)
    model = model.merge_and_unload().eval()
    for layer_name, adapted_layer in adapted:
        if not torch.isfinite(adapted_layer.get_base_layer().weight).all():
            raise ValueError(f"Nonfinite merged weight: {layer_name}")
    state = model.state_dict()
    if {key: tuple(value.shape) for key, value in state.items()} != shapes:
        raise ValueError("Merged state differs from the complete base model schema")
    merged_key = name.removeprefix("base_model.model.") + ".weight"
    if not torch.equal(state[merged_key], expected) or tensor_sha256(state[merged_key]) == before:
        raise ValueError("Merged weight does not match base plus LoRA update")
    for key, digest in frozen.items():
        if tensor_sha256(state[key]) != digest:
            raise ValueError(f"Frozen speech component changed: {key}")
    if any("lora_" in key for key in state):
        raise ValueError("Unmerged adapter tensors remain")
    model.config._name_or_path = ""
    print("Saving the full merged checkpoint", flush=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")

    from huggingface_hub import snapshot_download
    tokenizer_path = snapshot_download(
        "Qwen/Qwen2.5-7B", revision=TOKENIZER_REVISION, cache_dir=args.cache_dir,
        allow_patterns=["*token*", "*.json", "merges.txt", "vocab.json"],
    )
    processor = VibeVoiceASRProcessor.from_pretrained(
        "microsoft/VibeVoice-ASR", revision=BASE_REVISION,
        language_model_pretrained_name=tokenizer_path, cache_dir=args.cache_dir,
    )
    processor.save_pretrained(args.output)
    processor.tokenizer.save_pretrained(args.output)
    manifest = {
        "model_name": "VibeVoice-CSK1", "base_model": "microsoft/VibeVoice-ASR",
        "base_revision": BASE_REVISION, "tokenizer": "Qwen/Qwen2.5-7B",
        "tokenizer_revision": TOKENIZER_REVISION, "vibevoice_revision": VIBEVOICE_REVISION,
        "checkpoint": "450h", "step": 2564, "adapter_sha256": ADAPTER_SHA256,
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "dtype": "bfloat16", "merge": "PeftModel.merge_and_unload()",
        "lora_layers": len(adapted), "state_tensors": len(state),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": {"full_state_schema": True, "sample_lora_update_exact": True,
                       "frozen_speech_samples_unchanged": True, "no_adapter_tensors": True},
        "versions": {package: importlib.metadata.version(package) for package in
                     ("torch", "transformers", "peft", "safetensors", "huggingface_hub")},
        "sha256": {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()},
    }
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved and verified {args.output}", flush=True)


if __name__ == "__main__":
    main()
