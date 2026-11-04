"""Run pinned Whisper large-v3-turbo on the existing, fixed evaluation audio.

Run in the existing ASR evaluation environment. Writes only to --output.
No training, reference edits, or paid API calls are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import subprocess
import time

MODEL = "openai/whisper-large-v3-turbo"
REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
EVENTS = {"[Unintelligible Speech]", "[Music]", "[Human Sounds]", "[Environmental Sounds]", "[Noise]", "[Silence]"}
DECODE = {
    "task": "transcribe", "num_beams": 5, "do_sample": False,
    "temperature": 0.0, "condition_on_prev_tokens": False,
    "return_timestamps": True, "max_length": 448,
    "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
    # Decode all windows. Transformers 4.51.3's optional no-speech processor
    # passes input_ids to Whisper.forward(), which does not accept that name.
    "no_speech_threshold": None,
}


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=pathlib.Path, required=True)
    parser.add_argument("--release", type=pathlib.Path, required=True)
    parser.add_argument("--normalizer", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    import imageio_ffmpeg
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    spec = importlib.util.spec_from_file_location("number_wer", args.normalizer)
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read(args.markers / "gold-manifest.json")
    selection = read(args.release / "selection.json")
    samples = []
    for row in manifest["samples"]:
        reference_path = args.markers / row["reference_path"]
        assert sha(reference_path) == row["reference_sha256"]
        samples.append({
            "id": "marker-" + row["language"], "cohort": "dialogue_markers",
            "language": row["language"], "audio": args.markers / row["audio_path"],
            "audio_sha256": row["audio_sha256"], "reference": read(reference_path),
            "reference_sha256": row["reference_sha256"], "start": 0,
            "duration": row["duration_seconds"], "join": "",
        })
    for row in selection["rows"]:
        if row["cohort"] != "new-confirmation":
            continue
        folder = args.release / "evaluations/450h" / row["window_id"]
        reference = read(folder / "reference.json")
        prior = read(folder / "result.json")
        assert reference == row["target"]
        assert sha(folder / "reference.json") == prior["reference_sha256"]
        samples.append({
            "id": row["window_id"], "cohort": "new-confirmation", "language": row["language"],
            "audio": pathlib.Path(row["audio_file"]), "audio_sha256": row["audio_sha256"],
            "reference": reference, "reference_sha256": prior["reference_sha256"],
            "start": row["source_start"], "duration": row["duration"], "join": " ",
            "expected_reference_words": prior["metrics"]["number_normalized_reference_words"],
        })
    assert len(samples) == 10
    settings = {
        "model": MODEL, "revision": REVISION, "runtime": "transformers",
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "numpy", "scipy", "rapidfuzz", "imageio-ffmpeg"]},
        "dtype": "float16", "decoding": DECODE, "known_language": True,
        "external_prompt": None, "vad": False, "sample_rate": 16000,
        "audio_preparation": "Full source decoded at 24 kHz; integer sample crop matching the release evaluator; polyphase resample to 16 kHz.",
        "normalizer_sha256": sha(args.normalizer), "runner_sha256": sha(__file__),
        "selection_sha256": sha(args.release / "selection.json"),
        "marker_manifest_sha256": sha(args.markers / "gold-manifest.json"),
    }
    contract = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261105)
    settings["gpu"] = torch.cuda.get_device_name(args.device)
    save(args.output / "settings.json", settings)
    print("Loading pinned Whisper large-v3-turbo", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL, revision=REVISION, cache_dir=args.cache_dir,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(f"cuda:{args.device}").eval()
    model.generation_config.forced_decoder_ids = None
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, cache_dir=args.cache_dir)
    results = []
    for row in samples:
        result_path = args.output / (row["id"] + ".json")
        if result_path.exists():
            saved = read(result_path)
            assert saved["contract_sha256"] == contract and saved["reference_sha256"] == row["reference_sha256"]
            assert saved["audio_sha256"] == row["audio_sha256"] and saved["complete"]
            results.append(saved)
            print("Reused " + row["id"], flush=True)
            continue
        print("Decoding " + row["id"], flush=True)
        started = time.monotonic()
        assert sha(row["audio"]) == row["audio_sha256"]
        decoded = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-threads", "1",
            "-i", str(row["audio"]), "-vn", "-ac", "1", "-ar", "24000", "-f", "f32le", "-",
        ], capture_output=True, check=True)
        assert not decoded.stderr
        full_wave = np.frombuffer(decoded.stdout, dtype="<f4")
        first = round(row["start"] * 24000)
        count = round(row["duration"] * 24000)
        wave = full_wave[first:first + count].copy()
        assert 0 <= count - len(wave) <= 3 and np.isfinite(wave).all()
        pcm_hash = hashlib.sha256(wave.tobytes()).hexdigest()
        del full_wave, decoded
        wave = resample_poly(wave, 2, 3).astype(np.float32)
        inputs = processor(wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, truncation=False, padding="longest")
        inputs = inputs.to(f"cuda:{args.device}", dtype=torch.float16)
        with torch.inference_mode():
            predicted = model.generate(**inputs, language=row["language"], **DECODE)
        text = processor.batch_decode(predicted, skip_special_tokens=True)[0]
        assert text.strip(), "Whisper returned no words"
        reference_text = row["join"].join(s["Content"] for s in row["reference"] if s["Content"].strip() not in EVENTS)
        measured = normalizer.score(reference_text, text, row["language"])
        if "expected_reference_words" in row:
            assert measured["number_normalized_reference_words"] == row["expected_reference_words"]
        result = {
            "id": row["id"], "cohort": row["cohort"], "language": row["language"],
            "complete": True, "duration_seconds": len(wave) / 16000,
            "source_start": row["start"], "audio_sha256": row["audio_sha256"],
            "reference_sha256": row["reference_sha256"], "canonical_pcm_sha256": pcm_hash,
            "contract_sha256": contract, "hypothesis": text, "metrics": measured,
            "wall_seconds": time.monotonic() - started,
        }
        save(result_path, result)
        results.append(result)
        print(json.dumps({"completed": len(results), "expected": len(samples), "id": row["id"], "wer": measured["number_normalized_wer"], "seconds": result["wall_seconds"]}), flush=True)
        del inputs, predicted
        torch.cuda.empty_cache()
    save(args.output / "result.json", {
        "complete": True, "expected": len(samples), "scored": len(results), "settings": settings,
        "rows": [{k: v for k, v in row.items() if k != "hypothesis"} for row in results],
        "created_at": time.time(),
    })
    print("Whisper comparison complete", flush=True)


if __name__ == "__main__":
    main()
