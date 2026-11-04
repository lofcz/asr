---
license: mit
language:
- cs
- sk
base_model: microsoft/VibeVoice-ASR
pipeline_tag: automatic-speech-recognition
tags:
- vibevoice
- qlora
- speech-recognition
- speaker-diarization
---

# VibeVoice-CSK1

VibeVoice fine-tuning for Czech and Slovak using QLoRA.

Full BF16 checkpoint with the 450h adapter merged into VibeVoice-ASR. Includes both speech encoders, tokenizer, and processor. Outputs text, speaker labels, and timestamps.

## Inference

Requires Python 3.11+ and a CUDA GPU with sufficient memory. Tested on a 24 GB RTX A5000. Longer recordings require more memory. Install FFmpeg for audio formats not supported by SoundFile.

```sh
pip install huggingface-hub==0.36.2
hf download lofcz/VibeVoice-CSK1 --local-dir VibeVoice-CSK1
cd VibeVoice-CSK1
pip install -r requirements.txt
python inference.py recording.wav --device cuda:0 --output transcription.json
```

The script uses Microsoft's pinned VibeVoice implementation and greedy decoding. `--model` accepts a local directory or Hugging Face repo ID. Use `--local-files-only` after downloading to run offline. `--max-new-tokens` defaults to 4096; incomplete generation returns exit code 2 and preserves the partial output. Split long recordings into shorter clips when needed.

Python:

```python
from inference import load_model, transcribe

model, processor = load_model(".", device="cuda:0")
result = transcribe(model, processor, "recording.wav")
print(result["text"])
print(result["segments"])
```

## Merge

The export follows the [official merge recipe](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/finetuning-asr/README.md#merging-lora-weights-optional): `PeftModel.from_pretrained`, `merge_and_unload`, then `save_pretrained`. The base revision, adapter hashes, dependency versions, and tensor checks are recorded in `merge_manifest.json`.

```sh
pip install -r requirements-merge.txt
python merge_lora.py --adapter /path/to/adapter-450h --output merged-model
```

The adapter uses rank 64, alpha 32, and checkpoint step 2564. The exported weights are unquantized BF16. [Training and evaluation code](https://github.com/lofcz/asr).
