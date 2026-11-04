# ASR figures

Paper-style figures with white backgrounds, standard axes, direct values, and consistent model colors. The [repository README](../README.md) contains the figures and numeric comparison tables.

| Figure | PNG | SVG | PDF |
|---|---|---|---|
| Model comparison | [PNG](01-commercial-open-source.png) | [SVG](01-commercial-open-source.svg) | [PDF](01-commercial-open-source.pdf) |
| Czech and Slovak | [PNG](02-language-comparison.png) | [SVG](02-language-comparison.svg) | [PDF](02-language-comparison.pdf) |
| Checkpoint comparison | [PNG](03-checkpoint-progress.png) | [SVG](03-checkpoint-progress.svg) | [PDF](03-checkpoint-progress.pdf) |
| WER and DER | [PNG](04-text-speaker-tradeoff.png) | [SVG](04-text-speaker-tradeoff.svg) | [PDF](04-text-speaker-tradeoff.pdf) |

[Combined PDF](asr-results.pdf) · [CSV values](values.csv) · [Evidence snapshot](results.json)

## Methods

**Reference and scoring.** Soniox transcripts are held fixed. All systems use the existing `cs-sk-cardinals-v1` number normalizer: punctuation and case are normalized, numeric variants are standardized, and diacritics remain significant. WER is the word-level edit distance divided by reference word count. Soniox is the reference provider, not a competitor with an assigned zero error rate.

**Model comparison.** Figures 1–2 use two generated dialogue recordings, one Czech and one Slovak: 11.48 minutes and 1,776 reference words. Figure 1 averages the two language WERs equally. The Gemini score uses the saved independent transcription before adjudication. VibeVoice-ASR and the 450h QLoRA checkpoint use the saved outputs from the release evaluation. These dialogue recordings were excluded from training but repeatedly used as development markers.

**Checkpoint comparison.** Figures 3–4 use all eight `new-confirmation` recordings in the saved release evaluation: four per language, 79.83 minutes and 11,271 normalized reference words. Every plotted WER covers the same recordings. WER pools edit counts and reference words; DER pools diarization-error seconds and scored speaker seconds, with a 250 ms collar. One of the eight scored 450h outputs is not strictly valid in its original form; the source evaluation records this separately. This cohort was consumed for release selection.

**Whisper.** The benchmark runs the official [Whisper large-v3-turbo weights](https://huggingface.co/openai/whisper-large-v3-turbo) at revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. It uses Transformers 4.51.3, PyTorch 2.8.0, FP16, one NVIDIA RTX A5000, known language, beam size 5, temperature 0, sequential long-form decoding, and no previous-window conditioning. No reference text, hotwords, external prompt, or VAD is supplied. Silence skipping is disabled because the optional no-speech processor in this Transformers version passes an unsupported argument to Whisper's forward method. The decoder uses its 448-token total window limit. These settings were fixed before the first successful transcription.

For each recording, the runner verifies source and reference hashes, decodes the source at 24 kHz, takes the same integer-sample crop as the release evaluator, and resamples that crop to 16 kHz. The original marker WAVs are used in full. All ten transcriptions are scored with the shared normalizer. Whisper produces text but no speaker identities in this run, so its DER is unmeasured, not zero. Model revision, library versions, generation settings, crop hashes, source hashes, elapsed times, and per-recording edit counts are retained in `results.json`.

**Interpretation.** These are comparisons against machine transcripts, not independently annotated human ground truth or a general SOTA ranking. The [full release evaluation](../experiments/qda-v1-release-eval/report.md) covers 32 main recordings plus two long sessions. The selected 450h checkpoint scores 32/32 main recordings and 0/2 unchunked long sessions. The final checkpoint has the lowest full-panel WER point estimate, but the saved paired show-level interval does not resolve its text difference from 450h.

## Reproduction

Render from the committed, transcript-free snapshot:

```sh
uv run --no-project --python /usr/bin/python scripts/build_result_charts.py
```

The script generates 300 dpi PNGs, vector SVGs and PDFs, the combined PDF, CSV values, and the README tables. It needs no model weights or original audio for this step.

Verify and refresh the snapshot when the original evaluations are present locally:

```sh
uv run --no-project --python /usr/bin/python scripts/build_result_charts.py --refresh-data
```

Run Whisper in the existing ASR evaluation environment on `mlserver`:

```sh
/data/work/asr-pilot/venv/bin/python scripts/benchmark_whisper.py \
  --markers /data/work/asr-evaluation/production-reference-audit-v2 \
  --release /data/work/asr-experiments/qda-v1-release-eval \
  --normalizer /data/work/asr-evaluation/periodic/code/number_wer.py \
  --output /data/work/asr-experiments/whisper-large-v3-turbo \
  --cache-dir /data/work/asr-pilot/hf/hub \
  --device 1
```

Completed per-recording outputs are reused only when their inference contract, source audio, and reference hashes match. Mirror the resulting JSON files into local `experiments/whisper-large-v3-turbo/` before refreshing the figures. Raw hypotheses remain ignored by Git; only scalar evidence and provenance enter the committed snapshot. No training or paid API calls are performed.
