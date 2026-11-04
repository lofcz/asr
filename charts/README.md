# ASR figures

Paper-style figures with white backgrounds, standard axes, direct values, and consistent model colors. The [repository README](../README.md) contains the two figures.

| Figure | PNG | SVG | PDF |
|---|---|---|---|
| 1. Checkpoint comparison | [PNG](01-checkpoint-progress.png) | [SVG](01-checkpoint-progress.svg) | [PDF](01-checkpoint-progress.pdf) |
| 2. Model comparison | [PNG](02-commercial-open-source.png) | [SVG](02-commercial-open-source.svg) | [PDF](02-commercial-open-source.pdf) |

[Combined PDF](asr-results.pdf) · [CSV values](values.csv) · [Evidence snapshot](results.json)

## Methods

**Metric.** We report word error rate (WER), using the standard edit-count definition employed in [NIST OpenASR evaluation](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934557):

$$
\mathrm{WER}(r,h) = 100\,\frac{S+D+I}{N}\;\%,
$$

where $S$, $D$, and $I$ are the substitutions, deletions, and insertions in a minimum-cost word-token alignment, and $N$ is the number of reference tokens. Each edit has unit cost. Reference $r$ and hypothesis $h$ undergo the same text preprocessing before alignment; $N$ is counted after that preprocessing. Text normalization is part of the evaluation protocol, not the name of a separate metric. Insertions can make WER exceed 100%.

**Text preprocessing.** The versioned [`cs-sk-cardinals-v1` implementation](../experiments/periodic-eval/number_wer.py) applies Unicode NFC and case folding, tokenizes Unicode word sequences, and maps supported Czech/Slovak cardinal expressions and integers from 0 to 9999 to shared numeric tokens. Selected inflected forms are included; diacritics remain significant. Ordinary punctuation is excluded, while recognized decimal, date, ordinal-number, and leading-zero numeric tokens are retained without cardinal conversion. Unsupported numeral expressions are not standardized. Reference chunks in the dialogue recordings are concatenated losslessly; model segments and corpus reference segments are joined with whitespace. Explicit non-speech event tags are excluded according to the existing scorer. These are project-specific preprocessing rules, not a claim of equivalence to another benchmark's scoring protocol.

**Aggregation.** Figure 2 reports the macro-average over languages, not a pooled WER:

$$
\mathrm{WER}_{\mathrm{macro}} = \tfrac12\left(\mathrm{WER}_{\mathrm{cs}}+\mathrm{WER}_{\mathrm{sk}}\right).
$$

Figure 1 reports corpus WER, computed by aligning each recording separately and pooling its edit counts and reference-token counts:

$$
\mathrm{WER}_{\mathrm{corpus}} = 100\,\frac{\sum_u(S_u+D_u+I_u)}{\sum_u N_u}\;\%.
$$

Corpus WER is not the unweighted mean of recording-level WERs.

**Reference provenance.** Soniox supplied the fixed machine-generated reference transcripts. They are used as the scoring reference for every system. Their origin affects interpretation of the measurements, not the definition or name of WER; no reference-provider self-score is reported.

**Model comparison.** Figure 2 uses two generated dialogue recordings, one Czech and one Slovak: 11.48 minutes and 1,776 reference words. Figure 2 averages the two language WERs equally. The Gemini score uses the saved independent transcription before adjudication. VibeVoice-ASR and the 450h QLoRA checkpoint use the saved outputs from the release evaluation. These dialogue recordings were excluded from training but repeatedly used as development markers.

**Checkpoint comparison.** Figure 1 uses all eight `new-confirmation` recordings in the saved release evaluation: four per language, 79.83 minutes and 11,271 reference tokens after preprocessing. Every plotted corpus WER covers the same recordings. DER pools diarization-error seconds and scored speaker seconds, with a 250 ms collar. One of the eight scored 450h outputs is not strictly valid in its original form; the source evaluation records this separately. This cohort was consumed for release selection.

**Whisper.** The benchmark runs the official [Whisper large-v3-turbo weights](https://huggingface.co/openai/whisper-large-v3-turbo) at revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. It uses Transformers 4.51.3, PyTorch 2.8.0, FP16, one NVIDIA RTX A5000, known language, beam size 5, temperature 0, sequential long-form decoding, and no previous-window conditioning. No reference text, hotwords, external prompt, or VAD is supplied. Silence skipping is disabled because the optional no-speech processor in this Transformers version passes an unsupported argument to Whisper's forward method. The decoder uses its 448-token total window limit. These settings were fixed before the first successful transcription.

For each recording, the runner verifies source and reference hashes, decodes the source at 24 kHz, takes the same integer-sample crop as the release evaluator, and resamples that crop to 16 kHz. The original marker WAVs are used in full. All ten transcriptions are scored with the shared normalizer. Whisper produces text but no speaker identities in this run, so its DER is unmeasured, not zero. Model revision, library versions, generation settings, crop hashes, source hashes, elapsed times, and per-recording edit counts are retained in `results.json`.

**Interpretation.** These are comparisons against machine transcripts, not independently annotated human ground truth or a general SOTA ranking. The [full release evaluation](../experiments/qda-v1-release-eval/report.md) covers 32 main recordings plus two long sessions. The selected 450h checkpoint scores 32/32 main recordings and 0/2 unchunked long sessions. The final checkpoint has the lowest full-panel WER point estimate, but the saved paired show-level interval does not resolve its text difference from 450h.

## Reproduction

Render from the committed, transcript-free snapshot:

```sh
uv run --no-project --python /usr/bin/python scripts/build_result_charts.py
```

The script generates 300 dpi PNGs, vector SVGs and PDFs, the combined PDF, and CSV values. It needs no model weights or original audio for this step.

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
