# ASR

Czech and Slovak speech recognition experiments: data collection, QLoRA fine-tuning of VibeVoice-ASR, and reproducible evaluation.

## Model comparison

Word error rate (WER), reported as a percentage. Lower is better. Text preprocessing and reference provenance are specified in the [evaluation protocol](charts/README.md#methods).

<!-- MODEL_TABLE_START -->
| Model | Czech WER (%) | Slovak WER (%) | Macro-averaged WER (%) |
|---|---:|---:|---:|
| Gemini 4 Argon | 0.85 | 0.00 | 0.42 |
| Whisper large-v3-turbo | 2.75 | 2.64 | 2.70 |
| VibeVoice-ASR | 2.33 | 1.80 | 2.07 |
| Ours (450h) | 1.06 | 0.72 | 0.89 |
<!-- MODEL_TABLE_END -->

![Mean word error rate](charts/01-commercial-open-source.png)

**Figure 1.** Macro-averaged WER: the arithmetic mean of the Czech and Slovak WERs, with equal weight per language. All systems are evaluated on identical recordings using a shared scoring protocol.

![Word error rate by language](charts/02-language-comparison.png)

**Figure 2.** The same comparison, separated by language.

## Checkpoint comparison

<!-- CHECKPOINT_TABLE_START -->
| Model | Corpus WER (%) | DER (%) | Scored recordings |
|---|---:|---:|---:|
| Whisper large-v3-turbo | 8.95 | — | 8/8 |
| VibeVoice-ASR | 7.52 | 7.14 | 8/8 |
| Ours (300h) | 6.51 | 10.11 | 8/8 |
| Ours (450h) | 6.04 | 11.67 | 8/8 |
| Ours (final) | 6.01 | 13.62 | 8/8 |
<!-- CHECKPOINT_TABLE_END -->

![Checkpoint word error rate](charts/03-checkpoint-progress.png)

**Figure 3.** Corpus WER on the common confirmation recordings: total word edits divided by total reference tokens, expressed as a percentage. The 450h checkpoint is the candidate selected by the saved release evaluation.

![Word and diarization error rates](charts/04-text-speaker-tradeoff.png)

**Figure 4.** Corpus WER and diarization error rate (DER, 250 ms collar) on the same recordings. Lower values are better on both axes. Whisper does not produce speaker labels in this run and has no DER value.

[Vector figures and PDF](charts/README.md) · [All figures as one PDF](charts/asr-results.pdf) · [Numeric values](charts/values.csv) · [Methods and provenance](charts/README.md#methods) · [Full release evaluation](experiments/qda-v1-release-eval/report.md)

## Reproduce the figures

```sh
uv run --no-project --python /usr/bin/python scripts/build_result_charts.py
```

The committed numeric snapshot is sufficient; datasets and model weights are not needed to render the figures. To verify and rebuild that snapshot from the original local evaluations, append `--refresh-data`.

`scripts/benchmark_whisper.py` runs the pinned Whisper comparison on the original audio. Its arguments, inference settings, and model revision are documented in [the methods](charts/README.md#methods).

## Code

- `scripts/`: collection, dataset preparation, evaluation, and figure generation.
- `experiments/`: training recipes, inference, scoring, and experiment tests.
- `services/`: worker and monitoring service definitions.
- `tests/`: collection, data-integrity, and result-validation tests.

Audio, datasets, checkpoints, credentials, and raw experiment outputs remain outside Git.
