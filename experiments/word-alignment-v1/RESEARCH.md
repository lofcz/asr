# Czech and Slovak word alignment — candidates through 2026 Q3

Decision date: 2026-10-24. Availability cutoff: 2026-09-30.

The selected comparison is the existing **Czech/Slovak Wav2Vec2 300M aligners versus Canary's auxiliary 600M CTC model**, as requested by the user. OmniASR is background research, not the recommended next deployment or a measured winner. The completed Soniox timing-agreement pilot is documented in [REPORT.md](benchmarks/soniox-v1/REPORT.md). No source examined establishes an independent Czech-and-Slovak word-boundary SOTA ranking. ASR CER/WER results and model size do not establish alignment accuracy.

Meta's v2 model cards existed at commit `94925bae799b77a3e0c802cb02447637b1f70a84`, dated 2025-12-12, before the requested cutoff. Official language IDs include `ces_Latn` and `slk_Latn`. CTC uses Wav2Vec2 encoders and a character tokenizer. Model/code license: Apache 2.0. The official table quotes approximately 3 GiB for 1B and 15 GiB for 7B at batch 1, 30-second audio, BF16 on A100; those are not our A5000 measurements. The reference CTC pipeline limits individual inputs to 40 seconds. Long sessions need alignment over bounded segments. This family uses fairseq2, so it needs an adapter rather than being an unmodified WhisperX Hugging Face model-name substitution.

Sources:
- https://github.com/facebookresearch/omnilingual-asr
- https://github.com/facebookresearch/omnilingual-asr/blob/94925bae799b77a3e0c802cb02447637b1f70a84/src/omnilingual_asr/cards/models/rc_models_v2.yaml
- https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py
- https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/README.md

## Other candidates

- **NVIDIA Canary-1B-v2 auxiliary 600M CTC aligner**: separate FastConformer/Parakeet-style acoustic model, both languages, CC-BY-4.0. NVIDIA trained it specifically to align transcripts and uses it through NeMo Forced Aligner. It is a strong task-specific comparison despite not being a Wav2Vec2 model. It can align an existing VibeVoice transcript without using Canary's decoder. Prefer the original NVIDIA weights and NFA implementation when making accuracy claims. A third-party GGUF conversion exists, but its English comparison against another aligner's output is not a Czech/Slovak ground-truth benchmark.
  - https://huggingface.co/nvidia/canary-1b-v2
  - https://arxiv.org/html/2509.14128v1#S5
- **MahmoudAshraf/mms-300m-1130-forced-aligner**: dedicated Wav2Vec2/MMS forced aligner covering both languages, but weights are CC-BY-NC-4.0. Do not select as the default for this product's commercial deployment merely because the wrapper code has a permissive license.
  - https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner
  - https://github.com/MahmoudAshraf97/ctc-forced-aligner#license
- **mahesh27/mms-300m-ipa-fleurs**, August 2026: interesting alignment-specific research and Czech adapter, but no Slovak adapter in the released inventory. Not a verified solution for both target languages.
  - https://arxiv.org/abs/2608.28508
  - https://huggingface.co/mahesh27/mms-300m-ipa-fleurs
- **Qwen3-ForcedAligner-0.6B**: official 11-language list includes neither Czech nor Slovak. Qwen3-ASR's broader language list does not establish forced-aligner support.
  - https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- **Czech ClTRUS / Slovak fav-kky encoders**: public checkpoints are self-supervised encoders without ready ASR tokenizer/CTC head. Their papers' fine-tuned WER does not make these uploaded checkpoints usable aligners without further training.
  - https://huggingface.co/fav-kky/wav2vec2-base-cs-80k-ClTRUS
  - https://huggingface.co/fav-kky/wav2vec2-base-sk-17k

## Existing pipeline verification

WhisperX commit `5d3678b8535698a2826a5db9eb3e4105aef85dd9`, alignment only. No Whisper transcription, speaker re-estimation, training changes or paid API requests. Original VibeVoice segment content, speaker IDs and timestamps are preserved; nested `words` adds timings and review flags. Missing timing interpolation is explicitly disabled. This implementation snapshot is current at investigation time, not asserted to be the Q3 software snapshot.

| Sample | Audio | Alignment | Peak GPU memory | Words with times | Review flags |
| --- | ---: | ---: | ---: | ---: | --- |
| Czech marker, v4 final | 364.841 s | 8.362 s | 1.524 GiB | 941/941 | 1 numeral unsupported by the character vocabulary |
| Slovak marker, v4 final | 323.830 s | 3.351 s | 1.501 GiB | 831/831 | 25 low CTC path scores |

These are whitespace-delimited display words, not the evaluation WER tokenizer's counts. Durations exclude initial model downloads/loading and audio decoding. The Czech first alignment also initialized NLTK sentence splitting. The two languages ran on separate A5000s. A score below 0.5 is only a review heuristic, not a calibrated probability. Timed coverage is not a timestamp-accuracy measurement. A numeral can get a high wildcard alignment score despite lacking a vocabulary match; unsupported-character flags are essential.

Results: `results/cs-finetuned.json`, `results/sk-finetuned.json`. Model revisions are pinned in `model-cs.json` and `model-sk.json`. Remote isolated environment: `/data/work/asr-word-alignment/venv`. Existing training environment packages were not changed.

## Required validation before choosing a production aligner

The user authorized saved Soniox word timestamps as the comparison reference. This benchmark uses both markers and one natural podcast per language, identical saved VibeVoice transcripts and segment crops, and paired matching words. The report includes raw NFA timings alongside lexical word boundaries, separate start/end errors, coverage, runtime, and memory. Soniox's timestamps are an agreement reference, not independently annotated timing truth. Never count interpolated or wildcard-only timings as verified boundaries. Numbers, Scio/AI proper names, acronyms, hesitations and overlapping speakers remain relevant production edge cases; this pilot does not establish accuracy for every such category.

Synthetic generation and fresh training remain on hold at the user's request until this approach is reviewed. The separately approved synthetic scope is 50 hours with 1–4 speakers in equal proportions plus 10 additional single-speaker hours, 30 hours per language overall, Scio/AI vocabulary, library and private cloned voices, and headless GPT-6.1 Sol medium script generation.
