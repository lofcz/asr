# VibeVoice-ASR: one-hour long-session pilot

This experiment uses **one intact 59m58s Czech interview for training**, plus a **separate 10-minute Slovak interview holdout**. It establishes a hardware/training recipe for 10–60-minute sessions on `mlserver`; it is not evidence of general QDA accuracy or a finished production model.

See [RESULTS.md](RESULTS.md) for measured training and generation comparisons, [DATASET_POLICY.md](DATASET_POLICY.md) for rich-label/curriculum requirements, and [PROJECT_REVIEW.md](references/PROJECT_REVIEW.md) for inspected external projects.

## Artifacts and provenance

- Local: `/run/media/lofcz/ssd_external/GitHub/asr/experiments/long-context-pilot`
- Server: `/data/work/asr-pilot/long-context-pilot`
- Environment: `/data/work/asr-pilot/venv`; exact packages in `environment.json`.
- Model: `microsoft/VibeVoice-ASR`, pinned revision `d0c9efdb8d614685062c04425d91e01b6f37d944`.
- Original recordings, hashes, crop offsets, and split: `sample/sources.json`.
- Soniox `stt-async-v5` results: `sample/<source-id>/soniox.json`. Recording-level diarization was requested, then training windows derived locally. Both remote Soniox files/jobs were deleted after downloading results; `api-state.json` records cleanup. Credentials were read locally from the requested `.env`, not copied to the server.
- `dataset/manifest.jsonl`: audio references, relative timestamps, transcript segments, local speaker labels, original session speaker IDs, provenance, and confidence.
- `data-validation.json`: split, timestamp, speaker-mapping, feature, and token checks.
- `runs/`: measurements and adapters. Adapter weights remain on `mlserver`; local results include configuration and metrics.
- `generation/`: actual generated text and evaluation results. References are Soniox pseudo-labels, not human-corrected ground truth.
- `results.json`: machine-readable experiment summary; regenerate with `python summarize_results.py`.

The train and validation sources/shows are disjoint. The training source is Czech and the holdout is Slovak, so this is a narrow transfer check, not a language-balanced evaluation. The available interviews are a proxy for QDA sessions.

## Efficient data representation

Keep original compressed recordings and raw Soniox output as the durable source. The pilot uses mono 24 kHz FLAC for reproducible derived audio, with one manifest instead of a directory of tiny per-example transcript files.

The hot training path uses memory-mapped NumPy arrays: projected audio features in float16, input IDs and labels in int32, and an acoustic mask in bool. `cache/index.jsonl` holds offsets, lengths, and provenance. Training reads contiguous slices and does not decode audio or rerun the frozen speech encoders on every update.

This cache is valid only while **both speech encoders and both connectors remain frozen**. It stores one fixed draw from the audio encoder per example. It must be regenerated when those modules or audio processing change. Targets remain the model's `Start`, `End`, `Speaker`, `Content` JSON format; the storage optimization does not change that interface.

There are six approximately 10-minute training windows, two approximately 30-minute windows, and one full-hour window, all derived from the **same hour**. These are alternative context experiments, not three independent hours. Do not count their durations as unique training data. Full-hour inference on this source is a context/coverage probe, not held-out accuracy.

The longest example has 27,050 prompt tokens and 34,784 supervised target tokens: **61,834 tokens total**. Bucketing should use total token count, not duration alone. More densely spoken hours may exceed the tested 65,536-token limit; cut at a suitable boundary rather than truncating the answer.

At 7.5 feature frames/second × 3,584 dimensions × 2 bytes, 500 hours require approximately **96.8 GB (90.1 GiB)** of projected features for one set of views. Caching all three context variants would roughly triple this. Use bounded shards or choose each recording's context view before caching when expanding beyond the pilot; the current server does not have enough free disk for every variant of 500 hours plus all other artifacts.

## Training recipe

- QLoRA: NF4 4-bit frozen decoder weights, double quantization, BF16 computation.
- Rank 16, alpha 32, dropout 0.05; adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Total trainable adapter parameters: **40,370,176**.
- AdamW, learning rate `1e-5`, weight decay `0.01`, gradient norm clipping `1.0`.
- One complete session/window per GPU. Two GPUs use synchronous DDP, not model sharding.
- Native PyTorch Flash SDPA with full causal attention over the complete sequence.
- Non-reentrant activation checkpointing; feed-forward computations split into 1,024-token chunks; exact cross-entropy computed in 128-target-token vocabulary-head chunks.
- CPU activation offload for full-hour contexts. Two-GPU pinned-memory offload failed on this host; ordinary host-memory offload succeeded on both GPUs at 148 seconds per update, about 18.4 GiB allocated per GPU.

Chunking feed-forward layers and the vocabulary head does **not** split the attention context or independently transcribe pieces. Output, input-gradient, and parameter-gradient equivalence for checkpointed/chunked/offloaded feed-forward computation is checked by `verify_optimizations.py`.

BF16 LoRA is also viable for 10-minute windows, and for 30-minute windows with feed-forward chunking and activation offload. It ran out of memory on the full-hour example with the tested implementation. QLoRA is the practical common configuration for a long-session curriculum. Full-parameter tuning was not benchmarked: conventional mixed-precision AdamW states for a roughly 7B decoder exceed the two cards' combined memory before long-context activations. Heavy CPU/NVMe offloading would be a separate, substantially more demanding experiment.

The learning rate and two-update pilots are conservative probes, not a selected optimum. A duration mixture, epoch count, and final checkpoint need selection on representative generated outputs. The paper motivates retaining long contexts; it does not supply this LoRA optimizer recipe.

## Reproduce on mlserver

Use a fresh run name; existing runs are never overwritten:

```bash
cd /data/work/asr-pilot/long-context-pilot
/data/work/asr-pilot/venv/bin/python validate_data.py
bash run_recipe.sh my-30m-run 1800 2
```

`run_recipe.sh` launches one job across both GPUs. It trains from the existing cache and saves adapters after each update. It is an experiment runner; its checkpoints do not include optimizer/RNG state for exact training resumption.

For generation, after training releases the selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /data/work/asr-pilot/venv/bin/python generate.py \
  --name my-heldout-evaluation \
  --adapter runs/my-30m-run/checkpoint-2
```

Omit `--adapter` for the base model. Add `--split train --bucket 3600` for the full-hour coverage probe. Generation uses the BF16 base with a merged adapter, including for QLoRA-trained adapters. This evaluates the intended BF16 inference configuration.

## Evaluation and interpretation

Generated-output checks include EOS vs token limit/repetition abort, JSON parsing, spoken-segment schema, timestamp bounds, predicted speakers, and last timestamp. Bracket-only event labels are retained in output and reported separately, but excluded from speech word scoring because the Soniox-derived reference contains speech only.

WER uses lowercase Unicode word tokens with punctuation removed. cpWER optimally permutes speaker labels. tcpWER uses a 5-second collar and MeetEval's character-based pseudo-word timing from segment boundaries. DER uses the complete recording interval as UEM, a 250 ms collar, and includes overlap. Segment-level timing and pseudo-label error limit interpretation of all these numbers. Differences on this one holdout do not establish statistical significance.

The paper PDF, complete text/Markdown conversion, and methodology review are in [`../vibevoice-pilot/references/`](../vibevoice-pilot/references/TRAINING_INSIGHTS.md). Relevant source: [VibeVoice-ASR technical report](https://arxiv.org/abs/2601.18184v2), especially its 8K→64K context curriculum, >50-minute SFT examples, annotation verification, and WER/DER/cpWER/tcpWER evaluation. Upstream training interface: [Microsoft finetuning-asr](https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr).
