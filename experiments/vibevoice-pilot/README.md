# VibeVoice ASR microbatch pilot

Local workspace: `asr/experiments/vibevoice-pilot/`.
Server workspace: `mlserver:/data/work/asr-pilot/pilot/`.
Isolated Python environment: `/data/work/asr-pilot/venv/`.
Upstream checkout: `/data/work/asr-pilot/VibeVoice/`.

See [RESULTS.md](RESULTS.md) and [results.json](results.json) for measured results. Adapters remain on mlserver under `runs/<experiment>/adapter/`. They are experimental checkpoints, not production models.

## Dataset and provenance

- 12 distinct shows, 180 seconds from each: 36 minutes submitted to Soniox `stt-async-v5` with diarization and language identification enabled. Eight Czech shows, four Slovak shows.
- Ten shows for training (~30 minutes), two separate shows for validation (~6 minutes), one validation show per language. This prevents recording/show overlap; it does not establish globally disjoint speaker identities.
- `sample/sources.json` records original recording IDs, source SHA256, show, language, and excerpt offsets. `sample/<id>/soniox.json` preserves the teacher's token output. API job IDs/settings are in `api-state.json`; uploaded files and transcription jobs were deleted after downloading results.
- `prepare_sample.py` reads only `SONIOX_API_KEY` from the sibling `tazatelka/.env` locally. The credential and .env were never copied to mlserver.
- `build_dataset.py` merges Soniox subword pieces into words before splitting clips or segments. It prefers pauses/speaker changes near 30/60/120 seconds, normalizes speaker IDs by first appearance within each clip, rebases timestamps, and groups text into same-speaker segments up to ~20 seconds.
- The 132 rows contain three overlapping duration variants. **Choose one bucket per experiment; these are not 106 minutes of independent training data.**
- `dataset/manifest.jsonl` is the canonical labeling manifest; FLAC audio is 24 kHz mono. `dataset/validation.json` checks file durations and timestamp bounds. `customized_context` is intentionally omitted.
- These are machine labels. Confidence was recorded, not treated as correctness. Excerpt edges can be mid-sentence. Human review and a larger gold evaluation set are still needed before judging ASR/diarization quality.

## Efficient training cache

`cache_features.py` uses the pinned model's own audio preprocessing and frozen speech encoders/connectors. It writes:

- `features.npy`: contiguous memory-mapped float16 projected audio embeddings. The source encoders run in BF16.
- `tokens.npy`, `labels.npy`: int32 token IDs and response-only loss labels.
- `mask.npy`: Boolean audio-placeholder mask.
- `index.jsonl`: token/feature offsets, lengths, bucket, split, provenance, and labels.
- `metadata.json`: model revision, manifest checksum, extraction time, dimensions, and seed.

Features are written incrementally to disk; the full feature tensor is never accumulated in RAM. A `.building` marker prevents training on an interrupted cache. Token arrays/metadata remain in RAM during construction. Training memory-maps arrays and loads only the selected clip. This removes FLAC decoding, resampling, tokenization, and the frozen audio modules from each training step.

The cache fixes one seeded stochastic acoustic-encoder draw per clip. It is valid only while the encoders **and connectors** stay frozen, with the same model/processor and clip boundaries. Rebuild for changes to those components or audio augmentation. This format is not appropriate for full end-to-end fine-tuning.

At 7.5 audio frames/s × 3584 features × 2 bytes, 500 hours need approximately 97 GB decimal (~90 GiB) of projected features plus token/index data, **for one duration variant**. Keep source audio locally if server disk capacity is tight. For larger corpora use multiple cache directories/shards; do not create all duration variants at production scale.

`benchmark.py` computes the same response-only next-token objective using a checkpointed 128-token output-head chunk, avoiding a full sequence × vocabulary logits tensor. The numerical verification checks loss and gradients against ordinary cross-entropy. It uses SDPA with the installed CUDA PyTorch backend, not the upstream FlashAttention dependency.

## Reproduce on mlserver

Dependencies and upstream revision are recorded in `requirements.lock.txt` and `upstream-revision.txt`. `model_paths.py` pins both model and tokenizer snapshots. The environment was installed with PyTorch 2.8.0/cu128, Transformers 4.51.3, PEFT 0.17.1, Accelerate 1.10.1, and bitsandbytes 0.47.0. The upstream package is installed editable with `--no-deps`.

```bash
cd /data/work/asr-pilot/pilot
export HF_HOME=/data/work/asr-pilot/hf

# Rebuild only when no training process is using the cache.
CUDA_VISIBLE_DEVICES=0 ../venv/bin/python cache_features.py
../venv/bin/python verify_cache.py

# Single-GPU matched comparison: use a new name for each experiment.
CUDA_VISIBLE_DEVICES=0 ../venv/bin/python benchmark.py \
  --mode bf16 --bucket 60 --steps 8 --accum 4 --name my-bf16-60
CUDA_VISIBLE_DEVICES=1 ../venv/bin/python benchmark.py \
  --mode qlora --bucket 60 --steps 8 --accum 4 --name my-qlora-60

# Two-GPU measured pilot (global batch = 1 × 2 GPUs × accumulation 2).
CUDA_VISIBLE_DEVICES=0,1 ../venv/bin/torchrun --standalone --nproc_per_node=2 \
  benchmark.py --mode bf16 --bucket 60 --steps 50 --accum 2 \
  --lr 0.00002 --name my-bf16-ddp-60

CUDA_VISIBLE_DEVICES=0 ../venv/bin/python evaluate_generation.py \
  --adapter runs/my-bf16-ddp-60/adapter --name my-adapter
../venv/bin/python summarize.py
```

The benchmark cycles a seeded microdataset to measure throughput and confirm learning. It saves adapters and metrics, not optimizer-resume checkpoints; it is not yet a full 500-hour production Trainer. Do not choose a final learning rate, epoch count, or diarization strategy from this tiny pilot. The adapters use standard PEFT modules and can be loaded onto the full pinned VibeVoice model for ordinary raw-audio inference; only the benchmark training instance removes the frozen audio modules.

## Decision from this pilot

Use cached BF16 rank-16 LoRA on both GPUs, with approximately 60–120-second clips. QLoRA remains useful if substantially longer contexts exhaust VRAM, but BF16 is faster at the measured lengths. Allow about 5 hours per 500-hour training epoch, plus roughly 3.5 hours for a one-time single-GPU feature pass and time for evaluation/checkpoints. These are extrapolations from a small representative sample, not a completed 500-hour run.

Do not deploy the 50-step timing adapter: it loops on one held-out 30-second Slovak excerpt despite improving teacher-forced loss. The shorter 8-step adapter produces valid JSON on both 60-second probes, but changes the Slovak speaker count from the teacher's 3 to 4. Neither establishes improved diarization. A useful next quality experiment is a larger 5–10-hour diverse training subset, a manually checked held-out set, and checkpoint selection using generated transcripts, WER, speaker-aware WER/DER, and repetition/JSON-failure rates. Keep 60–120-second context initially; the base model's speaker counts also deteriorated on the two 30-second probes.

Generation smoke tests use explicit greedy KV-cache decoding, matching the model's forward path. The pinned upstream `generate(inputs_embeds=...)` path failed with an empty-input shape error, so the probe bypasses that wrapper; no inference repetition penalty is applied. The tests preserve raw output, including failures, in `runs/generation-*.json`.

## Paper-informed methodology

The full VibeVoice-ASR v2 report has been downloaded and read. See [training insights](references/TRAINING_INSIGHTS.md) and the [Markdown conversion](references/2601.18184v2.md). The paper motivates extending the 60–120-second hardware pilot toward longer contexts, independent pseudo-label checks, and generation-based checkpoint selection. The measured throughput remains specific to the tested clip lengths.
