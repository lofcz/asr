# VibeVoice ASR pilot measurements

Measured on mlserver, 2026-10-22. All rows below use the corrected word-boundary dataset.

| Run | GPUs | Peak allocated GiB/GPU | Audio throughput | 500 h epoch | Held-out loss before → after |
|---|---:|---:|---:|---:|---:|
| bf16-cached-120 | 1 | 16.65 | 55.8× | 8.96 h | 0.427 → 0.367 |
| bf16-cached-60 | 1 | 15.84 | 53.9× | 9.27 h | 0.466 → 0.398 |
| bf16-ddp-60 | 2 | 15.98 | 102.0× | 4.90 h | 0.466 → 0.374 |
| bf16-raw-60 | 1 | 20.01 | 37.9× | 13.18 h | 0.466 → 0.438 |
| qlora-cached-60 | 1 | 7.17 | 40.7× | 12.29 h | 0.470 → 0.403 |

Throughput excludes model load, first optimizer step, feature preparation, validation, and adapter saving. DDP figures sum audio across GPUs and use the slowest rank’s step time. VRAM figures are PyTorch allocations; reserved/driver memory is higher (see results.json).

Single-GPU runs: 8 optimizer steps, accumulation 4 (raw-audio run: 4 steps). DDP run: 50 optimizer steps, accumulation 2 per GPU = 200 microbatches total. DDP uses learning rate 2e-5; other runs use 1e-4. These runs compare engineering feasibility and throughput, not optimal hyperparameters.

Validation is four clips spanning Czech and Slovak from two held-out shows. Soniox is the teacher, not human ground truth. Lower loss is not a measured WER or DER improvement. Target-120s variants include shorter remainder clips.

All training paths use SDPA, gradient checkpointing, rank-16 LoRA on attention/MLP layers, frozen speech encoders/connectors, and chunked FP32 cross-entropy. The raw-audio comparison retains the optimized loss; it is not a timing of the unmodified upstream Trainer.

The archive/token-boundary-v1 directory preserves the preliminary dataset and experiments, including a failed QLoRA dtype smoke test. They are excluded from the table.

## Generation smoke test

Two held-out clips per probe, one per language (30 or 60 seconds as recorded in JSON). WER uses normalized words against machine labels and is not a quality benchmark.
- generation-base-60: valid JSON 2/2; pseudo-reference WER 15.6%.
- generation-base: valid JSON 2/2; pseudo-reference WER 13.5%.
- generation-bf16-short-60: valid JSON 2/2; pseudo-reference WER 13.0%.
- generation-bf16-ddp-60: valid JSON 1/2; pseudo-reference WER not aggregated: invalid JSON output.

The 50-step tiny-dataset adapter repeats a word until the token limit on the Slovak 30-second probe. Do not deploy this adapter. Teacher-forced validation loss alone did not catch this generation regression. Speaker-count checks on two clips are not DER measurements.
