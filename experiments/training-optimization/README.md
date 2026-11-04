# Training optimization at checkpoint 350

The first 700 training windows are unchanged. The remaining 710 are paired by
estimated compute cost (total tokens, quadratic attention component, and target
tokens), with matched pairs shuffled within the original curriculum stages.
The original bucket sequence, sample dictionaries, window identities and
training coverage are preserved exactly. Pair-cost proxy falls 17.75%; this is
an estimate of scheduling improvement, not a measured end-to-end speedup.

Runtime changes:
- MLP chunks 1024 -> 2048; vocabulary-loss chunks 128 -> 512.
- No activation offload at <=36,000 tokens (520 remaining windows).
- Above that threshold, retain at most 1 GiB of boundary tensors and offload
  the rest to pageable CPU memory; no unbounded pinned-memory allocation.
- One-window CPU prefetch overlaps checksummed decoding and tokenization with
  GPU work. Tokenizer calls are serialized with validation. Prefetch is bounded.
- Audited exact target-token counts replace a pre-forward all-reduce. Each
  prepared sample asserts its actual token count. Global token-weighted loss
  is unchanged.
- Release redundant CPU checkpoint state after restoring optimizer/RNG.
- Per-rank preparation, prefetch wait, encoding/transfer, forward, backward/sync
  and offload measurements are logged to W&B. Dashboard exposes token balance.
- ETA excludes pre-optimization timings when fitting new stage throughput.

Controlled benchmark (same checkpoint and cached inputs, dropout disabled,
second iteration, forward/backward only):

| Input | Previous | Optimized | Reduction |
|---|---:|---:|---:|
| 30 min / 30,806 tokens | 57.35 s | 47.58 s | 17.0% |
| 60 min / 61,834 tokens | 143.15 s | 132.00 s | 7.8% |

Gradient cosine similarities: 0.999943 (30m), 0.999914 (60m). Relative gradient
L2 differences: 1.07% and 1.32%; norm ratios within 0.18%. Maximum absolute loss
difference 0.000325. The mathematical objective is unchanged, but BF16 kernels
with different chunk shapes are not bitwise identical. Future dropout draws
and the optimizer trajectory also change with chunking and sample reordering.

With normal training dropout enabled, synthetic memory stress at the no-offload
cutoff (36,000 tokens) peaked at 19.40 GiB; at the dataset maximum (65,147 tokens)
it peaked at 19.63 GiB. Both completed backward and optimizer-state allocation
with finite gradients. Synthetic padded targets are memory tests, not quality
evaluation. All benchmarks used the existing two A5000 GPUs.

Checkpoint migration verifies exact optimizer/RNG equality after serialization,
byte-identical adapter files, and identical completed prefix. Original checkpoint
and plan/config are retained under before-migration on both server and SSD.
The full token/coverage audit passed: 745 training sources, 1,410 windows,
maximum 65,147 tokens, each source covered exactly once, no holdouts included.

Runtime hashes and benchmark results are in results/. The production trainer
is experiments/qda-496h-v1/train_release.py, deployed to
/data/work/asr-pilot/train_release.py. Periodic evaluations remain enabled.

Live verification: resumed from checkpoint 350 and completed updates 351–352;
update 353 was underway at verification. Update 352 target-token balance was
99.886%, with ~1 ms prefetch waits on both ranks. Both used zero activation
offload, with finite losses/gradients. W&B phase metrics were queried and verified.
The dashboard rendered without JavaScript errors. The original checkpoint backup
was checksum-compared between server and SSD: all file contents matched.
