# Full corpus rank64 preparation

The capacity probes are stopped. This preparation targets one unique pass through
the existing paid corpus, starting from the pinned vanilla VibeVoice-ASR model.
Rank64 is the user's selected capacity. The short probes did not establish a clear
WER advantage over rank16. No adapter or optimizer state from either the old 250h
run or an LR range test is used to initialize the full run.

## Learning rate: measure first

The selected full-run learning rate is **3e-4**, following approval of the measured
useful band. It lies within the low held-out-content-loss region around 2e-4 to
5e-4 and below the loss increase near 1e-3. `run-spec.json` records the selection
and the SHA-256 of its evidence. The full-run schedule is **linear warmup over
10% of optimizer updates to 3e-4, followed by linear decay to zero over the
remaining 90%**. It spans the complete final corpus plan.

`../qda-lr-calibration-v1/lr_find.py` uses the actual
[`torch-lr-finder` 0.2.2 implementation](https://github.com/davidtvs/pytorch-lr-finder)
of [Smith's LR range test](https://arxiv.org/abs/1506.01186). Its exponential sweep,
smoothing (0.05), divergence threshold (5× best smoothed loss), and suggestion
routine come from that implementation. DDP training and fixed-panel validation
hooks adapt it to the existing ASR objective; they do not substitute a new LR curve.

- Fresh rank64, alpha32, NF4/BF16, AdamW, no weight decay, gradient clipping 1.0.
- Two GPUs, same global token-weighted objective and original token-matched pairs.
- Up to 100 disposable updates from 1e-6 to 1e-2, bounded to 45 minutes of loop time.
- Seventeen existing short-window DDP pairs, shuffled reproducibly and cycled.
  This is a calibration diagnostic, not 500h training exposure.
- Score the same held-out CS/SK interview content loss after every update; preserve
  total and structure loss, gradient norm, LR, raw points and upstream smoothing.
- Restore and verify all trainable parameters afterward, then discard the process.
- The upper-bound sweep is intended to expose instability; it is never continued
  into the full run. Keep raw measurements alongside the upstream LR suggestion.

The test completed 92 updates and stopped at the upstream divergence threshold.
Restoration of every trainable parameter was verified. The raw held-out content
loss minimum was 0.07453 at 4.64e-4. Nearby measurements were 0.08579 at 9.55e-5,
0.07626 at 2.92e-4, and 0.09297 at 9.77e-4.

The library's automatic steepest-gradient heuristic suggested **3.76e-5**, a
different diagnostic from the raw minimum or the selected **3e-4**. The selection
uses the observed low-loss band, not that heuristic. A sequential range test
confounds increasing LR with accumulated updates, so these points are not
independent fixed-LR trials and do not prove an optimal LR or WER improvement.
Raw records, `result/analysis.json`, and PNG/SVG plots are retained in
`../qda-lr-calibration-v1/` on mlserver and the SSD.

An LR range test supports an LR range. It cannot establish which decay schedule
will give the best 500h WER. Changing the decay family requires an explicit,
evidence-linked decision, recorded in the run specification; a repository default
alone is not evidence that it is optimal on this dataset.

The schedule choice uses the [current Thinking Machines SFT guidance](https://tinker-docs.thinkingmachines.ai/tinker/optimizers/#rules-of-thumb),
which reports 1.3x step efficiency for linear decay to zero versus a constant LR.
That comparison is text SFT evidence, not a measured ASR advantage. The 10%
warmup follows the [VibeVoice fine-tuning recipe](https://github.com/microsoft/VibeVoice/blob/main/finetuning-asr/README.md).
It is a conventional initialization choice, not a warmup duration inferred from
the range test. Neither parameter is claimed to be optimal on this corpus.

The trainer uses Hugging Face 4.51.3 `get_linear_schedule_with_warmup` through
`get_scheduler`. No optimizer, momentum, objective or adapter settings change.
Its clock is the optimizer steps in the complete immutable run plan. Evaluation
pauses and checkpoint resumes preserve scheduler state instead of restarting the
clock. A CPU regression test verifies an interrupted run's exact LR sequence and
parameter updates against uninterrupted execution.

| Completed optimizer updates | LR available for the next update |
| --- | --- |
| 0% | 0 (warmup begins) |
| 10% | 3e-4 |
| 25% | about 2.5e-4 |
| 50% | about 1.67e-4 |
| 75% | about 8.33e-5 |
| 90% | about 3.33e-5 |
| 100% | 0 (run complete) |

Warmup uses `ceil(0.1 * total_steps)`, so the exact points depend on the final
update count. Audio hours are not interchangeable with update percentages when
crop lengths vary. `schedule-preview.json` samples the actual library scheduler;
before the plan exists it is explicitly a normalized preview. Exact tokenization
regenerates it with the real update count and cumulative audio hours. A 50h
evaluation pause does not end or reset the schedule.

## Corpus accounting and defect recovery

The original release contains 496.339h in total, including 466.715h in its original
training split. Combining both paid releases and applying all current held-out
show exclusions leaves 479.639h of candidate training audio.

The earlier 20h screen rejected entire files containing an ambiguous >2s gap
between pieces of one word, leaving only 306.972h structurally eligible. That
was an unnecessarily broad exclusion for preparation of the full corpus.

The revised builder preserves the exact Soniox token stream, speaker attribution,
and timestamp extents. It marks every entire segment containing such an ambiguous
word, then forbids training crops from intersecting those segments. It does not
repair timestamps, split a word, invent an event, or silently remove speech inside
a retained audio crop. Speaker conflicts within a word still quarantine a source.
An independent verifier checks the whole reconstructed lexical stream separately.

The rebuild retains 860/871 sources structurally; 11 remain quarantined. Training
sources comprise **473.217h before local-span and acoustic exclusions**. Their
426 marked ambiguous segments total **1.0655h**. Final retained hours must come
from `selection-summary.json`, not from these earlier-stage totals.

The selector partitions the complete eligible training corpus into nonoverlapping
windows, preferring a 10/30/60-minute mix and retaining shorter clean tails where
appropriate. It preserves source-level and show-level holdout exclusions. No
repeated crops or extra epochs are added to manufacture a nominal 500h total.
Every excluded interval and reason is recorded. Confidence, missing-speech,
clipping, decoding, music/noise and conversation gates retain their previous
thresholds. Human review and agreement with a second ASR remain optional.

Exact input tokenization is verified against a 65,536-token limit. Selection uses
a conservative token bound to choose shorter intact windows when necessary;
preparation verifies the exact decoded input. Waveforms and targets are never
truncated to fit. Frozen BF16 features use lossless NPZ storage to fit the larger
corpus on mlserver. Cached inputs/features are hashed and prepared progressively.

Selection runs in parallel by source, with a durable result and provenance
signature for each recording. Results are combined in the original source order;
the first preferred length follows a deterministic 10/30/30/60-minute pattern,
then each recording's remaining windows follow the declared length preference.
Quality thresholds and whole-word crop checks are unchanged.

Startup tokenizes every selected target and fixes the complete optimizer-step
plan before training. It does not wait for all training waveforms and features
to be cached. Each crop is decoded, checked against its source hash and duration,
and encoded before its first update. The frozen encoder returns to CPU before
the gradient forward pass. Completed input tensors and lossless features have
atomic, checksum-bound receipts, so resumes reuse them. Held-out tensors and
features are reused only after exact equality checks. The longest selected
context exercises this input path in the discarded memory check before launch.

## Evaluation, durability and launch

The same eight held-out interviews and both approved synthetic CS/SK markers stay
fixed. Vanilla decoding is reused only after exact token-tensor, feature-bit,
audio-crop and reference equality checks; raw predictions are preserved and CPU
rescored. Short interview WER/DER plus both markers run every 50 training audio
hours. The complete interview panel is scored at the end. Existing regression
guards remain in place; output failures are never silently dropped or repaired.

Checkpoints preserve adapters, AdamW state, scheduler state and each DDP worker's
RNG. The first checkpoint, all evaluation milestones, and the newest two available
checkpoints at each backup cycle are copied to the SSD and independently hashed.
The server's newest two checkpoints always remain available. Intermediate server
snapshots may retire after two newer full SSD backups pass checksum verification;
this prevents a slow connection from creating an unbounded checkpoint queue.
Retirements have explicit receipts. Previous experiments and paid raw data are
unchanged, and the pinned training implementation is unchanged.

The 1,314-update plan produces roughly 263 checkpoints; keeping every optimizer
state would exceed the SSD's remaining space. SSD retention keeps every adapter
snapshot that was copied, plus full resume states for the first checkpoint, latest two,
and every 50h evaluation milestone (including the final checkpoint among the latest
two). An intermediate optimizer state is retired only after its remote copy has
already been archived and two newer full SSD checkpoints pass checksum verification.
Receipts explicitly record state retirement; all transcripts, audio, predictions,
metrics and all copied adapters remain intact. Every 50h model-evolution snapshot
is preserved. This policy applies only to the new v4 run.

Preparation is a separate service that **cannot launch full training**:

```sh
systemctl --user start qda-v4-preparation.service
```

The LR and schedule are recorded. After the full dataset/runtime preparation
passes, the training entry point is:

```sh
cd /data/work/asr-experiments/qda-full-r64-v4
/data/work/asr-pilot/venv/bin/python run_full.py
```

The user has authorized the full launch. `qda-v4-training.service` is started
with a CPU-only readiness wait before `run_full.py`: it waits for the preparation
service to finish successfully and for its completed data/runtime contract. A
failed preparation stops the launch before model loading. The launcher then
rechecks data, executable hashes, the exact plan, LR evidence and the discarded
rank64 memory-check receipt. Waiting is reported separately from training.

Preparation releases its exclusive data lease only after all dataset mutations,
downgrading it to a read lease for the preflight subprocess. Writers remain
excluded while that subprocess obtains its own read lease. Subprocess tests
verify this handoff and failed/incomplete preparation cannot launch training.

Preparation status, LR diagnostics and accounting are available at
`http://localhost:8767/preparation`. W&B receives numeric metrics only.

## Crop-duration recovery, 2026-10-23

At update 345, FFmpeg's output-side timestamp crop returned 1799.166625 seconds
for a 1799.07-second M4A window. The duration guard stopped before that input
reached an optimizer update. The service previously had `Restart=no`, leaving
the GPUs idle after the failure. Checkpoint 340 preserved the complete state;
four subsequent, unsaved updates are replayed on recovery.

Valid timestamp crops retain their exact previous samples. If the duration check
fails, the decoder now obtains the full PCM timeline and selects the requested
integer sample range, with no padding or target changes. Real-file verification
compares the repaired crop with the full PCM slice and confirms that sampled
valid training/development crops remain bit-identical. The corrected frozen
features were generated and checked before restarting training.

`runtime-repair.json` pins the old configuration, new executable hashes, unchanged
plan/cache contract and validation evidence. Old checkpoint files remain intact.
Resume compatibility permits precisely the documented decoder and compatibility
code changes; it rejects changes to LR, rank, schedule, data or other training
settings. New checkpoints record the new runtime and repair receipt. All 37
repository checks passed, including decoder, resume and supervision tests.

The service now supervises progress and allows up to three automatic retries of
known transient failures at the same checkpoint. Integrity, numeric and quality
failures stop with a visible dashboard alert. An idle-GPU watchdog detects a lack
of progress within an attempt. Backups use the same SSH connection settings for
all subprocesses and publish failures instead of leaving a stale success status.
