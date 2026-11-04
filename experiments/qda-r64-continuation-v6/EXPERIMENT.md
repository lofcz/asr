# Rank64 continuation on the remaining audio

Resume from qda-full-r64-v4 checkpoint 001314 (188.165313 audio hours). Restore
the adapter, AdamW moments and per-worker RNG. The previous schedule ended at
zero: the new continuation has its own measured peak LR and standard Hugging
Face linear schedule with the existing 10% warmup convention. This schedule
choice is inherited from the approved recipe, not claimed to be an LR-finder
result. No plateau, new optimizer or capacity sweep.

The new pass contains 285.051740 hours / 2,703 windows from 717 recordings.
Consumed parent intervals plus the continuation cover all 473.217053 eligible
training hours exactly once. 62.474438 held-out hours remain separate, by
recording/audio hash and show. The 11 unresolved source conversion conflicts
remain separately recorded. No speaker-count, confidence, VAD, noise, clipping
or music-score filters are applied. Exact Soniox text and speaker transitions
are retained; local speaker IDs are consistently remapped within a crop.

Safe boundaries preserve complete Soniox segments and assign zero-duration
segments once. An independent interval audit checks no overlap or missing
seconds. Targets and actual waveform prompts must fit 65,536 tokens without
truncation. Audio SHA256 is rechecked before decoding; decoder timestamp errors
use an exact sample-index slice of the fully decoded source, without padding.

One disposable torch-lr-finder 0.2.2 sweep runs from the parent weights and Adam
moments: 60 exponentially spaced updates, 1e-6 through 1e-3, fixed CS/SK
held-out content loss, two GPUs, matched 45–180-second training windows. All
calibration weight/optimizer changes are discarded and restoration verified.
The selected rate and rationale are recorded in run-spec.json with evidence
hashes before the full pass starts. The sweep is diagnostic, not independent
trials or a claim of optimality.

Training keeps rank64, alpha32, dropout0.05, NF4/BF16, AdamW with zero weight
decay, two-worker token-weighted DDP, norm clipping1.0, frozen speech features,
Flash SDPA and activation offload. Token-matched pairs balance GPU work; the
actual length mix is preserved without duplicating short or long windows.

Checkpoint and fixed-panel content loss every five updates. Decoded interview
NN-WER/DER plus both approved CS/SK markers every50 additional audio hours.
Vanilla and parent outputs are reused with provenance. Full eight-interview
evaluation at the end. Raw output defects must be reported with all words
retained in WER; never silently drop malformed or out-of-bounds samples.
Long held-out sessions are evaluated separately before claiming 30–60-minute
QDA readiness. Neither training-source stress tests nor synthetic markers are
independent confirmation.

Restore checks pin runtime, data, model, tokenizer, target and adapter hashes.
Two recent full local checkpoints plus first/milestones are retained even if
the SSD backup connection fails. Intermediate frozen train features are
recomputable and retired after durable checkpoints to bound server disk usage.
Independent live metrics, disk retention, bounded crash recovery, and checksum
verified SSD backup run as persistent services. W&B receives metrics only.

## Recorded continuation LR decision

The 60-update sweep completed and restored the parent state. Use the upstream
steepest-descent suggestion, 2.099103720108554e-5. The raw minimum is nearby at
2.3598334667821945e-5 (held-out content loss 0.059298); loss stays nearly flat
through 1e-4, then worsens toward 0.061406 at 3.10e-4 and 0.071847 at 1e-3. This
supports a lower continuation rate than the fresh-run 3e-4. These short sequential
measurements do not establish an optimal rate or a WER gain.

The continuation contains 1,352 optimizer updates. Standard linear warmup lasts
136 updates, then linear decay reaches zero at the end of the complete remainder.
Periodic decoded evaluations occur at updates 233, 453, 682, 911 and 1169, where
50, 100, 150, 200 and 250 additional audio hours are reached. The reserved long
panel is 30.006 minutes of Czech and 55.056 minutes of Slovak, both development
recordings separate from the fixed eight clips and all training material.
