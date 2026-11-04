# Clean restart: data first, then a bounded capacity comparison

Status: automated data admission passed on 2026-10-23. Preparation and both fresh
experiments are launched by `qda-v3-experiments.service`. Human review and second-ASR
agreement are not required. All previous training and cancelled evaluation services remain stopped.
This experiment starts both branches from the pinned vanilla VibeVoice-ASR model,
never from the contaminated 250h adapter. Existing raw Soniox results, releases,
checkpoints, and evaluation outputs remain unchanged.

The frozen initial cohort has 61 training recordings (19.9573h): 5.4710h short,
9.4351h medium, 5.0512h long; 12.7356h Czech and 7.2218h Slovak. Each language
has three 45–60 minute training crops. The development panel has eight held-out
recordings, 79.513 minutes total, using Soniox pseudo-references. Twenty-two targeted
regression checks passed, including the original word-splitting failure, audio
admission failures, number normalization and loss/gradient identity.

## What the first run established

Six Czech unseen recordings gave 9.8344% → 9.1495% number-normalized WER against
Soniox pseudo-references. The paired recording bootstrap interval for the change
included regression. DER increased from 14.6513% to 15.0502%, and one adapted
output was invalid JSON. This did not establish a useful ASR improvement.
The target builder also split some words at pauses between Soniox subword tokens.
Correcting that defect is necessary; it does not prove that data defects alone
explain the weak result or that a larger rank will succeed.

## Three independent data checks

1. **Identity and lexical integrity.** Rehash source audio, paid raw response and
   receipt; verify transcription IDs and original pairing. Reconstruct whole
   words before applying punctuation, pause or speaker segmentation. Preserve
   the exact lexical text; no LLM rewrite. Independently compare serialized
   segments to the raw token stream, including token ownership, words, speaker
   IDs and timestamp extents. Quarantine ambiguous intraword timing or speaker
   changes rather than guessing. Check whole-recording/show split isolation.
2. **Acoustic and coverage checks.** Decode the full recording; reject decode
   errors and nonfinite samples; measure clipping, zero samples and energy.
   Cross-check the pinned Silero evidence with word timing. Run pinned AST
   music/noise classification over the full timeline; verified identical old
   classifier evidence may be reused, while PCM checks are rerun. Classifier
   scores are not SNR or verified event annotations. Ordinary non-speech
   gaps are allowed; missing speech and substantial music/noise are excluded.
   No VAD-negative interval is automatically called silence.
3. **Exact training inputs.** Before training, verify the exact tokenizer
   output, content masks, feature counts/hashes and 65,536-token bound. No
   target or waveform truncation. Existing independent Whisper results are
   retained as diagnostics only: disagreement does not veto Soniox labels and
   a second transcription is not required before training.

Thresholds are explicit in `policy.json`; they are conservative heuristics,
not guarantees of accuracy. Underfilled length/language quotas block admission.
They are never weakened automatically to reach 20 hours. Sub-second VAD/word
edge disagreements remain available as evidence; missing-utterance screening
uses the previous pipeline's >=1s interval criterion. Confidence filtering uses
**token confidence**, not the minimum token confidence of every multi-token word.

The first screen is restricted to clean speech conversations. It does **not**
claim to teach the full manuscript's acoustic-event task. Verified event-rich
training is a later, explicitly annotated subset. This avoids adding invented
music/noise/silence targets while the speech recipe is being diagnosed.

Soniox is the accepted label source, following the user's explicit instruction.
Human review and agreement with a second ASR are optional diagnostics, never
admission prerequisites. Ordinary pauses without detected speech remain gaps;
we do not fabricate silence or other event labels. Missing-speech coverage,
clipping, decoding, music/noise and confidence checks remain in force.

## Data and reference contracts

- Aim for 19–20 unique audio hours; no repeated or overlapping training crops.
  Target audio-time mix: 25% short (~10m), 50% medium (~30m), 25% long (45–60m).
  Require >=45m examples in both languages and long examples early in the run.
- Target 70% Czech / 30% Slovak audio time. Show-balanced selection is preferred.
- Exclude all previously reserved evaluation recordings and their shows from
  training, plus the original validation/test shows. Never select on vanilla
  model errors. The old test split is reserved for confirmation.
- Development: 60+ minutes from at least six independent **held-out interviews
  on disk**, following the user's selection. Use corrected, source-paired Soniox
  words/speakers/timing as pseudo-references. Human review is optional; measured
  WER/DER against these references measures agreement with Soniox.
- **Both existing synthetic Czech and Slovak recordings remain fixed markers.**
  Retain their approved references and report each one's WER and DER at vanilla,
  ~10h and ~20h. They are not removed or pooled into the independent-interview
  confidence interval. Meaningful development improvement accompanied by marker
  regression is a warning, not a success to hide in an average.
- Empty `customized_context` for this comparison. A later context experiment
  must use information available before ASR, never a reference-derived prompt.
- Optional review UI: `http://localhost:8767/`. Audio is a hash-verified crop with HTTP
  byte-range seeking. Decisions are saved atomically with source hashes and an
  audit history. No approval is preselected. Rejecting a training crop blocks it;
  editing training text requires a new derived target revision.
- Experiment dashboard: `http://localhost:8767/experiments`.

## Compute and controlled comparison

Run on the existing **two RTX A5000 24GB GPUs**, using two DDP workers. Start with
NF4 QLoRA and BF16 compute, frozen audio encoders/connectors, cached BF16 speech
features, activation checkpointing, chunked LM-head loss and adaptive activation
offload. The same quantization is used for both ranks; this is not a BF16-vs-NF4
comparison. A discarded longest-window update checks memory for each capacity,
but even that preflight is forbidden until data admission passes.

| Setting | Branch A | Branch B |
|---|---:|---:|
| LoRA rank | 16 | 64 |
| LoRA alpha | 32 | 32 |
| Initial adapter B | zero | zero |
| Initial weights/optimizer | vanilla/fresh | vanilla/fresh |
| Modules | q/k/v/o + gate/up/down | same |
| LR calibration | 3e-5 and 1e-4 | 3e-5 and 1e-4 |
| Data order / target exposure | identical | identical |
| Optimizer | AdamW, weight decay 0 | same |

Each LR calibration is six matched DDP updates, discarded afterward.
A measured quality stop ends that capacity branch and preserves its checkpoint;
the other requested capacity is still attempted. Integrity or execution failures
stop the service for investigation. Choose the
lower held-out content loss among safe trials; if neither is safe, stop. Each
20h probe restarts from vanilla with the selected LR and fresh optimizer.
The first LR calibrations run before the full vanilla decoding panel. Vanilla
decoding is cached once and reused for both ranks; the fixed CS/SK vanilla markers
are already cached with verified input/model/reference provenance.
WSD uses a target-token clock: 10% warmup, stable to 80%, final 20% decay to
10% of peak. Rank scaling remains alpha/r, with alpha fixed across ranks.
Do not add a new optimizer as another changing variable in this first diagnosis.

Checkpoints include adapter, optimizer, every rank's RNG and the immutable plan
and runtime hashes. A restart can resume only its own exact contract. Keep all
previous-run pause markers intact.

## Trajectory, stopping and saturation

- Record token-weighted train and held-out loss, **content loss separately from
  JSON/timing/speaker structure loss**, LR, gradient norm, tokens/audio seen,
  wall time, cache time, compute time and peak GPU memory.
- Save and validate every five updates. Stop for nonfinite gradients/loss or
  content loss >15% above the initial value on two checks.
- At ~10h, checkpoint and release the training models before decoding. Evaluate
  a fixed short development panel plus **both CS/SK markers**. Stop automatic
  continuation for >15% relative dev-WER regression, >1pp dev-DER regression or
  >0.5pp WER regression on either fixed marker. Flat small-panel results are
  inconclusive; the bounded 20h screen can continue.
- At ~20h, decode the full Soniox-reference dev panel plus both markers. Test the longest
  training-source examples for JSON/termination stability; label that as a
  structural stress test, not an independent long-form accuracy benchmark.
- Score strict and number-normalized WER (Czech `tri` is not `tři` or `3`), DER
  with 250ms collar, and raw JSON validity. Never silently repair invalid JSON or
  drop failed samples from rankings. Aggregate counts, not per-clip percentages.
  An observed `[Speech]` placeholder is an explicit transcription abstention:
  retain the raw tag, give it no recognized words or diarized speaker, and score
  against the complete reference. Missing words and speaker coverage remain
  errors. Ordinary speech without a speaker is still invalid. Existing vanilla
  outputs were rescored on CPU under this rule with original results archived;
  inference, references and training contracts were unchanged.
- Compare with paired **recording-level** bootstrap intervals (20k draws).
  A practical screening target is >=20% relative normalized-WER reduction,
  interval excluding zero, <=0.5pp DER degradation and stable long outputs.
  Markers remain separately visible and must not materially regress. This is a
  decision rule, not a promise or confirmatory significance claim after tuning.
- Capacity saturation is assessed from the rank16/rank64 **matched-token
  learning curves**, content-vs-structure loss, dev decoding and train/dev gap.
  Rank64 improving training content loss without dev WER improvement points to
  generalization/labels/domain rather than demonstrated useful capacity.
  Both flattening is not proof of saturation after one short pass. Extend only
  a justified winner, then test untouched recordings. No automatic 50h/500h run.

W&B logs numeric metrics only to the existing account/project. Local events and
results are durable if W&B is offline. Cloud migration can be considered only
after trajectory is useful, with the dataset, pinned code/environment, adapter,
optimizer and RNG states transferred together.

## Commands on mlserver

Working directory: `/data/work/asr-experiments/qda-restart-v3`.

```sh
# Recompute data admission; this FAILS until all requirements are actually met.
/data/work/asr-pilot/venv/bin/python gates.py --data /data/work/asr-datasets/qda-clean-v3

# After admission: exact inputs, frozen features, discarded memory preflights.
/data/work/asr-pilot/venv/bin/python launch.py prepare

# Both bounded experiments, fresh from vanilla; no automatic scale-up.
/data/work/asr-pilot/venv/bin/python launch.py both
```

Methodology references:
[VibeVoice-ASR report](https://arxiv.org/abs/2601.18184),
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/),
[Microsoft fine-tuning guide](https://github.com/microsoft/VibeVoice/blob/main/finetuning-asr/README.md).
Their results motivate checks and controlled experiments; they do not establish
that this Czech/Slovak QDA adaptation will improve.
