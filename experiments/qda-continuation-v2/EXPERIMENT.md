# QDA continuation v2: AdamW versus PoLoRA

Status: all 49.9080 hours transcribed and durably archived; checksummed release copied to mlserver. CPU plan and pairing/split/token audit passed. GPU caching and preflight wait for the existing QDA496 run to finish. Preparing this experiment does not start its training runs.

## Question and scope

Can a short continuation on fresh conversational Czech/Slovak audio improve word recognition beyond the completed rank-16 baseline? Compare two practical continuation recipes at the same rank: AdamW and PoLoRA. This is conditional on an AdamW-trained starting checkpoint; it does not establish an optimizer ranking from scratch or prove rank saturation.

The first stage is frozen at **19.7638 hours, 41 paired updates and 720,932 supervised tokens per contender**. Exposure is 13.8604 hours CS and 5.9035 hours SK. Actual durations comprise 5.1492 hours up to 11 minutes, 10.3021 hours from 11–31 minutes, and 4.3126 hours above 31 minutes. Both languages include an intact recording over 45 minutes. No full rerun, automatic extra epoch, rank sweep, or automatic promotion follows. Remaining usable audio is retained for a later, explicitly selected extension.

## New data and provenance

Frozen selection: `datasets/soniox-50h-v2/selection.json` and its SHA256, copied into `datasets/releases/qda-50h-v2/`.

- 85 intact recordings, 49.9080 hours; 34.8904 Czech, 15.0176 Slovak; 73 shows.
- Initially 48.6054 hours were selected for training and 1.3026 hours for development. Quality gates retained **46.2459 fresh training hours** (55 CS and 24 SK recordings) and **0.5445 fresh CS development hours**. Five recordings were excluded: two for language mismatch and three for untranscribed detected speech; all their paid artifacts remain archived.
- The unused pool had no suitable SK validation recording after decode checks. Development therefore also includes **six existing validation-only recordings**, three per language, totaling 4.5637 hours. They never entered baseline training and exclude the baseline loss-monitor recordings. These required no new Soniox requests. Total eligible development source duration is 5.1081 hours; the evaluation plan samples one middle ~10-minute window per source plus a separate longest whole session per language.
- Every **newly paid** audio hash, source ID and normalized show/title pair is absent from the original 500-hour selection. Original pilot sources are excluded. Existing show-level train/validation/test assignments are preserved; original test shows are untouched.
- Each selected source passes full-file decoding, duration and SHA256 checks before a paid request. Selection favors conversational metadata, 10–60-minute sessions, length diversity and at most 2.5 hours per show.
- Soniox `stt-async-v5`, speaker diarization and language identification; CS/SK language hints. Maximum 80 account-wide outstanding jobs and 8 GB uploaded storage ceiling. Files are deleted remotely only after both local replicas and pairing receipts verify.
- Internal write-ahead archive: `/home/lofcz/.local/state/asr-soniox/soniox-50h-v2`; SSD archive: `datasets/soniox-50h-v2`. Ambiguous POST responses are reconciled or held, never blindly resubmitted.
- Silero ONNX full-timeline speech audit runs independently. The release preserves raw tokens, confidence, exact teacher text, source hashes, IDs, timestamps, speaker IDs, request/response metadata, review flags and checksums.
- Structural errors, >20% low-confidence tokens, >15% non-CS/SK text, or excessive untranscribed detected speech exclude a source from the training catalog. The paid archive and original audio remain preserved even for excluded sources.

This is the same `speech-diarization-pseudo-v1` supervision profile as the baseline. It is **not human-verified gold** or a fully event-supervised dataset. Silence, music and noise are not inferred from VAD negatives. Review candidates remain available in `review-queue.jsonl`. Both contenders use empty `customized_context`; source-only metadata is preserved for a separate experiment.

The development panel consists of real source recordings from the fresh selection and the original validation split, not production QDA interviews. Its scores measure agreement with Soniox and do not establish human transcription accuracy. The two existing user-approved CS/SK dialogue fixtures remain a separately reported anecdotal gold guardrail. The frozen small panel contains seven independent recordings, one middle ~10-minute window each. A broader manually verified QDA benchmark remains needed.

## Storage and efficiency

Audio is stored once per recording, with SQLite indexes and source-offset windows. Tokenization and encoder outputs are cached per deterministic window, with source/label/input/model/code hashes. Only the initial trial and evaluation windows need GPU feature caching; the entire usable release is token-audited and available in the full plan.

The current encoder/connector weights are frozen. Reusing their output therefore avoids repeated encoding while preserving the fixed per-window latent seed. Changing the encoder, augmentation, tokenization, context or crop requires a new cache contract. Arrays store exact BF16 bit patterns, not lossy feature compression.

The planner balances actual audio duration alongside language; a nominal one-hour view of a short recording is not counted as long context. Medium and long context windows are mixed throughout the continuation. Similar token lengths are paired across GPUs to reduce synchronization idle time. Both options use the identical immutable `plan.json`, ordering, initial weights and exposure. Different cached views never become separate train/validation recordings.

## Shared settings

- Original VibeVoice-ASR model revision `d0c9efdb8d614685062c04425d91e01b6f37d944`.
- NF4 QLoRA, BF16 compute; rank 16, alpha 32, dropout 0.05.
- Attention q/k/v/o and MLP gate/up/down adapters; encoder/connectors frozen.
- Same token-weighted causal objective; two A5000 GPUs, one window per rank. No padding or overlapping full-epoch views.
- Fresh optimizer state in **both** branches; identical pinned final baseline adapter. This avoids granting AdamW accumulated state while starting PoLoRA fresh.
- Weight decay 0 for both short continuation recipes. This is a deliberate shared change from the first run's 0.01, not an optimizer-specific advantage.
- Gradient clipping 1.0; existing chunked MLP/loss and adaptive activation offload retained.
- Maximum 65,536 tokens. Oversized source views fall back to shorter intact partitions before any GPU run; no transcript truncation.

## Bounded calibration and training

| Setting | AdamW | PoLoRA |
|---|---|---|
| Initial safety-probe LR | 3e-5 | 1e-3 |
| One fallback, only if rejected | 1e-5 | 3e-4 |
| Safety probe | Same six short paired updates from baseline | Same |
| Probe acceptance | Finite training and <=10% increase in held-out total/content loss | Same |
| Main experiment | Fresh restart from baseline, ~20 audio hours | Same |

Numerical LR values are optimizer-specific. Both preserve PEFT scaling alpha/r = 2; rates are calibrated for this parameterization, not copied as equivalent rates from a paper. These brief probes reject unsafe settings; they do **not** estimate an optimal learning rate. At most two rates per optimizer are attempted. Runtime errors stop the launcher for inspection; only a failed finite quality check triggers the lower-rate fallback.

The main schedule is warmup–stable–decay, driven by supervised target tokens, not audio hours or step fraction. Warmup spans up to the first six updates (at least one, fewer in a short plan). The final 20% of target-token exposure cools down to 10% of the peak LR. A checkpoint before cooldown is retained, in addition to the final checkpoint, for future continuation without losing the stable-phase weights.

## Measurements and decision gates

Every update records total loss, content-token loss, structure/timing/speaker-token loss, LR, gradient norm, processed tokens, audio exposure, elapsed time, optimizer overhead and peak VRAM. Content masks cover JSON `Content` string values, including punctuation and mixed-boundary tokenizer tokens; this is a diagnostic, not a new training objective or a WER surrogate.

- Small fixed held-out loss panel every 10 updates; full development loss panel before and after each main run.
- Number-normalized WER, strict WER and DER against the held-out Soniox development references. Same crops, greedy decoding, BF16 evaluation, normalization and deterministic audio features for baseline and both contenders.
- Complete held-out recordings (44.6 minutes CS and 57.7 minutes SK) are decoded separately to check long-session behavior. These overlap the development window panel and are not additional independent evidence.
- Existing approved CS/SK gold panel is scored separately. Existing baseline inference is reused when its adapter hash matches. Vanilla inference is not repeated.
- Incomplete generation, invalid JSON, invalid timestamps, missing metrics or missing samples fail evaluation; never average only successful samples.
- Development and gold results are separated in logs and W&B. Only scalar metrics/configuration are sent to W&B; audio and transcripts remain local/server files.

A contender is flagged **eligible for manual extension**, never auto-promoted, if it improves development normalized WER by at least 0.1 percentage point, increases development DER by no more than 0.25 point, increases the small gold-panel WER by no more than 0.1 point, and passes the same WER/DER regression limits on full-session checks. These are practical screening thresholds, not statistical-significance claims. Per-language results and error patterns must be reviewed before choosing an extension.

Training stops early if held-out content loss is >15% above its initial value on two successive checks. A short inconclusive trial does not disprove longer-horizon optimizer or rank benefits.

## Durability and implementation

PoLoRA is vendored at the commit recorded in `vendor-provenance.json`, with its Apache-2.0 license. The upstream numerical update is unchanged. Its upstream `state_dict()` omits custom optimizer state; `optimizers.py` adds versioned serialization of all momentum, curvature and power-iteration state, plus shape/hyperparameter checks. Interrupted-versus-uninterrupted continuation tests verify exact equality for float32 and BF16 factors, including independent RNG states. Another test confirms content diagnostics preserve the original loss and gradients.

All branch checkpoints include adapter weights, complete optimizer state, both ranks' CPU/CUDA/Python/NumPy RNG states, exact token position and immutable configuration. Writes are atomic; checkpoints are retained. Resume rejects changed data, code, baseline or schedule. Original training and checkpoint files are never modified by this experiment.

## Commands on mlserver

Directory: `/data/work/asr-experiments/qda-continuation-v2`.

```bash
# Inspect preparation and experiment outputs.
/data/work/asr-pilot/venv/bin/python launch.py status

# Cache features and perform both longest-window GPU preflights after the baseline ends.
# This command does not launch either training experiment.
/data/work/asr-pilot/venv/bin/python launch.py prepare --wait

# Explicitly run both bounded experiments as a persistent service, using both GPUs.
# This unit is installed but is NOT enabled or started during preparation.
systemctl --user start qda50-v2-experiments.service

# Equivalent foreground command:
/data/work/asr-pilot/venv/bin/python launch.py both

# Or select just one; the same baseline/plan are used.
/data/work/asr-pilot/venv/bin/python launch.py adamw
/data/work/asr-pilot/venv/bin/python launch.py polora
```

GPU admission requires the baseline's completion marker, inactive baseline/evaluation services, and idle GPUs. An experiment-wide lock prevents concurrent launchers. `qda50-v2-prepare.service` prepares the cache and runs discarded preflight updates automatically after baseline completion. It does not run either experiment. The experiment service remains unstarted. Only an explicit experiment service start or `both`, `adamw`, or `polora` command trains the probes.

`data-audit.json` passed for all 91 original/supplemental paid pairings, source isolation, contiguous train windows, and 201 cached input windows. Seven CPU tests passed, including exact optimizer resume and causal loss/gradient equivalence. Real GPU memory/finite-update preflights remain deferred until baseline completion. `preparation-status.json` reports waiting/caching/preflight/ready state.

`logs/` contains durable subprocess output; `runs/` contains complete checkpoints and update metrics; `evaluations/` retains references, raw generations, metrics and errors. `<optimizer>-result.json` gives the final comparison. W&B run links are in each run's `wandb-status.json`, grouped under `qda-continuation-v2`.

Planning estimate from the first run: 25–50 minutes of training per 20–30-hour candidate on both GPUs. Calibration, one-time caching, long-context decoding and evaluation are additional. This estimate is provisional until the automatic preflight measures the new workload; no published optimizer speedup is assumed.

## Research

- [Thinking Machines: LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): compare matched exposure and account for optimization; early agreement does not rule out later capacity divergence.
- [PoLoRA method and experiments](https://parshakova.github.io/polora/), [official implementation](https://github.com/nikhilgsh/polora): promising text-SFT evidence; no validated VibeVoice/long-audio speedup is assumed.
