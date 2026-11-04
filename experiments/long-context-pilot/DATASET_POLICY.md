# Dataset policy for QDA adaptation

The existing `dataset/` and `cache/` are **speech-only hardware-pilot artifacts**. Their labels are not the approved recipe for the complete corpus. Keep them immutable so the existing measurements remain interpretable. The richer annotation audit is under `annotation-audit/`.

## Preserve the complete session

Store original audio identity/hash, source offsets, recording/show split, word-level Soniox evidence, session speaker IDs, events, overlap, context provenance, model revisions, and verification status. Train/validation/test assignment happens before windowing. Related windows and context variants stay in the same split and count as one source's exposure.

Do not remove non-speech audio and close up the timeline. Do not merge unrelated recordings into a purported conversation. Windows should end at safe word/turn boundaries; preserve timestamps relative to the selected window and retain original offsets. Do not drop a partially crossing word without recording a boundary exclusion or choosing a better boundary.

For QDA, retain verbatim wording, hesitation, repetitions, dialect, and intelligible overlapping speech. An LLM may propose punctuation corrections or flag inconsistent terminology; it must not silently paraphrase participant answers.

## Separate speech detection from acoustic classification

1. Soniox provides transcript and recording-level diarization evidence.
2. Silero supplies an independent speech timeline. A gap in either output is not proof of silence. Audit both speech without labels and labels outside VAD speech.
3. Inspect the remaining audio with an acoustic-event model. The pilot runs an AudioSet AST scan over the full timeline as well as individual gaps. Its class scores are uncalibrated triage evidence, not accepted annotations.
4. Review/cross-check uncertain intervals, missed speech, overlap, music under speech, and speaker changes. Distinguish intelligible speech from genuinely unintelligible speech; ASR disagreement alone does not establish unintelligibility.
5. Accepted targets use the report's tags: `[Unintelligible Speech]`, `[Music]`, `[Human Sounds]`, `[Environmental Sounds]`, `[Noise]`, `[Silence]`. `target_format.py` supports speakerless events exactly as the upstream formatter does. Assign an event a speaker only when the source identity is supported.

Keep speech and background-event evidence separately when they overlap. Background music does not erase the spoken transcript. Short breaths are not necessarily meaningful annotation units: calibrate the granularity on reviewed examples, rather than tagging every 100 ms pause.

`audit_audio.py` preserves Soniox events if present, flags missing speaker labels, and records suspected missing speech. `classify_events.py` never converts its proposals into training truth. A record with unresolved material gaps is not ready for final long-form supervision. Neither the student's own transcript nor a low-energy heuristic is an independent gold reference.

The paper's speech-occupancy and independent-ASR filtering thresholds were used in a particular pretraining pipeline. Do not blindly apply them to every QDA or auxiliary sound-only example: quiet interviews and deliberately selected music/silence examples have different purposes.

## Length mixture and curriculum

Use total **audio-prompt + context + answer tokens** for allocation. Keep 8K, 16K, 32K, and 64K buckets, with variable durations within each. The approximate 10/30/60-minute pilot views are capacity probes; they are not a finished curriculum. Add short conversation windows and short event-only examples, while retaining intact 10–60-minute sessions for the target task.

A proposed ablation, not an established optimum or the paper's exact LoRA schedule:

- First 10% of updates: 8K/16K buckets, equally likely.
- Next 20%: 8K/16K/32K with probabilities 0.2/0.3/0.5.
- Remaining 70%: 8K/16K/32K/64K with probabilities 0.1/0.2/0.3/0.4.
- Keep a separate, bounded auxiliary stream of verified acoustic-only examples. Start by testing a 5% update share; measure false speech and event recall before selecting its final weight.

These are **batch-selection probabilities**, not unique audio-hour shares. Log actual processed audio, supervised tokens, and exposure per source in addition to steps. Select the same length bucket across DDP ranks to avoid one GPU waiting on an hour-long example while the other processes a short clip. Within each bucket, balance language/domain and cap repeated exposure to one source. The current pilot trainers are fixed-bucket benchmarks; this mixed schedule has not been trained or selected yet.

Because the starting model already supports long context, compare gradual introduction against mixed-length training from the start. Do not assume the model must relearn its entire pretraining curriculum. Select on held-out 10-, 30-, and 60-minute sessions, not just a pooled short-clip score. The one-hour sample cannot provide independent training and held-out full-hour sessions simultaneously.

## Context without answer leakage

Accept context that would exist before transcription: study glossaries, episode metadata, organization/topic names, and participant lists. Record its source and cutoff time. Keep participant names as vocabulary hints unless acoustic identity mapping is separately known. Reject reference-derived sentence lists and participant answers.

Mix absent, helpful partial, and plausible irrelevant context during a future controlled experiment. A starting ablation is 50% absent / 40% supplied / 10% distractor; this is a proposal, not a measured best mixture. Cap context tokens, vary order, and retain context dropout. Context and length sampling are independent dimensions.

`sample/contexts.json` contains title-derived terms only. `build_context_cache.py` prepares three variants with shared audio features. No-context inputs were verified token-identical to the original cache, and all answers remain unchanged. Default training benchmarks still select no-context rows; do not count all three variants as independent hours or claim that mixed-context training has already been tested.

## Acceptance and evaluation

Before scaling labels, review a stratified subset of speech, gaps, overlap, low-confidence words, speaker switches, language switches, and each acoustic tag. Evaluate an independent ASR on flagged speech; disagreement is a review signal rather than an automatic correction. Report source-level coverage and rejection reasons.

Evaluate WER, cpWER, tcpWER, DER, event precision/recall and temporal overlap, false spoken words during verified non-speech, JSON/schema validity, repetition, end-of-session coverage, and context-induced insertions. Keep no-context, supplied-context, and distractor-context evaluations separate. Current Soniox comparisons are pseudo-reference checks and do not measure acoustic-event correctness.

The small adapters have not improved the available held-out comparison. Do not promote them merely because they train successfully. The hardware recipe is feasible; label quality and representative evaluation remain the next selection criteria.
