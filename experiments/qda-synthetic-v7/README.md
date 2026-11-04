# Cartesia Czech/Slovak domain corpus

This prepares synthetic ASR data. It does not start training. Cartesia's native word timestamps are canonical; no second ASR model or forced-alignment pass is run on these recordings. FP32 remains the optional offline alignment choice for other audio; INT8 is the selected inference choice.

## Scope and content

- 60 hours planned: 50 hours core, with 12.5 hours each of one, two, three and four speakers; plus 10 hours extra single-speaker material. Czech and Slovak each have 30 planned hours.
- Completed general QDA is retained. The remaining recordings prioritize Scio and AI vocabulary in varied QDA situations. The original 20/20/20 topic split is superseded by vocabulary-v4; language, speaker-count and total-duration targets remain unchanged.
- 872 planned recordings, spanning 30 seconds to 60 minutes, with 2,072 speaker slots. The 74 already paid recordings were preserved during the expansion.
- All 52 sourced Scio product/term families have explicit corpus targets in EACH language: 200 for named products/core families and 100 for the longer catalogue entries. All 381 AI terms are covered: 50 for initialisms/named tools and 25 for other technical terms. Exact per-term targets are saved in vocabulary/catalog.json. These are coverage targets, not established thresholds for model learning. General QDA draws from 210 seed scenarios.
- vocabulary/plan.json subtracts observed completed coverage and immutable partially submitted text, then allocates the deficits across remaining recordings. Every future part has a hashed contract listing term counts and required forms. Whole parts stay within one domain; prompts require coherent discussion rather than lists of unrelated terms. Product descriptions and provenance come from official URLs in scio-products.json.
- New scripts are saved under scripts-v4/ and revalidated immediately before paid TTS. Old unpaid drafts cannot bypass the new contracts. The start gate binds structural tests and actual script checks to the current vocabulary plan hash. Existing paid request hashes and source text remain unchanged.
- vocabulary-v4.1 supersedes the unsubmitted flat-quota stress test. That test produced unnatural repetition and was archived without purchasing its long-form audio. Required term repetition is capped at six occurrences per part; planned mandatory vocabulary is capped at 38% of words, with actual aggregate requirements around 29% Czech and 19% Slovak. Prompts require meaningful connected passages, not repeatedly enumerated term bundles. Minimum counts and script validity are checked automatically; these checks alone do not establish conversational naturalness or acoustic correctness.
- Script generation uses noninteractive Codex CLI, exactly gpt-6.1-sol with medium reasoning. Original fictional experiences only; no evaluation transcripts in prompts. Canonical written brand names and acronyms are retained in targets.

The original coverage audit found ScioLink missing and insufficient inflection diversity. domain-coverage-initial.json preserves that evidence. The live dataset domain-coverage.json and domain-coverage.md update asynchronously as audio accumulates and count actual completed training labels, observed forms, required-term failures and voice use. They distinguish actual coverage from planned requirements.

## Voices and timestamps

The completed bank contains 1,000 registered voices: eight native library voices and 992 private instant clones from the existing training corpus. There are 1,769 usable isolated speaker clips; a TitaNet similarity screen suppresses likely repeated speakers across recordings. Acoustic similarity is a duplicate-screening heuristic, not a guarantee of different human identities. Source IDs, speaker IDs, crop coordinates and hashes are retained. No held-out audio is used for voice references.

New recordings choose unused voices first, then voices with the least reserved audio. A speaker ID always maps to one fixed voice within a recording. Identities rotate between recordings while preserving the requested speaker-count mix. Native library voices are Jan, Marek, Milena, Tereza, Jana, Petr, Peter and Katarina.

Cartesia sonic-3.6-2026-08-27, API 2026-08-14, returns 24 kHz mono PCM and native word timestamps. New vocabulary-v4 requests use original-text timestamps, preserving technical names containing digits and symbols. Legacy requests retain their original normalized-timestamp mode and content hashes. Known silent PCM gaps and exact sample offsets compose each recording. Speaker labels come from assigned voice IDs. Native zero-duration word points are retained and identified in the word sidecar; segment durations remain positive. No word boundaries are inferred from silence gaps.

## Durable artifacts and training format

SSD: datasets/cartesia-qda-v7/. Independent replica: /home/lofcz/.local/share/asr-synthetic/cartesia-qda-v7/. Completed assets are copied progressively to mlserver:/data/work/asr-synthetic-v7/.

Each paid TTS request has a content hash. Before POST, request, intent and character reservation are committed to both local disks. SSE events are progressively fsynced to both disks. Original WAV, native words, response and pairing receipts are immutable. Completed responses can be recovered from either event journal. Ambiguous paid requests are never blindly resubmitted. Explicit rate limits back off. Credentials stay in the existing Tazatelka environment file.

- scripts/ preserves prompts, schemas, attempts and accepted text.
- utterances/ preserves every paid provider result.
- sessions/ contains staging audio and native source composition. Its historical 16 kHz audio and uppercase segment keys are not the final training interface.
- training/audio/ contains lossless 24 kHz FLAC reconstructed from original PCM.
- training/labels/ contains the VibeVoice input schema: audio_duration, audio_path, lowercase segments with speaker/text/start/end, and customized_context. Optional language metadata is an array.
- training/words/ contains native word timing mapped to written transcript tokens.
- training/receipts/ binds audio, labels, words and source composition by SHA-256 and records the speaker-to-voice mapping.
- training/manifest.jsonl is the local export index. The root training-manifest.jsonl is the published transfer snapshot, written only after its assets and receipts have arrived on mlserver.

Nine historical phonetic acronym occurrences (OSP, AI, API, GPU) are normalized only in the derived export. The original paid scripts/audio/native words remain intact, and each correction is recorded in its export receipt. A grouped acronym receives the union of its original native word timestamps. Diacritics are preserved; tri and tři are never collapsed.

Checks cover decoding, clipping/level, exact sample counts, paired hashes, known voice assignments, word and segment bounds, transcript reconstruction and required domain terms. Paid artifacts are retained when a review is needed. These checks establish structural consistency; they do not prove perfect TTS pronunciation.

## Services and budget

Generation: asr-cartesia-v7.service, twelve script workers and a ceiling of fifteen concurrent TTS requests. A shared limiter reduces concurrency on explicit provider 429 responses; concurrency.json reports configured, effective and peak values. The queue admits at most ninety active plus waiting TTS tasks. Export/coverage: asr-cartesia-export-v7.service. Progressive copying: asr-cartesia-alignment-sync-v7.service (legacy name, no alignment). Measured duration completion: asr-cartesia-quotas-v7.service. Voice expansion is complete.

The quota worker fills actual duration deficits with newly generated content under fresh vocabulary contracts; it does not loop or stretch audio. Final fulfillment also requires the completed-label vocabulary audit. Coverage counts actual completed exports, reports distinct voices per term, and caches by label and catalog hashes. Training stays stopped until the mixed real/synthetic dataset is ready for the separately requested fresh continuous run.

Codex account usage-limit responses set scripts-control.json to quota_blocked. Workers reuse accepted scripts and finish cached TTS, but launch no further Codex calls. A quota rejection does not consume a content-validation attempt. After draining usable scripts, the generation service exits successfully with waiting_for_script_quota instead of repeatedly restarting. The provider message and all failed attempts are preserved. Once the account quota is restored, set scripts-control.json status to ready through core.save, then start asr-cartesia-v7.service; the same model, contracts and paid request hashes are retained. No alternative provider or paid API fallback is selected automatically. Transfer errors now retain the actual rsync stderr so SSH outages are visible separately from generation.

The TTS reservation ceiling is 5,000,000 characters against the user-stated 9,075,783-credit balance. TTS costs approximately one credit per character, rather than three credits per second; provider normalization can affect metering. Instant voice cloning is included on the paid plan. No subscription or overage setting is changed. Sources: https://docs.cartesia.ai/pricing and https://www.cartesia.ai/vs/cartesia-vs-elevenlabs .
