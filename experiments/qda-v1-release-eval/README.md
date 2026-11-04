# Temporary QLoRA v1 release evaluation

This experiment compares the v8 300h checkpoint (step 1729), 450h (step 2564), final (step 3110), and vanilla. It does not train or modify any checkpoint. Full-weight AdamW work is deferred until this release comparison is complete.

The fixed panel contains 32 approximately ten-minute recordings: 16 Czech and 16 Slovak, totaling 319.12 minutes. It includes the eight existing development clips, 16 additional development recordings, and eight previously unused confirmation recordings. Two additional sessions last about 30 and 55 minutes. Both fixed synthetic markers remain separate reported measures.

Selection uses a fixed seed and balances shows without filtering speaker count, acoustic quality, Soniox confidence, or vocabulary. All selected recordings and shows are absent from the completed v8 training plan. Confirmation recordings used here are consumed for release selection and must not later be described as an untouched cloud-training test set.

New clips use whole Soniox segments and source PCM sample indices, avoiding container-seek discontinuities. Each audio file is checked against its source SHA-256; target label hashes are recorded. Identical frozen features, prompts, BF16 merged weights, and greedy decoding are used across candidates. Complete earlier decodes are reused only after checking adapter, audio-input, feature, model-revision and reference identities and reproducing their metrics.

Every clip saves its raw output, reference, metrics or explicit failure, word edits, and provenance. A repeating output is stopped after at least 256 exactly periodic tokens; a period need not divide 64. Other incomplete or invalid outputs remain failures, not missing observations. WER is never invented for unparseable output. Report completion counts alongside scored-clip metrics.

The final comparison reports number-normalized WER, ordinary WER, 250 ms DER, language and cohort results, recording- and show-cluster paired bootstrap intervals, long-session failures and markers. The recommendation favors completed evaluations and lower balanced-language WER; statistically unresolved text differences are resolved using long-session completion then DER. The intervals are descriptive during checkpoint selection. They are not a blind confirmatory significance claim.

`run.py` resumes completed work and uses both A5000s. Results are progressively mirrored to this SSD directory by `mirror.py`. No Soniox calls, paid API calls, training, or deployment are performed. At completion, `v1-candidate.json` identifies the selected adapter and any long-session limitation. The original adapters remain in the SSD checkpoint archive.

Live page: http://localhost:8766/release-evaluation/

Remote service: `qda-v1-release-eval.service`

Local mirror: `asr-v1-release-eval-mirror.service`

Final artifacts: `report.md`, `report.json`, `word-errors.csv`, `v1-candidate.json`, plus raw per-recording results under `evaluations/`.
