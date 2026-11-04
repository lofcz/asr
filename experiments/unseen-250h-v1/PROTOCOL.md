# Unseen recordings: vanilla versus 250-hour checkpoint

User-requested diagnostic evaluation. Training is paused at resumable checkpoint `checkpoint-000515`, corresponding to 250.2181398 audio hours. No training, continuation, or automatic resume is authorized by this evaluation.

The panel was frozen before generation: ten fresh recordings from `qda-50h-v2`, six Czech and four Slovak. One intact source-offset window nearest each recording's center, approximately ten minutes each: 100.2405 minutes total. No source audio hash occurs in the original 500-hour paid selection. No source occurs in the frozen 19.76-hour continuation probe. Nine series were represented in original training and one in original validation, so this is recording-disjoint, not a claim of unseen speakers or series.

Selection uses source coverage and speaker counts, before model predictions. The references are durable Soniox pseudo-labels; no human-gold claim is made. Original text, source hashes, exact offsets, references, output JSON, raw decoded text, and per-word errors are preserved. These ten entire recordings are reserved from future training extensions in `qda-continuation-v2/reserved-evaluation-records.json`.

Both models use the same pinned VibeVoice revision, BF16 inference, greedy decoding, empty customized context, and identical frozen speech features. The fine-tuned model merges the rank-16 adapter from checkpoint #515. Vanilla runs on GPU 0 and fine-tuned on GPU 1. Their audio, input, feature and reference hashes must match before aggregation. No new Soniox calls are needed.

Metrics: existing strict WER and conservative language-specific cardinal-number-normalized WER; DER with a 250ms collar, including missed speech, false alarms and speaker confusion. Diacritics are preserved. Czech `tri` is not normalized to `3`; Czech `tři` and Slovak `tri` are normalized within their respective language profiles. The same existing scorer is used for both variants.

Scoring correction during generation: the initial output validator incorrectly required a speaker on bracketed non-speech events such as `[Music]`. `rescore.py` accepts these valid speakerless events and preserves the initial inference receipt as `result.inference.json`. Speech still requires a speaker; EOS, JSON structure, timestamps and reference hashes are checked. The established WER/DER scorer already excludes bracketed events, so this correction changes validation only, not the metric definition or decoded output. Each rescored receipt records the scorer hash.

Supplementary recovery policy added after observing the checkpoint's `cs-04` output: repeated all-zero decimal timestamps such as `0.0.0` can be replaced by `0.0` in a separately saved copy. `recover.py` applies exactly this rule to either variant, records every edit and original-output hash, and never changes transcript wording or speaker IDs. Original output failures remain failures. `report.py`/`summary.json` retain strict validity gating; `analyze.py`/`analysis.json` additionally report scores including the documented repairs, alongside raw valid-output counts. The same paired inputs and metric definitions apply. Unrecoverable outputs prevent a whole-panel aggregate; they are not silently dropped.

Aggregate WER weights by reference word count; aggregate DER weights by scored speaker duration. Report per-language and per-recording results, and paired 95% bootstrap intervals using 20,000 resamples of whole recordings. Invalid/incomplete outputs are failures, not silently dropped. An incomplete panel receives no aggregate ranking.

The comparison tests ten-minute excerpts, not complete hour-long sessions. Agreement with Soniox is not identical to human transcription accuracy. Interpret changes on this panel, and review representative changed errors before proposing another long run.

Server: `/data/work/asr-evaluation/unseen-250h-v1`; unit: `qda250-unseen-eval.service`. `run.py` never starts training. The training service has a persistent `ExecCondition` guarding `paused-by-user.json`, and the periodic evaluation timer and continuation preparation service are stopped.

Resumption requires a new user instruction. Once requested, preserve this evaluation, remove the pause marker, and start the original training service plus its periodic evaluation timer. The optimizer and RNG state remain in checkpoint #515. Continuation preparation should only be restarted when consistent with the chosen next experiment.
