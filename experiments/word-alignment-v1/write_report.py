"""Render the measured comparison without hand-copying metric values."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent/'benchmarks/soniox-v1'
data = json.loads((P/'summary.json').read_text())['samples']
manifest = json.loads((P/'manifest.json').read_text())
names = {'cs': 'Czech marker', 'sk': 'Slovak marker', 'cs-natural': 'Czech natural podcast', 'sk-natural': 'Slovak natural podcast'}
total_words = sum(s['paired_words'] for s in data)
total_seconds = sum(s['models']['canary']['runtime']['audio_seconds'] for s in data)
lines = [
    '# Word alignment comparison against Soniox', '', 'Measured 2026-10-24 on mlserver, one RTX A5000 24 GB.', '',
    '**Recommendation: Canary auxiliary CTC with lexical-token word boundaries is the better overall candidate in this pilot.** '
    'It improves mean boundary agreement and usually the error tail. The 300M models remain competitive for word starts; '
    'the Slovak marker improvement is small. Raw NFA word boundaries that include punctuation are not a consistent upgrade.', '',
    f'{total_seconds/60:.2f} minutes across four recordings; {total_words:,} paired words / {total_words*2:,} scored boundaries. '
    'Two synthetic production markers plus two natural podcast excerpts. Existing Soniox responses were reused; no paid transcription calls.', '',
    '## Mean absolute boundary deviation from Soniox', '',
    'Each word contributes one start and one end error, equally weighted. Lower is better. All three columns use exactly the same words.', '',
    '| Sample | Paired words | 300M WhisperX | Canary raw NFA | Canary lexical boundaries |',
    '| --- | ---: | ---: | ---: | ---: |',
]
for s in data:
    a,b=s['models']['wav2vec2'],s['models']['canary']
    lines.append(f"| {names[s['sample']]} | {s['paired_words']} | {a['boundaries']['mae_ms']:.1f} ms | {b['raw_nfa_including_punctuation']['mae_ms']:.1f} ms | {b['boundaries']['mae_ms']:.1f} ms |")
lines += ['', 'Lexical boundaries use the first and last token containing letters or digits. Standalone commas, periods, quotation marks and blank tokens do not extend a spoken word into silence. The full original NFA timestamps and token paths are retained. This is a word-boundary definition, not an offset fitted against Soniox.', '',
          '## Starts, ends and tails', '',
          '| Sample | Start MAE 300M → Canary | End MAE 300M → Canary | P90 300M → Canary | Within 100 ms, 300M → Canary |',
          '| --- | ---: | ---: | ---: | ---: |']
for s in data:
    a,b=s['models']['wav2vec2'],s['models']['canary']
    lines.append(f"| {names[s['sample']]} | {a['starts']['mae_ms']:.1f} → {b['starts']['mae_ms']:.1f} ms | {a['ends']['mae_ms']:.1f} → {b['ends']['mae_ms']:.1f} ms | {a['boundaries']['p90_ms']:.1f} → {b['boundaries']['p90_ms']:.1f} ms | {a['boundaries']['within_100ms_pct']:.1f}% → {b['boundaries']['within_100ms_pct']:.1f}% |")
lines += ['', 'Canary is not uniformly better: the old Czech aligner has lower start error in both recordings, and slightly more natural-Czech boundaries fall within 100 ms. Canary reduces mean error largely through better endings and fewer large deviations.', '',
          '## Runtime and memory', '',
          '| Sample | Audio | 300M alignment | Canary alignment | 300M peak VRAM | Canary peak VRAM |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for s in data:
    a,b=s['models']['wav2vec2']['runtime'],s['models']['canary']['runtime']
    lines.append(f"| {names[s['sample']]} | {a['audio_seconds']:.1f} s | {a['alignment_seconds']:.2f} s | {b['alignment_seconds']:.2f} s | {a['peak_gpu_gib']:.2f} GiB | {b['peak_gpu_gib']:.2f} GiB |")
old=sum(s['models']['wav2vec2']['runtime']['alignment_seconds'] for s in data)
new=sum(s['models']['canary']['runtime']['alignment_seconds'] for s in data)
lines += ['', f'Total warm alignment time: **{old:.2f} s → {new:.2f} s**, a {old/new:.2f}× speedup. '
          'Median of three consecutive runs after warmup; repeated word timings were identical. Same GPU, batch size one, FP32, two CPU threads. '
          'Model download/loading and audio decoding are excluded. Initial model loading took roughly 19 s for Canary and 3–4 s for each 300M model; keep the model resident for production. '
          'Peak memory is PyTorch allocated GPU memory, not the entire process/context reservation. Both GPUs were available; only GPU 0 was used for comparable measurements.', '',
          '## Pairing, exclusions and uncertainty', '',
          '| Sample | Supplied words | Exact text matches | Scored pairs | Exclusions | Paired MAE change, 95% segment bootstrap |',
          '| --- | ---: | ---: | ---: | --- | ---: |']
for s in data:
    diff=s['paired_difference'];lo,hi=diff['segment_bootstrap_95pct_ms']
    lines.append(f"| {names[s['sample']]} | {s['supplied_words']} | {s['matched_words']} | {s['paired_words']} | {', '.join(f'{k}: {v}' for k,v in s['exclusions'].items())} | {diff['canary_minus_wav2vec2_mae_ms']:.1f} ms [{lo:.1f}, {hi:.1f}] |")
lines += ['',
    '- Negative changes favor Canary. Bootstrap resamples original VibeVoice segments, preserving within-segment correlation (2,000 resamples, fixed seed). These intervals are conditional on these recordings and Soniox; they do not establish population-wide accuracy.',
    '- Soniox subword text is concatenated losslessly before word splitting. Word timing uses the first/last token intersecting alphanumeric characters; punctuation-only tokens are excluded.',
    '- Matching is monotonic and exact after case/punctuation normalization. Diacritics remain significant. No number equivalence is used. Unmatched ASR words, invalid reference durations, and unsupported-character wildcard alignments are excluded and counted. Low CTC-score words are **not** discarded.',
    '- Both aligners receive the same existing VibeVoice words, speaker IDs and audio crops with 200 ms padding. Soniox word boundaries are never provided to either aligner. Two standalone music events in natural Czech are excluded from both alignment inputs. Natural recordings use saved vanilla transcripts; markers use the saved v4-final transcripts.',
    '- All supplied words and original segment/speaker fields are retained. All supplied speech words received timestamps in both systems, but numeric wildcard timings in Wav2Vec2 are not treated as verified.',
    '- Some large shared discrepancies involve repetitions, hesitations or Soniox words spanning a pause. They remain in the scores; no error-based trimming or timing-offset calibration was applied.',
    '- Soniox timing is a practical reference, not independent human boundary annotation. Canary emits an 80 ms frame grid; the older Wav2Vec2 models have a finer frame grid. Small millisecond differences should not be overinterpreted.', '',
    '## Reproduction and artifacts', '',
    '- `benchmark_soniox.py prepare` reconstructs reference words and fixed inputs. `benchmark_wav2vec2.py` and `align_canary.py` run the corresponding aligner on mlserver. `benchmark_soniox.py score` performs paired scoring. `write_report.py` regenerates this report.',
    '- `manifest.json` records audio/transcript/raw-Soniox SHA-256 hashes, source recordings and exact crop offsets. `protocol.json` records scoring rules. Per-sample `*.comparison.json` contains every scored word and boundary error.',
    '- `*.canary.json` preserves original NFA token paths and punctuation-inclusive word timings. `*.wav2vec2.json` preserves the upstream WhisperX output. All result files are saved on SSD and copied to mlserver.',
    '- Interactive review: http://localhost:8781/ — sample switching, cumulative error curves, sortable word errors and click-to-seek audio.',
    '- Canary: original `nvidia/canary-1b-v2` auxiliary CTC weights, revision `d455706339a6b32e1aa40f82c713a482a0c938e2`. Download archive SHA-256 `ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094`. Only auxiliary weights/tokenizer were loaded; main Canary decoder was not loaded or used.',
    '- NeMo 2.5.3, PyTorch 2.8.0+cu128. `get_utt_obj`, `viterbi_decoding` and `add_t_start_end_to_utt_obj` are official NeMo implementations. Strict checkpoint loading passed.',
    '- WhisperX commit `5d3678b8535698a2826a5db9eb3e4105aef85dd9`. Czech `comodoro/wav2vec2-xls-r-300m-cs-250`; Slovak `comodoro/wav2vec2-xls-r-300m-sk-cv8`. Exact model revisions remain in `model-cs.json` / `model-sk.json`.',
    '- Isolated alignment environments: `/data/work/asr-word-alignment/venv` and `/data/work/asr-word-alignment/nemo-venv`. Existing training environment versions verified unchanged.', '',
    '## Primary implementation references', '',
    '- [NVIDIA Canary-1B-v2 model card](https://huggingface.co/nvidia/canary-1b-v2)',
    '- [Canary v2 paper, timestamp alignment](https://arxiv.org/html/2509.14128v1#S5)',
    '- [NeMo 2.5.3 alignment implementation](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.5.3/nemo/collections/asr/parts/utils/aligner_utils.py)',
    '- [WhisperX alignment implementation](https://github.com/m-bain/whisperX/blob/5d3678b8535698a2826a5db9eb3e4105aef85dd9/whisperx/alignment.py)', '',
    'Synthetic generation and fresh training remain on the existing user-requested hold pending this review.', '',
]
(P/'REPORT.md').write_text('\n'.join(lines))
print(f'{total_words} words; {total_seconds/60:.2f} minutes; {old:.2f} s vs {new:.2f} s')
