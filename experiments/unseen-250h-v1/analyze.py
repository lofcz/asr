"""Keep raw-output validity separate from transcript scores with logged recovery."""
import json
import pathlib

import recover
from contracts import atomic_json, sha
from report import aggregate, paired

HERE = pathlib.Path(__file__).resolve().parent
VARIANTS = ('vanilla', 'finetuned')


def main():
    recover.main()
    panel = json.loads((HERE / 'panel.json').read_text())
    rows, failures, recovery_records = [], [], []
    valid = {v: 0 for v in VARIANTS}
    for sample in panel['samples']:
        row = {k: sample[k] for k in ('sample_id', 'language', 'show', 'title', 'record_id')}
        row['duration_seconds'] = sample['audio_duration']
        for variant in VARIANTS:
            original = HERE / variant / sample['sample_id']
            p = original / 'result.json'
            assert p.exists(), f'Generation pending: {variant}/{sample["sample_id"]}'
            raw = json.loads(p.read_text())
            if raw.get('complete'):
                valid[variant] += 1
                row[variant] = {**raw, 'raw_output_valid': True, 'scoring_status': 'raw_valid'}
                continue
            failures.append({'sample': sample['sample_id'], 'variant': variant, 'error': raw.get('error')})
            rp = HERE / 'recovered' / variant / sample['sample_id'] / 'result.json'
            if not rp.exists():
                row[variant] = raw
                continue
            fixed = json.loads(rp.read_text())
            assert fixed['original_output_sha256'] == sha(original / 'output.txt')
            assert fixed['recovery_code_sha256'] == sha(HERE / 'recover.py')
            for key in ('input_sha256', 'feature_sha256', 'reference_sha256', 'contract_sha256'):
                assert raw[key] == fixed[key]
            row[variant] = fixed
            recovery_records.append({'sample': sample['sample_id'], 'variant': variant,
                                     'rule': fixed['recovery_rule'], 'repairs': fixed['repairs']})
        for key in ('input_sha256', 'feature_sha256', 'reference_sha256'):
            assert row['vanilla'][key] == row['finetuned'][key]
        if all(row[v].get('complete') for v in VARIANTS):
            for key in ('reference_words', 'number_normalized_reference_words'):
                assert row['vanilla']['metrics'][key] == row['finetuned']['metrics'][key]
            a, b = (float(row[v]['metrics']['der_details']['scored_speaker_time']) for v in VARIANTS)
            assert a == b
        rows.append(row)
    scoreable = all(row[v].get('complete') for row in rows for v in VARIANTS)
    result = {
        'status': 'complete' if scoreable else 'unscoreable_outputs',
        'checkpoint': panel['checkpoint'], 'checkpoint_audio_hours': panel['checkpoint_audio_hours'],
        'panel_sha256': sha(HERE / 'panel.json'), 'reference_quality': panel['reference_quality'],
        'audio_minutes': panel['audio_minutes'], 'recordings': len(rows),
        'metric_scope': 'All ten recordings, including explicitly logged syntax recovery when needed',
        'raw_valid_outputs': valid, 'raw_failures': failures,
        'recoveries': recovery_records, 'rows': rows,
    }
    if scoreable:
        result['aggregate'] = {v: aggregate(rows, v) for v in VARIANTS}
        result['languages'] = {lang: {v: aggregate([r for r in rows if r['language'] == lang], v)
                                     for v in VARIANTS} for lang in ('cs', 'sk')}
        result['paired'] = {metric: paired(rows, metric) for metric in ('number_normalized_wer', 'der_250ms')}
        a, b = (result['aggregate'][v]['number_normalized_wer'] for v in VARIANTS)
        result['relative_word_error_reduction'] = (a - b) / a
    atomic_json(HERE / 'analysis.json', result)
    lines = ['# Ten unseen recordings: vanilla versus the 250-hour checkpoint', '',
             f'Checkpoint #515, {panel["checkpoint_audio_hours"]:.3f} training audio hours. '
             f'Ten separate recordings: six Czech, four Slovak, {panel["audio_minutes"]:.2f} audio minutes.', '',
             '**Training remains paused. References are Soniox machine labels, not human gold.**', '',
             f'Raw valid outputs: vanilla **{valid["vanilla"]}/10**, checkpoint **{valid["finetuned"]}/10**.']
    if recovery_records:
        lines += ['', 'The transcript scores below include separately saved syntax repairs. '
                  'Raw outputs and their failures remain preserved. No words or speaker identities were corrected.']
        for r in recovery_records:
            edits = ', '.join(f'`{x["before"]}` → `{x["after"]}` at character {x["offset"]}' for x in r['repairs'])
            lines.append(f'- {r["sample"]}, {r["variant"]}: timestamp {edits}.')
    if not scoreable:
        lines += ['', 'At least one output remains unscoreable. No whole-panel aggregate is reported.',
                  '```json', json.dumps(failures, ensure_ascii=False, indent=2), '```']
    else:
        def pct(value):
            return f'{100 * value:.2f}%'
        lines += ['', '| Panel | Normalized WER vanilla | Normalized WER 250h | DER vanilla | DER 250h |',
                  '|---|---:|---:|---:|---:|']
        for name, group in [('All', result['aggregate']), ('Czech', result['languages']['cs']),
                            ('Slovak', result['languages']['sk'])]:
            a, b = (group[v] for v in VARIANTS)
            lines.append(f'| {name} | {pct(a["number_normalized_wer"])} | {pct(b["number_normalized_wer"])} | '
                         f'{pct(a["der_250ms"])} | {pct(b["der_250ms"])} |')
        lines += ['', f'Relative word-error reduction: **{100 * result["relative_word_error_reduction"]:.1f}%**.', '',
                  'Intervals below resample entire recordings in 20,000 paired bootstrap draws. '
                  'Negative deltas favor the checkpoint. Intervals describe this small development panel; '
                  'they do not capture uncertainty in Soniox labels or establish accuracy on unseen shows.', '']
        for metric, label in [('number_normalized_wer', 'Normalized WER'), ('der_250ms', 'DER')]:
            d = result['paired'][metric]
            lo, hi = d['paired_bootstrap_95_interval']
            lines.append(f'- {label}: {100*d["delta"]:+.2f} percentage points; 95% interval '
                         f'[{100*lo:+.2f}, {100*hi:+.2f}]. Improved/worse/tied recordings: '
                         f'{d["improved_samples"]}/{d["worse_samples"]}/{d["tied_samples"]}.')
        lines += ['', '| Sample | Show | Normalized WER vanilla → 250h | DER vanilla → 250h | Raw checkpoint output |',
                  '|---|---|---:|---:|---|']
        for row in rows:
            a, b = (row[v]['metrics'] for v in VARIANTS)
            state = 'valid' if row['finetuned']['raw_output_valid'] else 'invalid; syntax recovered'
            lines.append(f'| {row["sample_id"]} | {row["show"]} | '
                         f'{pct(a["number_normalized_wer"])} → {pct(b["number_normalized_wer"])} | '
                         f'{pct(a["der_250ms"])} → {pct(b["der_250ms"])} | {state} |')
        a, b = (result['aggregate'][v] for v in VARIANTS)
        lines += ['', f'Strict WER, without number normalization: {pct(a["wer"])} → {pct(b["wer"])}.', '',
                  'Both variants used identical cached speech features, empty context, BF16 inference '
                  'and greedy decoding. Hash equality is checked before aggregation. WER is weighted '
                  'by reference words; DER is weighted by scored speaker time with a 250ms collar.', '',
                  'The panel was selected before generation and is disjoint from original training '
                  'recordings and the frozen continuation probe. Shows and speakers can overlap training. '
                  'These are approximately ten-minute excerpts, so they do not establish hour-long session quality. '
                  'The full source recordings are reserved from future training extensions.']
    (HERE / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
