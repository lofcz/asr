"""Describe segmentation and error components without changing the frozen scores."""
import json
import pathlib
import argparse

HERE = pathlib.Path(__file__).resolve().parent
VARIANTS = ('vanilla', 'finetuned')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='summary.json')
    args = parser.parse_args()
    summary = json.loads((HERE / args.summary).read_text())
    assert summary['status'] == 'complete', 'Wait for every paired result'
    groups = {}
    samples = []
    for row in summary['rows']:
        sample = {'sample_id': row['sample_id'], 'show': row['show'],
                  'language': row['language']}
        for variant in VARIANTS:
            result = row[variant]
            assert result['complete']
            m = result['metrics']
            sample[variant] = {
                'strict_substitutions': m['substitutions'],
                'strict_deletions': m['deletions'],
                'strict_insertions': m['insertions'],
                'speech_segments': result['speech_segment_count'],
                'acoustic_events': result['acoustic_event_count'],
                'predicted_speakers': m['predicted_speakers'],
                'reference_speakers': m['reference_speakers'],
                'generated_tokens': result['tokens'],
                'decode_seconds': result['wall_seconds'],
            }
        samples.append(sample)
    for label, rows in [('all', summary['rows'])] + [
            (lang, [r for r in summary['rows'] if r['language'] == lang])
            for lang in ('cs', 'sk')]:
        group = {}
        for variant in VARIANTS:
            ms = [r[variant]['metrics'] for r in rows]
            words = sum(m['reference_words'] for m in ms)
            speaker_time = sum(float(m['der_details']['scored_speaker_time']) for m in ms)
            group[variant] = {
                'recordings': len(rows), 'reference_words': words,
                'reference_speaker_seconds': speaker_time,
                'strict_word_error_components': {
                    k: {'errors': sum(m[k] for m in ms),
                        'percentage_points': 100 * sum(m[k] for m in ms) / words}
                    for k in ('substitutions', 'deletions', 'insertions')},
                'der_components': {
                    k: {'seconds': sum(float(m['der_details'][k]) for m in ms),
                        'percentage_points': 100 * sum(float(m['der_details'][k]) for m in ms) / speaker_time}
                    for k in ('missed_speaker_time', 'falarm_speaker_time', 'speaker_error_time')},
                'speech_segments': sum(r[variant]['speech_segment_count'] for r in rows),
                'generated_tokens': sum(r[variant]['tokens'] for r in rows),
                'decode_seconds': sum(r[variant]['wall_seconds'] for r in rows),
            }
        groups[label] = group
    result = {'checkpoint': summary['checkpoint'], 'panel_sha256': summary['panel_sha256'],
              'note': 'Descriptive decomposition of the same frozen results. No extra normalization or inference.',
              'groups': groups, 'samples': samples}
    (HERE / 'diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Error and segmentation diagnostics', '',
             'The same ten paired recordings and Soniox references as the accompanying report. '
             'DER components are percentage points of scored reference speaker time. '
             'This is a descriptive decomposition, not a second evaluation.', '',
             '| Panel | Metric | Vanilla | 250h checkpoint |',
             '|---|---|---:|---:|']
    for label, group in groups.items():
        a, b = (group[v] for v in VARIANTS)
        for key, name in [('missed_speaker_time', 'Missed speech'),
                          ('falarm_speaker_time', 'False-alarm speech'),
                          ('speaker_error_time', 'Speaker confusion')]:
            x, y = a['der_components'][key], b['der_components'][key]
            lines.append(f'| {label} | {name} | {x["percentage_points"]:.2f} pp ({x["seconds"]:.2f}s) | {y["percentage_points"]:.2f} pp ({y["seconds"]:.2f}s) |')
        for key, name in [('speech_segments', 'Speech segments'), ('generated_tokens', 'Generated tokens')]:
            lines.append(f'| {label} | {name} | {a[key]} | {b[key]} |')
        lines.append(f'| {label} | Sum of decoding minutes | {a["decode_seconds"]/60:.2f} | {b["decode_seconds"]/60:.2f} |')
    lines += ['', 'Decoding durations exclude model loading and feature preparation. '
              'The variants ran concurrently on separate A5000 GPUs. '
              'Additional segments can increase JSON tokens and generation time. '
              'These timing measurements do not predict rented-GPU performance.', '',
              '| Sample | Speech segments vanilla → 250h | Predicted speakers vanilla → 250h | Reference speakers |',
              '|---|---:|---:|---:|']
    for sample in samples:
        a, b = (sample[v] for v in VARIANTS)
        lines.append(f'| {sample["sample_id"]} | {a["speech_segments"]} → {b["speech_segments"]} | {a["predicted_speakers"]} → {b["predicted_speakers"]} | {a["reference_speakers"]} |')
    (HERE / 'DIAGNOSTICS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
