"""Report only completed matched pairs while the fixed ten-recording panel runs."""
import json
import pathlib
import time

from report import aggregate
from contracts import atomic_json, sha

HERE = pathlib.Path(__file__).resolve().parent
VARIANTS = ('vanilla', 'finetuned')


def main():
    panel = json.loads((HERE / 'panel.json').read_text())
    rows, failures = [], []
    completed = {v: 0 for v in VARIANTS}
    progress = {}
    for variant in VARIANTS:
        p = HERE / variant / 'progress.json'
        progress[variant] = json.loads(p.read_text()) if p.exists() else None
    for sample in panel['samples']:
        row = {k: sample[k] for k in ('sample_id', 'show', 'language')}
        for variant in VARIANTS:
            p = HERE / variant / sample['sample_id'] / 'result.json'
            if p.exists():
                row[variant] = json.loads(p.read_text())
                if row[variant].get('complete'):
                    completed[variant] += 1
                else:
                    failures.append({'sample': sample['sample_id'], 'variant': variant,
                                     'error': row[variant].get('error')})
        if not all(row.get(v, {}).get('complete') for v in VARIANTS):
            continue
        for key in ('input_sha256', 'feature_sha256', 'reference_sha256'):
            assert row['vanilla'][key] == row['finetuned'][key]
        for key in ('reference_words', 'number_normalized_reference_words'):
            assert row['vanilla']['metrics'][key] == row['finetuned']['metrics'][key]
        a, b = (float(row[v]['metrics']['der_details']['scored_speaker_time']) for v in VARIANTS)
        assert a == b
        rows.append(row)
    result = {
        'updated_at': time.time(),
        'status': 'complete' if len(rows) == len(panel['samples']) else 'provisional',
        'panel_sha256': sha(HERE / 'panel.json'),
        'completed': completed, 'paired': len(rows), 'total': len(panel['samples']),
        'reference_quality': panel['reference_quality'],
        'aggregate': {v: aggregate(rows, v) for v in VARIANTS} if rows else {},
        'samples': [{**{k: r[k] for k in ('sample_id', 'show', 'language')},
                     **{v: {k: r[v]['metrics'][k] for k in ('number_normalized_wer', 'der_250ms')}
                        for v in VARIANTS}} for r in rows],
        'progress': progress, 'failures': failures,
    }
    atomic_json(HERE / 'progress-summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
