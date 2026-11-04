"""Render the saved Czech subtotal after the user cancelled the remaining work."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
s = json.loads((HERE / 'stopped-summary.json').read_text())
a, b = (s['aggregate'][v] for v in ('vanilla', 'finetuned'))
lines = ['# Evaluation stopped at user request', '',
         'Training remains paused. This is the six-recording Czech subtotal, approximately '
         '60 minutes, using Soniox machine references. The planned ten-recording evaluation '
         'was cancelled. No conclusion about Slovak follows from this subtotal.', '',
         '| Metric | Vanilla | 250h checkpoint |', '|---|---:|---:|']
for key, label in [('number_normalized_wer', 'Number-normalized WER'),
                   ('wer', 'Strict WER'), ('der_250ms', 'DER, 250ms collar')]:
    lines.append(f'| {label} | {100*a[key]:.2f}% | {100*b[key]:.2f}% |')
lines += ['', f'Relative word-error reduction: {100*s["relative_word_error_reduction"]:.2f}%.', '',
          'The checkpoint produced one malformed timestamp in cs-04: `0.0.0`. '
          'Its original JSON remains invalid. Scores include a separately saved copy with '
          'only that timestamp repaired to `0.0`. Raw validity: vanilla 6/6; checkpoint 5/6.', '']
for metric in ('number_normalized_wer', 'der_250ms'):
    d = s['paired'][metric]
    lo, hi = d['paired_bootstrap_95_interval']
    lines.append(f'- {metric}: checkpoint minus vanilla {100*d["delta"]:+.2f} pp; '
                 f'paired recording bootstrap 95% interval [{100*lo:+.2f}, {100*hi:+.2f}] pp; '
                 f'improved/worse/tied: {d["improved_samples"]}/{d["worse_samples"]}/{d["tied_samples"]}.')
lines += ['', 'These intervals describe a small, selected development panel and do not account '
          'for Soniox label errors. The planned evaluation was stopped after inspecting interim '
          'results, which further limits confirmatory statistical interpretation.', '',
          '| Sample | Show | Normalized WER vanilla → 250h | DER vanilla → 250h |',
          '|---|---|---:|---:|']
for row in s['rows']:
    a, b = (row[v]['metrics'] for v in ('vanilla', 'finetuned'))
    lines.append(f'| {row["sample_id"]} | {row["show"]} | '
                 f'{100*a["number_normalized_wer"]:.2f}% → {100*b["number_normalized_wer"]:.2f}% | '
                 f'{100*a["der_250ms"]:.2f}% → {100*b["der_250ms"]:.2f}% |')
(HERE / 'STOPPED_REPORT.md').write_text('\n'.join(lines) + '\n')
