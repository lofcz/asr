"""Complete final reporting without hiding an out-of-bounds model prediction.

Reuse every saved generated output. Keep all words for WER; DER uses the same
original clip UEM for both models. Strict timestamp validity is a separate,
explicit result, and the original evaluator failure remains unchanged on disk.
"""
import collections
import json
import math
from pathlib import Path
import random
import time
from contracts import atomic_json, sha

HERE = Path(__file__).resolve().parent


def output_issues(hypothesis, duration):
    if not isinstance(hypothesis, list) or not hypothesis:
        raise ValueError('Empty or invalid JSON hypothesis')
    issues = []
    for i, segment in enumerate(hypothesis):
        if not isinstance(segment, dict) or not {'Start', 'End', 'Content'} <= segment.keys():
            raise ValueError('Invalid schema')
        a, b = segment['Start'], segment['End']
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b)) or not 0 <= a <= b:
            raise ValueError('Invalid segment timestamps')
        if not isinstance(segment['Content'], str) or not segment['Content'].strip():
            raise ValueError('Invalid content')
        if b > duration + .02:
            issues.append({'segment_index': i, 'start': a, 'end': b,
                           'clip_duration': duration, 'overrun_seconds': b-duration,
                           'kind': 'prediction_exceeds_clip'})
    return issues


def aggregate(rows):
    ms = [r['metrics'] for r in rows]
    words = sum(m['number_normalized_reference_words'] for m in ms)
    errors = sum(m['number_normalized_errors'] for m in ms)
    speaker_time = sum(float(m['der_details']['scored_speaker_time']) for m in ms)
    der_errors = sum(float(m['der_details'][k]) for m in ms
                     for k in ('missed_speaker_time', 'falarm_speaker_time', 'speaker_error_time'))
    return {'number_normalized_wer': errors/words, 'errors': errors, 'reference_words': words,
            'der_250ms': der_errors/speaker_time, 'diarization_error_seconds': der_errors,
            'speaker_seconds': speaker_time, 'samples': len(rows),
            'raw_json_valid': len(rows), 'strict_output_valid': sum(not r['output_issues'] for r in rows),
            'out_of_bounds_clips': sum(bool(r['output_issues']) for r in rows)}


def compare_all(base, current, draws=20000):
    if {r['window_id'] for r in base} != {r['window_id'] for r in current}:
        raise ValueError('Panels differ')
    groups = collections.defaultdict(lambda: [[], []])
    for side, rows in enumerate((base, current)):
        for row in rows:
            groups[row['record_id']][side].append(row)
    pairs = [(aggregate(a), aggregate(b)) for a, b in groups.values()]
    rng = random.Random(20261023)
    deltas = {'number_normalized_wer': [], 'der_250ms': []}
    for _ in range(draws):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        for key, num, den in [('number_normalized_wer', 'errors', 'reference_words'),
                              ('der_250ms', 'diarization_error_seconds', 'speaker_seconds')]:
            values = [sum(p[i][num] for p in chosen)/sum(p[i][den] for p in chosen) for i in (0, 1)]
            deltas[key].append(values[1]-values[0])
    a, b = aggregate(base), aggregate(current)
    return {'vanilla': a, 'finetuned': b, 'recordings': len(groups),
            'relative_word_error_reduction': 1-b['number_normalized_wer']/a['number_normalized_wer'],
            'paired_recording_bootstrap_95': {k: [sorted(v)[int(.025*draws)], sorted(v)[int(.975*draws)]] for k, v in deltas.items()},
            'strict_acceptance_passed': b['strict_output_valid'] == b['samples'],
            'interpretation': f'{len(groups)} development interviews, not untouched confirmation. All outputs scored; strict output defects remain explicit. Repeated development selection limits confirmatory interpretation.'}


def score_saved(name, plan, adapter_hash):
    from scoring import metrics, validate_hypothesis
    rows = []
    for row in plan['validation']:
        source = HERE / 'evaluations' / name / row['window_id']
        original = json.loads((source / 'result.json').read_text())
        if original['termination'] != 'eos' or original['adapter_sha256'] != adapter_hash or original['input_sha256'] != row['input_sha256']:
            raise ValueError('Saved output identity or termination mismatch')
        reference = json.loads((source / 'reference.json').read_text())
        hypothesis = json.loads((source / 'output.txt').read_text())
        if reference != row['target'] or sha(source/'reference.json') != original['reference_sha256']:
            raise ValueError('Reference changed')
        issues = output_issues(hypothesis, row['duration'])
        # Check schema/speakers/finite times without accepting a late prediction
        # as valid. No timestamp or text is changed; late spans are reported.
        validate_hypothesis(hypothesis, max(row['duration'], max(s['End'] for s in hypothesis)))
        dest = HERE/'evaluations/final-audit'/name/row['window_id']
        dest.mkdir(parents=True, exist_ok=True)
        measured = json.loads(json.dumps(metrics(reference, hypothesis, dest, row['duration'], row['language']), default=str))
        if original.get('complete'):
            for key in ['number_normalized_errors', 'number_normalized_reference_words', 'number_normalized_wer', 'der_250ms']:
                if measured[key] != original['metrics'][key]:
                    raise ValueError('Existing metric changed on identical saved output: '+key)
        result = {'window_id': row['window_id'], 'record_id': row['record_id'], 'language': row['language'],
                  'metrics': measured, 'output_issues': issues, 'strict_output_valid': not issues,
                  'original_evaluation_complete': original['complete'], 'original_error': original.get('error'),
                  'raw_output_sha256': sha(source/'output.txt'), 'original_result_sha256': sha(source/'result.json'),
                  'raw_output_unchanged': True, 'all_predicted_words_scored': True,
                  'der_uem_seconds': row['duration'], 'adapter_sha256': adapter_hash}
        atomic_json(dest/'result.json', result)
        rows.append(result)
    return rows


def main():
    run = HERE/'runs/r64-full'
    done = json.loads((run/'completed.json').read_text())
    plan = json.loads((HERE/'plan.json').read_text())
    checkpoint = json.loads((run/'latest.json').read_text())
    adapter_hash = sha(run/checkpoint['checkpoint']/'adapter_model.safetensors')
    baseline = score_saved('vanilla', plan, 'vanilla')
    current = score_saved('r64-full-final', plan, adapter_hash)
    report = compare_all(baseline, current)
    stamp = HERE/'markers/r64-full-final/completed-v3.json'
    markers = json.loads(stamp.read_text())
    if markers['adapter_sha256'] != adapter_hash:
        raise ValueError('Final marker adapter differs')
    small = {r['window_id'] for r in plan['validation_small']}
    report.update(markers=markers, step=done['steps'], audio_hours_seen=done['training_audio_hours'],
                  status='completed_with_output_defect' if not report['strict_acceptance_passed'] else 'completed',
                  report_complete=True, completed_at=time.time(),
                  scoring_policy='All raw words retained for WER. DER uses the original clip UEM for both models. No raw text, timestamps, outputs or original failure results modified.',
                  output_issues=[{'window_id': r['window_id'], 'language': r['language'], 'issues': r['output_issues']} for r in current if r['output_issues']],
                  trajectory_panel=compare_all([r for r in baseline if r['window_id'] in small], [r for r in current if r['window_id'] in small]),
                  per_recording=[{'window_id': b['window_id'], 'language': b['language'], 'vanilla': b['metrics'], 'finetuned': c['metrics'], 'output_issues': c['output_issues']} for b,c in zip(baseline,current)],
                  provenance={'builder_sha256':sha(__file__),'plan_sha256':sha(HERE/'plan.json'),'adapter_sha256':adapter_hash})
    atomic_json(HERE/'r64-full-result.json', report)
    long_path=HERE/'long-evaluation-result.json'
    if long_path.exists():
        report['long_sessions']=json.loads(long_path.read_text())
        atomic_json(HERE/'r64-full-result.json',report)
    event = {'event':'decoded_validation', 'step':done['steps'], 'time':time.time(),
             **{'final/'+k:v for k,v in report['finetuned'].items() if isinstance(v,(int,float))},
             **{f'marker/{lang}/{k}':v for lang,m in markers['languages'].items() for k,v in m.items() if isinstance(v,(int,float))}}
    import os
    with (run/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
    terminal={'full_run_complete':True,'evaluation_report_complete':True,
              'strict_acceptance_passed':report['strict_acceptance_passed'],**report['finetuned']}
    atomic_json(run/'monitor-completed.json',terminal)
    atomic_json(HERE/'experiment-status.json',{'stage':report['status'],'running':False,'updated_at':time.time(),
                'reason':'Final metrics include all eight clips; timestamp-overrun outputs recorded' if report['output_issues'] else 'Training and evaluation completed'})
    atomic_json(HERE/'health-status.json',{'stage':report['status'],'running':False,'checked_at':time.time(),
                'training_complete':True,'evaluation_complete':True,'output_defect':bool(report['output_issues'])})
    print(json.dumps({k:v for k,v in report.items() if k not in ('markers','per_recording')},indent=2))


if __name__=='__main__':
    main()
