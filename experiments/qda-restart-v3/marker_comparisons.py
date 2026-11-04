"""Current r64 marker comparisons using only completed, provenance-matched runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / 'experiments/qda-full-r64-v4'
GOLD = ROOT / 'experiments/production-reference-audit-v2'
sys.path.insert(0, str(ROOT / 'experiments/periodic-eval'))
spec = importlib.util.spec_from_file_location('saved_comparison_alignment', ROOT / 'experiments/qda-496h-v1/comparisons.py')
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)
words, align = alignment.words, alignment.align
_cache = {}
_lock = threading.Lock()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    paths = [FULL / 'markers/vanilla/completed-v3.json', *sorted((FULL / 'markers').glob('r64-full-*/completed-v3.json'))]
    paths += sorted((ROOT/'experiments/qda-r64-continuation-v6/markers').glob('r64-full-*/completed-v3.json'))
    paths += sorted((ROOT/'experiments/qda-full-continuous-v8/markers').glob('r64-full-*/completed-v3.json'))
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in [*paths,ROOT/'experiments/active-training.json'] if p.exists())
    with _lock:
        if _cache.get('signature') == signature:
            return _cache['value']
        value = _build(paths)
        _cache.update(signature=signature, value=value)
        return value


def _build(paths):
    samples, metadata = {}, []
    baseline = json.loads(paths[0].read_text())
    for path in paths:
        if not path.exists():
            continue
        stamp = json.loads(path.read_text())
        if stamp['provenance'] != baseline['provenance'] or stamp['contract'] != baseline['contract']:
            raise ValueError('Marker input or scoring provenance differs from vanilla')
        vanilla = path.parent.name == 'vanilla'
        continuation=path.parents[2].name=='qda-r64-continuation-v6'
        fresh=path.parents[2].name=='qda-full-continuous-v8'
        sid = 'vanilla' if vanilla else ('v8-' if fresh else 'v6-' if continuation else '')+path.parent.name
        variant = 'vanilla' if vanilla else 'finetuned'
        metrics = stamp['languages']
        n = sum(m['number_normalized_reference_words'] for m in metrics.values())
        errors = sum(m['number_normalized_errors'] for m in metrics.values())
        der_seconds = sum(float(m['der_details'][k]) for m in metrics.values()
                          for k in ('missed_speaker_time', 'falarm_speaker_time', 'speaker_error_time'))
        speaker_seconds = sum(float(m['der_details']['scored_speaker_time']) for m in metrics.values())
        hours = 0 if vanilla else stamp.get('cumulative_audio_hours_seen',stamp['audio_hours_seen']+(188.16531321180554 if continuation else 0))
        metadata.append({'id': sid, 'variant': variant, 'rank': None if vanilla else 64,
                         'experiment_id': 'vanilla' if vanilla else 'v8' if fresh else 'v6' if continuation else 'v4',
                         'experiment_label': 'Vanilla' if vanilla else 'v8 · real + synthetic' if fresh else 'v6 · continuation' if continuation else 'v4 · earlier run',
                         'checkpoint_step': 0 if vanilla else stamp['step'], 'training_audio_hours': hours,
                         'milestone': int(hours // 50) * 50, 'final': sid.endswith('-final'),
                         'metrics': metrics, 'aggregate': {'number_normalized_wer': errors/n,
                                                          'errors': errors, 'reference_words': n,
                                                          'der_250ms': der_seconds/speaker_seconds}})
        for lang in ('cs', 'sk'):
            folder = path.parent / variant / lang
            reference_file = folder / 'reference.json'
            if sha(reference_file) != stamp['provenance'][lang]['reference_sha256']:
                raise ValueError('Marker reference file changed')
            result = json.loads((folder / 'result.json').read_text())
            if result['termination'] != 'eos' or result['metrics'] != metrics[lang]:
                raise ValueError('Marker metrics do not match completed evaluation')
            if lang not in samples:
                ref = json.loads(reference_file.read_text())
                groups, text = [], ''
                for segment in ref:
                    if not groups or groups[-1]['speaker'] != segment['Speaker']:
                        groups.append({'id': len(groups), 'start': segment['Start'], 'end': segment['End'],
                                       'speaker': segment['Speaker'], 'offset': len(text), 'text': '', 'segments': 0})
                    groups[-1]['text'] += segment['Content']
                    groups[-1]['end'] = segment['End']
                    groups[-1]['segments'] += 1
                    text += segment['Content']
                ref_words = words(text)
                for i, group in enumerate(groups):
                    end = groups[i+1]['offset'] if i+1 < len(groups) else len(text)
                    group['lo'] = sum(w['offset'] < group['offset'] for w in ref_words)
                    group['hi'] = sum(w['offset'] < end for w in ref_words)
                samples[lang] = {'text': text, 'turns': groups, 'models': {}, 'audio': '/marker-audio?lang=' + lang}
            hyp = json.loads((folder / 'hypothesis.json').read_text())
            speech = [s for s in hyp if not re.fullmatch(r'\s*\[[^\[\]]+\]\s*', s['Content'])]
            aligned, errors = align(samples[lang]['text'], ' '.join(s['Content'] for s in speech), lang)
            turns = []
            for group in samples[lang]['turns']:
                turns.append({'tokens': [w for w in aligned if group['lo'] <= w['anchor'] < group['hi']],
                              'errors': [w for w in errors if group['lo'] <= w['anchor'] < group['hi']],
                              'segments': [s for s in speech if group['start'] <= (s['Start'] + s['End'])/2 < group['end']]})
            gaps = [b['Start']-a['End'] for a, b in zip(speech, speech[1:]) if a.get('Speaker') == b.get('Speaker')]
            samples[lang]['models'][sid] = {'turns': turns, 'errors': errors, 'segment_count': len(speech),
                                           'same_speaker_boundaries': len(gaps), 'gap_seconds': sum(max(0, x) for x in gaps)}
    metadata.sort(key=lambda x: x['training_audio_hours'])
    contenders = [m for m in metadata if m['variant'] == 'finetuned']
    active=json.loads((ROOT/'experiments/active-training.json').read_text()).get('experiment','')
    active_id='v8' if active=='qda-full-continuous-v8' else 'v6' if active=='qda-r64-continuation-v6' else 'v4'
    best_by_experiment={}
    for item in contenders:
        group=best_by_experiment.setdefault(item['experiment_id'],[]);group.append(item)
    best_by_experiment={key:min(items,key=lambda m:(m['aggregate']['number_normalized_wer'],m['training_audio_hours']))['id'] for key,items in best_by_experiment.items()}
    best_overall=min(contenders,key=lambda m:(m['aggregate']['number_normalized_wer'],m['training_audio_hours']))
    best=next((m for m in contenders if m['id']==best_by_experiment.get(active_id)),best_overall)
    for m in metadata:
        m['best'] = m['id'] == best_by_experiment.get(m['experiment_id'])
    for lang, sample in samples.items():
        sample['pairs'] = {}
        for left_id, left in sample['models'].items():
            for right_id, right in sample['models'].items():
                pair = []
                for a, b in zip(left['turns'], right['turns']):
                    text = lambda t: ''.join(w['text'] for w in t['tokens'] if w['kind'] != 'missing')
                    right_tokens, _ = align(text(a), text(b), lang)
                    left_tokens, _ = align(text(b), text(a), lang)
                    pair.append({'left': [w for w in left_tokens if w['kind'] != 'missing'],
                                 'right': [w for w in right_tokens if w['kind'] != 'missing']})
                sample['pairs'][left_id + '|' + right_id] = pair
    return {'snapshots': metadata, 'samples': samples, 'best_checkpoint': best['id'],
            'active_experiment':active_id,'best_by_experiment':best_by_experiment,'best_overall_checkpoint':best_overall['id'],
            'selection_rule': 'Lowest combined Czech + Slovak number-normalized WER within the selected run; earliest checkpoint breaks a tie. Marker selection is exploratory, not a held-out generalization claim.',
            'note': 'v8 is the current fresh real + synthetic run. v4 is the earlier real-only run; v6 continued v4. Runs are labelled separately and never connected as one learning trajectory. Original references and model text are unchanged.',
            'alignment_note': 'Highlights show word differences against the approved reference, with number-equivalent wording in blue. Highlight groups are not WER edit counts; the cards use the saved evaluator metrics. Speaker IDs are local to each output.'}
