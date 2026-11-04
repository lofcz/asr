"""Training admission is a fresh recomputation, never a hand-written ready flag."""
import json
import pathlib
import contextlib
import fcntl
from contracts import sha, atomic_json
from targets import independent_verify, window_segments
from events import reviewed_events,with_events

HERE=pathlib.Path(__file__).resolve().parent


def check_data(data, reviewed=True):
    from collections import defaultdict
    from dataset import window_segments, merge_target_turns
    provenance=json.loads((data/'provenance.json').read_text())
    demand(sha(data/'screening-selection.json')==provenance['selection_sha256'],'Combined selection changed')
    demand(sha(data/'sources.json')==provenance['sources_sha256'],'Combined source identity changed')
    originals={}
    for source in provenance['source_plans']:
        path=pathlib.Path(source['path']);demand(sha(path)==source['sha256'],'Original immutable plan changed')
        plan=json.loads(path.read_text())
        for row in [r for pair in plan['probe_pairs'] for r in pair]+plan['validation']:
            originals[row['window_id']]=row
    selection=json.loads((data/'screening-selection.json').read_text())
    sources=json.loads((data/'sources.json').read_text());by_source={s['audio_sha256']:s for s in sources}
    heldout={s['audio_sha256'] for s in sources if s['split']!='train'}
    source_shows={s['show'] for s in sources if s['split']!='train'}
    labels={};coverage=defaultdict(list);seen=set()
    for row in selection['train']+selection['development']:
        demand(row['window_id'] not in seen,'Duplicate training/development window');seen.add(row['window_id'])
        path=data/row['label_path']
        if str(path) not in labels:
            demand(sha(path)==row['label_sha256'],'Label hash mismatch')
            labels[str(path)]=json.loads(path.read_text())
        demand(abs(row['source_end']-row['source_start']-row['duration'])<1e-6,'Wrong crop duration')
        if row['dataset_kind']=='real':
            old=originals[row['window_id']]
            demand(all(old[k]==row[k] for k in ('audio_sha256','label_sha256','source_start','source_end','split')),'Existing real source/crop was altered')
            expected=[t for t in old['target'] if not (isinstance(t.get('Content'),str) and not t['Content'].strip())]
            demand(expected==row['target'],'Existing real transcript wording changed')
        else:
            expected=merge_target_turns(window_segments(labels[str(path)]['segments'],row['source_start'],row['source_end'],row['audio_duration']))
            demand(expected==row['target'],'Synthetic crop differs from authored source')
        validate_reference(row['target'],row['duration'])
        if row['split']=='train':
            demand(row['audio_sha256'] not in heldout and row['show'] not in source_shows,'Held-out audio/show leaked into training')
            coverage[row['audio_sha256']].append(row)
    for key,source in by_source.items():
        if source['split']!='train':continue
        cursor=0.
        for row in sorted(coverage[key],key=lambda r:r['source_start']):
            demand(abs(row['source_start']-cursor)<1e-6,'Training source gap or duplicated audio')
            cursor=row['source_end']
        demand(abs(cursor-source['audio_duration'])<1e-6,'Training source was not fully covered')
    result={'status':'passed','provenance_sha256':sha(data/'provenance.json'),
            'selection_sha256':sha(data/'screening-selection.json'),'sources_sha256':sha(data/'sources.json'),
            'gates_sha256':sha(__file__),'dataset_code_sha256':sha(HERE/'dataset.py'),
            'train_hours':selection['train_hours'],'real_train_hours':selection['real_train_hours'],
            'synthetic_train_hours':selection['synthetic_train_hours'],'train_windows':len(selection['train']),
            'development_windows':len(selection['development']),'overlap_seconds':0,'uncovered_train_seconds':0,
            'speaker_count_filter':False,'acoustic_quality_filter':False,'vocabulary_admission_filter':False,
            'training_autostart':False}
    path=data/'admission.json'
    if path.exists():demand(json.loads(path.read_text())==result,'Immutable combined admission changed')
    else:atomic_json(path,result)
    return result


@contextlib.contextmanager
def lease_data(data,exclusive=False,wait=False):
    with (data/'.curation.lock').open('a') as handle:
        try:fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:raise RuntimeError('Dataset curation or training holds the data lease; no concurrent mutation is allowed') from None
        yield handle


def demand(condition, message):
    if not condition:raise RuntimeError(message)


def development_reference(data,row):
    """Soniox is sufficient; apply only explicit, source-bound optional reviews."""
    path=data/'reviews'/(row['window_id']+'.json')
    reference=row['target'];quality='Soniox pseudo-reference';fingerprint=None
    if path.exists():
        review=json.loads(path.read_text())
        demand(review.get('decision')!='reject','A reviewer rejected development data')
        if review.get('decision')=='approve':
            demand(review.get('audio_sha256')==row['audio_sha256'] and review.get('label_sha256')==row['label_sha256'],'Review is stale')
            demand(bool(review.get('reviewer')) and bool(review.get('reviewed_at')),'Review has no provenance')
            demand(all(review.get(k) is True for k in ('speech_verified','speakers_verified','events_verified')),'Incomplete optional review')
            reference=review['reference'];quality='Human-reviewed Soniox reference';fingerprint=sha(path)
    validate_reference(reference,row['duration'])
    return reference,quality,fingerprint



def validate_reference(reference,duration):
    demand(isinstance(reference,list) and bool(reference),'Empty reference')
    events={'[Unintelligible Speech]','[Music]','[Human Sounds]','[Environmental Sounds]','[Noise]','[Silence]'}
    import math
    for s in reference:
        demand(isinstance(s,dict) and {'Start','End','Content'}<=s.keys(),'Invalid reference schema')
        a,b=s['Start'],s['End']
        demand(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b)),'Invalid reference time')
        demand(0<=a<=b<=duration+.02,'Reference time outside clip')
        demand(isinstance(s['Content'],str) and bool(s['Content'].strip()),'Empty reference content')
        demand(s['Content'] in events or (isinstance(s.get('Speaker'),(int,str)) and not isinstance(s.get('Speaker'),bool)),'Speech has no speaker')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--machine-only',action='store_true');a=p.parse_args()
    print(json.dumps(check_data(a.data,reviewed=not a.machine_only),indent=2))
