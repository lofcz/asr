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
    """Prove identity, exact targets, disjointness and full remaining coverage."""
    from dataset import verify_partition, window_segments as crop
    demand(not (data/'.building').exists(),'Dataset build is incomplete')
    provenance=json.loads((data/'provenance.json').read_text())
    parent=HERE.parent/'qda-full-r64-v4'
    for path,key in [(data/'screening-selection.json','selection_sha256'),
                     (data/'sources.json','sources_sha256'),
                     (data/'pairing-inventory.jsonl','inventory_sha256'),
                     (parent/'plan.json','parent_plan_sha256'),
                     (parent/'runs/r64-full/completed.json','parent_completed_sha256')]:
        demand(sha(path)==provenance[key],'Data provenance changed: '+str(path))
    source=json.loads((data/'sources.json').read_text())
    selection=json.loads((data/'screening-selection.json').read_text())
    old=json.loads((parent/'plan.json').read_text())
    consumed=[r for pair in old['probe_pairs'] for r in pair]
    partition=verify_partition(source,consumed,selection['train'])
    demand(partition==provenance['partition'],'Remaining coverage changed')
    labels={}
    for row in selection['train']+selection['development']:
        key=row['label_path']
        if key not in labels:
            demand(sha(data/key)==row['label_sha256'],'Label identity changed: '+key)
            labels[key]=json.loads((data/key).read_text())
        expected=crop(labels[key]['segments'],row['source_start'],row['source_end'],row['audio_duration'])
        # Fixed development fixtures retain their explicitly approved reference.
        if row['split']=='train':demand(expected==row['target'],'Window target changed')
        else:
            oldrow=next(r for r in old['validation'] if r['window_id']==row['window_id'])
            demand(all(oldrow[k]==row[k] for k in ('audio_sha256','label_sha256','source_start','source_end','target')),'Held-out reference changed')
        validate_reference(row['target'],row['duration'])
    result={'status':'passed','provenance_sha256':sha(data/'provenance.json'),
        'selection_sha256':sha(data/'screening-selection.json'),'sources_sha256':sha(data/'sources.json'),
        'gates_sha256':sha(__file__),'dataset_code_sha256':sha(HERE/'dataset.py'),**partition,
        'reference_quality':'Original Soniox labels; fixed held-out references unchanged',
        'pairing_verified_in_recovery':provenance['inventory_sha256'],
        'audio_bytes_verified_again_before_each_decode':True}
    path=data/'admission.json'
    if path.exists():demand(json.loads(path.read_text())==result,'Immutable admission changed')
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
