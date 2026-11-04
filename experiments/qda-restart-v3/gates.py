"""Training admission is a fresh recomputation, never a hand-written ready flag."""
import json
import pathlib
import contextlib
import fcntl
from contracts import sha, atomic_json
from targets import independent_verify, window_segments
from events import reviewed_events,with_events

HERE=pathlib.Path(__file__).resolve().parent


@contextlib.contextmanager
def lease_data(data,exclusive=False,wait=False):
    with (data/'.curation.lock').open('a') as handle:
        try:fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:raise RuntimeError('Dataset curation or training holds the data lease; no concurrent mutation is allowed') from None
        yield


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


def check_data(data, reviewed=True):
    demand(not (data/'.building').exists(),'Dataset build is incomplete')
    policy=json.loads((HERE/'policy.json').read_text());struct=json.loads((data/'structural-audit.json').read_text())
    demand(struct['policy_sha256']==sha(HERE/'policy.json'),'Structural policy changed')
    demand(struct['target_code_sha256']==sha(HERE/'targets.py'),'Target code changed; rebuild and audit')
    demand(struct['sources_sha256']==sha(data/'sources.json'),'Source manifest changed')
    selection=json.loads((data/'screening-selection.json').read_text())
    acoustic_selection=json.loads((data/'acoustic-selection.json').read_text())
    demand(acoustic_selection['selector_sha256']==sha(HERE/'select_windows.py'),'Acoustic selection code changed')
    demand(selection['policy_sha256']==sha(HERE/'policy.json'),'Selection policy changed')
    demand(selection['candidates_sha256']==sha(data/'candidates.json'),'Candidate manifest changed')
    train=selection['train'];dev=selection['development'];allrows=train+dev
    demand(len({r['record_id'] for r in train})==len(train),'Repeated training recording')
    demand(not {r['show'] for r in train}&{r['show'] for r in dev},'Show leakage')
    demand(not {r['audio_sha256'] for r in train}&{r['audio_sha256'] for r in dev},'Audio leakage')
    demand(not {r['record_id'] for r in train}&set(struct['reserved_record_ids']),'Previously evaluated recording leaked')
    demand(not {r['show'] for r in train}&set(struct['heldout_shows']),'Held-out show leaked')
    seconds=sum(r['duration'] for r in train)
    demand(policy['training_audio_hours']*.95<=seconds/3600<=policy['training_audio_hours'],'Insufficient clean training budget; do not silently shrink or relax gates')
    for kind,fraction in policy['length_audio_fractions'].items():
        actual=sum(r['duration'] for r in train if r['duration_class']==kind)/seconds
        demand(abs(actual-fraction)<=.1,'Actual duration mix outside bounds: '+kind)
    sk=sum(r['duration'] for r in train if r['language']=='sk')/seconds
    demand(abs(sk-policy['slovak_audio_fraction'])<=.1,'Language mix outside bounds')
    for lang in ('cs','sk'):
        demand(any(r['language']==lang and r['duration']>=2700 for r in train),'Missing >=45-minute training session: '+lang)
    dev_seconds=sum(r['duration'] for r in dev)
    demand(dev_seconds>=policy['minimum_development_minutes']*60,'Insufficient held-out development audio')
    demand(len({r['record_id'] for r in dev})>=policy['minimum_development_recordings'],'Too few independent development recordings')
    demand({r['language'] for r in dev}=={'cs','sk'},'Both development languages are required')
    fingerprints={};reviewed_seconds=0.;reviewed_records=set()
    for row in allrows:
        rid=row['record_id'];wid=row['window_id']
        for path,expected in [(row['audio_file'],row['audio_sha256']),(row['raw_path'],row['raw_transcript_sha256']),
                              (data/row['label_path'],row['label_sha256']),(row['vad_path'],row['vad_sha256']),
                              (data/'acoustic'/(rid+'.json'),row['acoustic_sha256'])]:
            actual=sha(path);demand(actual==expected,'Artifact hash mismatch: '+str(path));fingerprints[str(path)]=actual
        label=json.loads((data/row['label_path']).read_text());raw=json.loads(pathlib.Path(row['raw_path']).read_text())
        independent_verify(raw['tokens'],label['segments'],row['audio_duration'])
        expected=window_segments(label['segments'],row['source_start'],row['source_end'])
        if row.get('acoustic_review_sha256'):
            events=reviewed_events(data,row,row['quality'])
            demand(events is not None and events[1]==row['acoustic_review_sha256'],'Acoustic review changed')
            expected=with_events(expected,events[0]);fingerprints[str(data/'acoustic-reviews'/(wid+'.json'))]=events[1]
        demand(expected==row['target'],'Window target changed')
        demand(not row['quality']['reasons'],'Quarantined window selected')
        review_path=data/'reviews'/(wid+'.json')
        if row['split']=='train' and review_path.exists():
            review=json.loads(review_path.read_text())
            demand(review.get('decision')!='reject','A reviewer rejected training data')
            demand(review.get('reference')==row['target'],'Reviewed training correction requires a versioned target rebuild; do not ignore it')
            fingerprints[str(review_path)]=sha(review_path)
        if policy.get('secondary_asr_required',False):
            secondary_path=data/'secondary'/(wid+'.json');secondary=json.loads(secondary_path.read_text())
            demand(secondary['status']=='passed','Secondary ASR rejected '+wid)
            demand(secondary['label_sha256']==row['label_sha256'] and secondary['audio_sha256']==row['audio_sha256'],'Secondary audit stale')
            demand(secondary['code_sha256']==sha(HERE/'secondary_audit.py') and secondary['policy_sha256']==sha(HERE/'policy.json'),'Secondary audit contract stale')
            fingerprints[str(secondary_path)]=sha(secondary_path)
        if row['split']=='development':
            reference,quality,fingerprint=development_reference(data,row)
            demand(not any(s['Content'].strip()=='[Unintelligible Speech]' for s in reference),
                   'Development clip has unresolved speech; choose an intelligible held-out crop for word-error measurement')
            if fingerprint:
                fingerprints[str(data/'reviews'/(wid+'.json'))]=fingerprint;reviewed_seconds+=row['duration'];reviewed_records.add(rid)
            if policy.get('requires_reviewed_development_before_training',False):demand(fingerprint is not None,'Development review required by policy')
    result={'status':'passed','selection_sha256':sha(data/'screening-selection.json'),'policy_sha256':sha(HERE/'policy.json'),
            'structural_audit_sha256':sha(data/'structural-audit.json'),'artifacts':fingerprints,
            'train_hours':seconds/3600,'development_minutes':dev_seconds/60,'development_records':len({r['record_id'] for r in dev}),
            'reference_quality':'Soniox pseudo-references; explicit human reviews are optional',
            'reviewed_development_minutes':reviewed_seconds/60,'reviewed_records':len(reviewed_records),
            'training_authorized_by_data_gate':reviewed}
    atomic_json(data/('admission.json' if reviewed else 'machine-admission.json'),result)
    return result


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
