"""Select nonoverlapping whole-word windows using full-timeline independent evidence."""
import argparse
import collections
import hashlib
import json
import pathlib
from contracts import atomic_json, sha
from targets import window_segments
from events import reviewed_events,with_events

HERE=pathlib.Path(__file__).resolve().parent


def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out


def subtract(left,right):
    result=[]
    for a,b in union(left):
        cursor=a
        for c,d in right:
            if d<=cursor:continue
            if c>=b:break
            if c>cursor:result.append([cursor,min(c,b)])
            cursor=max(cursor,d)
            if cursor>=b:break
        if cursor<b:result.append([cursor,b])
    return result


def intersect(intervals,a,b):return [[max(a,c),min(b,d)] for c,d in intervals if d>a and c<b]
def seconds(intervals):return sum(b-a for a,b in intervals)


def quality(source,label,vad,acoustic,a,b,policy):
    duration=b-a;reasons=[]
    if any(r['start'] < b and r['end'] > a for r in label.get('quarantined_regions', [])):
        reasons.append('ambiguous_intraword_gap')
    speech=union(intersect(vad['speech_intervals_seconds'],a,b))
    words=union([[max(a,w['start']-.1),min(b,w['end']+.1)] for w in label['words'] if w['end']>a and w['start']<b])
    # Sub-second VAD/word-edge disagreements are reported by the raw audit but do
    # not constitute missing utterances. This uses the original pipeline's 1s rule.
    missing=[x for x in subtract(speech,words) if x[1]-x[0]>=1.]
    gaps=[x for x in subtract([[a,b]],union(speech+words)) if x[1]-x[0]>=policy['maximum_unresolved_nonspeech_seconds']]
    blocks=[w for w in acoustic['windows'] if w['end']>a and w['start']<b]
    flagged=[w for w in blocks if max(w['scores']['music'],w['scores']['noise'])>=policy['music_or_noise_score_threshold']]
    acoustic_seconds=sum(min(b,w['end'])-max(a,w['start']) for w in flagged)
    clipped=sum(w['clipped_fraction']*(min(b,w['end'])-max(a,w['start'])) for w in blocks)/duration
    local_words=[w for w in label['words'] if w['end']>a and w['start']<b]
    indices=[i for w in local_words for i in w['token_indices']]
    low=sum(label['token_confidences'][i]<policy['low_confidence_threshold'] for i in indices)/max(1,len(indices))
    if seconds(speech)/duration<policy['minimum_vad_speech_fraction']:reasons.append('low_speech_fraction')
    if seconds(missing)/duration>policy['maximum_untranscribed_vad_fraction']:reasons.append('untranscribed_speech_fraction')
    if max((d-c for c,d in missing),default=0)>policy['maximum_untranscribed_vad_run_seconds']:reasons.append('untranscribed_speech_run')
    if gaps and policy.get('reject_unresolved_nonspeech',False):reasons.append('unresolved_nonspeech')
    if acoustic_seconds/duration>policy['maximum_flagged_acoustic_fraction']:reasons.append('music_or_noise')
    if clipped>policy['maximum_clipped_sample_fraction']:reasons.append('clipping')
    if source['target_language_fraction']<policy['minimum_target_language_fraction']:reasons.append('language_mismatch')
    if low>policy['maximum_low_confidence_token_fraction']:reasons.append('low_word_confidence')
    if not 2<=len({w['speaker'] for w in local_words})<=4:reasons.append('not_small_conversation')
    return {'reasons':reasons,'vad_speech_fraction':seconds(speech)/duration,'missing_speech_seconds':seconds(missing),
            'missing_speech_regions':missing,'unresolved_nonspeech':gaps,'flagged_acoustic_fraction':acoustic_seconds/duration,
            'clipped_fraction':clipped,'low_word_confidence_fraction':sum(w['confidence']<policy['low_confidence_threshold'] for w in local_words)/max(1,len(local_words)),
            'low_token_confidence_fraction':low}


def candidates(data):
    if (data/'.building').exists():raise RuntimeError('Structural rebuild in progress')
    policy=json.loads((HERE/'policy.json').read_text());sources=json.loads((data/'sources.json').read_text())
    rows=[];rejections=[];pending=[];acoustic_review_queue=[]
    for source in sources:
        if source['split']=='confirmation':continue  # keep the confirmatory set untouched
        rid=source['record_id'];ap=data/'acoustic'/(rid+'.json')
        if not ap.exists():pending.append(rid);continue
        acoustic=json.loads(ap.read_text());assert acoustic['audio_sha256']==source['audio_sha256']
        if acoustic.get('status')!='passed_decode':raise ValueError('Acoustic decode did not pass')
        vp=pathlib.Path(source['vad_path']);vad=json.loads(vp.read_text())
        assert vad['audio_sha256']==source['audio_sha256'] and vad['full_decode_verified']
        assert abs(vad['decoded_seconds']-source['audio_duration'])<.1
        label=json.loads((data/source['label_path']).read_text());assert sha(data/source['label_path'])==source['label_sha256']
        segments=label['segments'];cuts=[0.];extent=0.
        for x,y in zip(segments,segments[1:]):
            extent=max(extent,x['end'])
            if y['start']>=extent:cuts.append((extent+y['start'])/2)
        cuts.append(source['audio_duration'])
        # Starts every five minutes allow rejecting an intro without changing the audio/labels.
        starts=sorted({min(cuts,key=lambda c:abs(c-t)) for t in range(0,int(source['audio_duration']),300)})
        for nominal in (600,1800,3600):
            for a in starts:
                choices=[c for c in cuts if nominal*.75<=c-a<=nominal]
                if not choices:continue
                b=max(choices);q=quality(source,label,vad,acoustic,a,b,policy)
                target=window_segments(segments,a,b)
                key=f'{rid}|{a:.6f}|{b:.6f}|{source["label_sha256"]}'
                wid=hashlib.sha256(key.encode()).hexdigest()[:24]
                duration=b-a;kind='short' if duration<=660 else 'medium' if duration<=1860 else 'long'
                row={**source,'window_id':wid,'source_start':a,'source_end':b,'duration':duration,'bucket':nominal,'duration_class':kind,
                     'quality':q,'acoustic_sha256':sha(ap),'vad_sha256':sha(vp),'target':target}
                if q['reasons']==['unresolved_nonspeech'] and source['split']=='train' and kind=='long':
                    events=reviewed_events(data,row,q)
                    if events:
                        row['target']=with_events(target,events[0]);row['acoustic_review_sha256']=events[1]
                        row['quality']={**q,'reasons':[],'resolved_by':'explicit human acoustic review'}
                    else:acoustic_review_queue.append(row)
                if row['quality']['reasons']:
                    rejections.append({'record_id':rid,'start':a,'end':b,'nominal':nominal,**q});continue
                rows.append(row)
    atomic_json(data/'candidates.json',rows);atomic_json(data/'window-quarantine.json',rejections)
    # One best long crop per source; make listening to ambiguous gaps a small task.
    acoustic_review_queue.sort(key=lambda r:(len(r['quality']['unresolved_nonspeech']),sum(d-c for c,d in r['quality']['unresolved_nonspeech']),r['window_id']))
    unique={}
    for row in acoustic_review_queue:unique.setdefault(row['record_id'],row)
    atomic_json(data/'acoustic-review-queue.json',list(unique.values()))
    report={'candidates':len(rows),'recordings':len({r['record_id'] for r in rows}),
            'counts':dict(collections.Counter('/'.join((r['split'],r['language'],r['duration_class'])) for r in rows)),
            'rejection_reasons':dict(collections.Counter(reason for r in rejections for reason in r['reasons'])),
            'acoustic_pending_records':pending,'policy_sha256':sha(HERE/'policy.json'),'sources_sha256':sha(data/'sources.json'),
            'candidates_sha256':sha(data/'candidates.json'),'selector_sha256':sha(pathlib.Path(__file__))}
    atomic_json(data/'acoustic-selection.json',report);print(json.dumps({k:v for k,v in report.items() if k!='acoustic_pending_records'},indent=2))


def select(data):
    """No oversampling overlapping crops: each source contributes one view to a run."""
    policy=json.loads((HERE/'policy.json').read_text());pool=json.loads((data/'candidates.json').read_text())
    def secondary_rejected(r):
        path=data/'secondary'/(r['window_id']+'.json')
        review=data/'reviews'/(r['window_id']+'.json')
        return (policy.get('secondary_asr_required',False) and path.exists() and json.loads(path.read_text())['status']!='passed') or (review.exists() and json.loads(review.read_text()).get('decision')=='reject')
    remaining=[r for r in pool if r['split']=='train' and not secondary_rejected(r)];chosen=[];used=set();total=0.;lang=collections.Counter();length=collections.Counter();shows=collections.Counter()
    while remaining:
        fits=[r for r in remaining if r['record_id'] not in used and total+r['duration']<=policy['training_audio_hours']*3600]
        if not fits:break
        def score(r):
            n=total+r['duration'];l=lang.copy();d=length.copy();l[r['language']]+=r['duration'];d[r['duration_class']]+=r['duration']
            return (sum((d[k]/n-v)**2/max(v,.01) for k,v in policy['length_audio_fractions'].items())+
                    (l['sk']/n-policy['slovak_audio_fraction'])**2/.3 + .05*shows[r['show']],
                    r['quality']['low_word_confidence_fraction'],r['window_id'])
        r=min(fits,key=score);chosen.append(r);used.add(r['record_id']);total+=r['duration'];lang[r['language']]+=r['duration'];length[r['duration_class']]+=r['duration'];shows[r['show']]+=1
        remaining=[x for x in remaining if x['record_id']!=r['record_id']]
    previous=data/'screening-selection.json';previous_dev=json.loads(previous.read_text())['development'] if previous.exists() else []
    dev=[]
    for language in ('cs','sk'):
        options=[r for r in pool if r['split']=='development' and r['language']==language and r['bucket']==600 and not secondary_rejected(r)]
        old_ids={r['window_id'] for r in previous_dev}
        options.sort(key=lambda r:(r['window_id'] not in old_ids,r['quality']['low_word_confidence_fraction'],r['window_id']))
        seen=set()
        for r in options:
            if r['show'] in seen:continue
            dev.append(r);seen.add(r['show'])
            if len(seen)==4:break
    rows=chosen+dev
    atomic_json(data/'screening-selection.json',{'train':chosen,'development':dev,'policy_sha256':sha(HERE/'policy.json'),
                'candidates_sha256':sha(data/'candidates.json'),'train_hours':total/3600,'development_minutes':sum(r['duration'] for r in dev)/60,
                'language_hours':{k:v/3600 for k,v in lang.items()},'length_hours':{k:v/3600 for k,v in length.items()},
                'status':'awaiting_integrity_admission','reference_quality':'Soniox pseudo-references; optional human review','training_authorized':False})
    print(json.dumps({'train_hours':total/3600,'train_sources':len(chosen),'dev_sources':len(dev),'length_hours':{k:v/3600 for k,v in length.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['candidates','select']);p.add_argument('--data',type=pathlib.Path,required=True)
    a=p.parse_args();globals()[a.command](a.data)
