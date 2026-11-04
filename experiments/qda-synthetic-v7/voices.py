"""Private voice clones from short, isolated TRAINING speaker excerpts."""
import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import defaultdict
import fcntl
import io
import json
import math
import random
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
import wave
import requests
from core import DATA, HERE, ROOT, digest, encode, env, put, read, save, sha, verified
from cartesia import VERSION


def headers():
    return {'Authorization': 'Bearer '+env()['CARTESIA_API_KEY'], 'Cartesia-Version': VERSION}


def owned():
    result, cursor = [], None
    while True:
        query = {'is_owner': 'true', 'limit': '100'}
        if cursor: query['starting_after'] = cursor
        req = urllib.request.Request('https://api.cartesia.ai/voices?'+urllib.parse.urlencode(query),headers=headers())
        with urllib.request.urlopen(req,timeout=60) as r: data=json.load(r)
        result.extend(data['data'])
        if not data.get('has_more'): return result
        cursor = data.get('next_page') or data['data'][-1]['id']


def candidates():
    cached = read('voices/candidates.json')
    if cached: return cached
    sources = json.loads((ROOT/'datasets/releases/qda-recovered-v5/sources.json').read_text())
    random.Random(20261024).shuffle(sources)
    pool = []
    usable_shows = defaultdict(set)
    for source in sources:
        if source['split'] != 'train' or source.get('original_split','train') != 'train': continue
        lang = source['language']
        if lang not in ('cs','sk'): continue
        directory = ROOT/'datasets/soniox-500h-v1/records'/source['record_id']
        if not (directory/'transcript.json').exists() or not (directory/'vad.json').exists(): continue
        tokens = json.loads((directory/'transcript.json').read_text())['timed_tokens']
        ordered = sorted(tokens,key=lambda t:t['start_ms'])
        starts = [t['start_ms'] for t in ordered]
        prefix_ends = []
        for t in ordered:
            prefix_ends.append(max(t['end_ms'],prefix_ends[-1] if prefix_ends else 0))
        vad = json.loads((directory/'vad.json').read_text())['speech_intervals_seconds']
        groups, group = [], []
        for t in tokens:
            valid = t.get('speaker') is not None and not t.get('is_audio_event') and t.get('language') == lang
            if group and (not valid or t['speaker'] != group[-1]['speaker'] or t['start_ms']-group[-1]['end_ms'] > 1200):
                groups.append(group); group=[]
            if valid: group.append(t)
        if group: groups.append(group)
        for group in groups:
            a=group[0]['start_ms']/1000
            if a < 90 or a > source['audio_duration']-90: continue
            selected=[]
            for t in group:
                if t['end_ms']/1000-a > 55: break
                selected.append(t)
            # Finish a sentence before the 60-second instant-clone limit.
            ends=[i for i,t in enumerate(selected) if t['text'].rstrip().endswith(('.', '?', '!')) and t['end_ms']/1000-a>=30]
            if not ends: continue
            selected=selected[:ends[-1]+1]; b=selected[-1]['end_ms']/1000
            confidence=statistics.mean(t.get('confidence',0) for t in selected)
            low=sum(t.get('confidence',0)<.8 for t in selected)/len(selected)
            coverage=sum(max(0,min(b,y)-max(a,x)) for x,y in vad)/(b-a)
            if confidence<.975 or low>.025 or coverage<.88: continue
            # Reject any other speaker crossing this exact interval, including out-of-order tokens.
            overlapping = ordered[bisect_right(prefix_ends,a*1000):bisect_left(starts,b*1000)]
            if any(t.get('speaker') != selected[0]['speaker'] and t['end_ms']/1000>a for t in overlapping): continue
            pool.append({'source_id':source['record_id'],'source_path':source['audio_path'],
                'source_sha256':source['audio_sha256'],'source_split':'train','show':source['show'],
                'speaker':selected[0]['speaker'],'language':[lang], 'start':max(0,a-.12),'end':b+.12,
                'transcript':''.join(t['text'] for t in selected).strip(),
                'confidence':confidence,'speech_fraction':coverage,'low_confidence_fraction':low})
            usable_shows[lang].add(source['show'])
        if all(len(usable_shows[l])>=20 for l in ('cs','sk')):
            break
    # Diversity first; select at most one identity from each show/recording.
    pool.sort(key=lambda x:x['confidence']+x['speech_fraction']*.03, reverse=True)
    save('voices/candidates.json',pool)
    return pool


def make_clips(per_language=8):
    existing = read('voices/clone-sources.json')
    if existing: return existing
    selected=[]; shows=set(); counts=defaultdict(int)
    for c in candidates():
        lang=c['language'][0]
        if counts[lang]>=per_language or c['show'] in shows: continue
        audio=ROOT/c['source_path']
        if sha(audio)!=c['source_sha256']: raise ValueError('Clone source audio hash mismatch')
        pcm=subprocess.run(['ffmpeg','-v','error','-ss',str(c['start']),'-i',str(audio),'-t',str(c['end']-c['start']),
            '-ac','1','-ar','24000','-f','s16le','pipe:1'],check=True,capture_output=True).stdout
        values=array('h'); values.frombytes(pcm)
        rms=math.sqrt(sum(x*x for x in values)/len(values))/32768
        clipped=sum(abs(x)>=32760 for x in values)/len(values)
        if rms<.008 or clipped>.001: continue
        idx=counts[lang]+1
        identity=f'qda-v7-{lang}-clone-{idx:02d}'
        buf=io.BytesIO()
        with wave.open(buf,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','not compressed')); w.writeframes(pcm)
        clip=buf.getvalue()
        c={**c,'name':identity,'clip_path':f'voices/clips/{identity}.wav','clip_sha256':digest(clip),
            'clip_seconds':len(values)/24000,'rms_db':20*math.log10(rms),'clipped_fraction':clipped,
            'provenance':'Isolated source speaker from already selected training corpus; no evaluation audio or original names in voice labels.'}
        put(c['clip_path'],clip); selected.append(c); shows.add(c['show']); counts[lang]+=1
        print('clip',identity,round(c['clip_seconds'],2),flush=True)
        if all(counts[l]>=per_language for l in ('cs','sk')): break
    if len(selected)!=2*per_language: raise ValueError(f'Only {dict(counts)} usable distinct clone sources')
    save('voices/clone-sources.json',selected)
    return selected


def create_clone(source):
    relative='voices/clones/'+source['name']
    existing=read(relative+'.json')
    if existing: return existing
    # Reconcile by exact deterministic private name before any uncertain POST retry.
    matches=[v for v in owned() if v['name']==source['name']]
    if len(matches)>1: raise ValueError('Duplicate private voice names')
    if matches:
        voice=matches[0]
    else:
        if read(relative+'.intent.json'):
            raise RuntimeError('Unresolved previous voice clone request: '+source['name'])
        audio=verified(source['clip_path'],source['clip_sha256'])
        payload={'name':source['name'],'language':source['language'][0],'access':'private',
            'description':'Private ASR training voice; source provenance stored locally.','tagline':'QDA synthetic training'}
        save(relative+'.request.json',{'fields':payload,'clip_sha256':source['clip_sha256']})
        save(relative+'.intent.json',{'submitted_at':time.time(),'clip_sha256':source['clip_sha256']})
        response=requests.post('https://api.cartesia.ai/voices/clone',headers=headers(),data=payload,
            files={'clip':(source['name']+'.wav',audio,'audio/wav')},timeout=180)
        if response.status_code!=200:
            save(relative+'.error.json',{'status':response.status_code,'body':response.text[:3000]})
            raise RuntimeError('Clone HTTP '+str(response.status_code))
        voice=response.json()
    if voice.get('access')!='private' or not voice.get('id'):
        raise ValueError('Unexpected clone visibility or identity')
    record={**voice,'kind':'private_clone','source_clip_sha256':source['clip_sha256'],
        'source_record_id':source['source_id'],'source_speaker':source['speaker'],'source_split':'train',
        'language':source['language'],'created_locally_at':time.time()}
    save(relative+'.json',record)
    print('clone',source['name'],'ready',flush=True)
    return record


def registry():
    voices=[]
    for lang in ('cs','sk'):
        for v in json.loads((HERE/'research'/f'voices-{lang}.json').read_text())['data']:
            voices.append({**v,'kind':'library','language':[lang]})
    for path in sorted((DATA/'voices/clones').glob('*.json')) if (DATA/'voices/clones').exists() else []:
        v=json.loads(path.read_text())
        if v.get('kind')=='private_clone': voices.append(v)
    save('voices/registry.json',voices)
    return voices


def assign(session):
    relative='sessions/'+session['id']+'/voices.json'
    ready=read(relative)
    if ready: return ready
    (DATA/'voices').mkdir(parents=True,exist_ok=True)
    with (DATA/'voices/.assignment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ready=read(relative)
        if ready:return ready
        bank=read('voices/registry.json',[]);lang=session['language'][0];n=session['speaker_count']
        pool=[v for v in bank if v['language']==[lang]]
        usage=defaultdict(float);appearances=defaultdict(int)
        for path in (DATA/'sessions').glob('*/voices.json'):
            spec=read(path.parent.relative_to(DATA)/'spec.json',{})
            # Reserve the planned speaker share immediately, so concurrent jobs do
            # not all pick the same unused voice. Completed actual usage replaces it.
            receipt=read(path.parent.relative_to(DATA)/'receipt.json')
            duration=(receipt or {}).get('duration_seconds',spec.get('planned_seconds',120))
            rows=json.loads(path.read_text())
            for v in rows:usage[v['voice_id']]+=duration/max(1,len(rows));appearances[v['voice_id']]+=1
        rng=random.Random(session['seed']);rng.shuffle(pool)
        pool.sort(key=lambda v:(appearances[v['id']]>0,usage[v['id']],appearances[v['id']]))
        chosen=pool[:n];rng.shuffle(chosen)
        if len(chosen)!=n or len({v['id'] for v in chosen})!=n:raise RuntimeError('Voice bank still preparing')
        result=[{'speaker':i,'voice_id':v['id'],'kind':v['kind'],'language':[lang],
            'name':v['name'],'speed':rng.choice([.96,1.0,1.04]),'allocation':'unused first, then least reserved audio'} for i,v in enumerate(chosen)]
        save(relative,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--create',action='store_true');args=p.parse_args()
    sources=make_clips()
    if args.create:
        for source in sources: create_clone(source); registry()
    print('registered',len(registry()),flush=True)
