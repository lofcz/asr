"""Resumable script -> TTS -> composed audio/labels; never autostarts training."""
import argparse
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import threading
import time
import wave
from core import DATA, HERE, REPLICA, ROOT, atomic, digest, encode, put, read, save, sha, verified
from plan import plan
from generate_scripts import make_part,validate as validate_script,ScriptQuotaBlocked
from voices import assign, registry
from cartesia import RATE, request_for, synthesize
from target_text import canonicalize
from vocabulary import contract as vocabulary_contract,validate_contract,REVISION
from concurrency import TTSLimit,BoundedPool

STOP = threading.Event()
STATUS_LOCK = threading.Lock()
TTS_LIMIT=TTSLimit(15)
LAST_PROGRESS=0.
PROGRESS_CACHE=None


def control():
    if STOP.is_set() or read('control.json',{}).get('status') == 'paused':
        raise InterruptedError('Dataset generation paused; completed paid responses remain saved')
    if not (ROOT/'.asr-volume-id').exists(): raise OSError('Dataset SSD missing')
    if min(shutil.disk_usage(DATA).free,shutil.disk_usage(REPLICA).free)<12*2**30:
        raise OSError('Insufficient space for two durable dataset copies')


def normalized(text):
    # Retain all diacritics; Czech tři and Slovak tri are never collapsed.
    return re.sub(r'[^\w]+','',canonicalize(text)[0].casefold(),flags=re.UNICODE)


def audio_quality(identity, text):
    relative=Path('utterances')/identity
    cached=read(relative/'quality.json')
    if cached: return cached
    receipt=read(relative/'receipt.json')
    data=verified(relative/'audio.wav',receipt['audio_sha256'])
    with wave.open(io.BytesIO(data),'rb') as w:
        if w.getparams()[:3]!=(1,2,RATE): raise ValueError('Unexpected TTS audio format')
        pcm=w.readframes(w.getnframes())
    values=array('h'); values.frombytes(pcm)
    rms=math.sqrt(sum(x*x for x in values)/len(values))/32768
    clipping=sum(abs(x)>=32760 for x in values)/len(values)
    words=read(relative/'words.json')
    native_text=' '.join(w['text'] for w in words)
    warnings=[]
    if normalized(native_text)!=normalized(text): warnings.append('native_normalized_text_differs_from_script')
    if clipping>.002: warnings.append('audio_clipping')
    if rms<.002: warnings.append('low_audio_level')
    duration=len(values)/RATE
    speed=len(text.split())/duration*60
    if speed<55 or speed>260: warnings.append('unusual_speaking_rate')
    # Complete all paid artifacts even if a quality flag needs subsequent review.
    result={'status':'review' if warnings else 'checked','warnings':warnings,
        'rms_db':20*math.log10(max(rms,1e-12)),'clipped_fraction':clipping,
        'words_per_minute':speed,'native_text':native_text,
        'script_native_text_equal_ignoring_punctuation':normalized(native_text)==normalized(text),
        'audio_sha256':receipt['audio_sha256'], 'word_count':len(words),
        'note':'Structural/audio checks, not proof that TTS pronounced every supplied word correctly.'}
    save(relative/'quality.json',result)
    return result


def tts_one(session, part, index, turn, voice, original_timestamps=False):
    control()
    request=request_for(turn['text'],voice['language'][0],voice['voice_id'],voice['speed'],original_timestamps)
    for attempt in range(7):
        try:
            with TTS_LIMIT.slot(control):identity,receipt=synthesize(request)
            break
        except RuntimeError as e:
            # Only an explicit HTTP rejection is retryable; lost/ambiguous POSTs are retained.
            if str(e)!='Cartesia HTTP 429' or attempt==6: raise
            TTS_LIMIT.rejected()
            if STOP.wait(min(60,2**attempt+random.random())): raise InterruptedError('Stopping')
    qc=audio_quality(identity,turn['text'])
    result={'part':part,'turn':index,'speaker':turn['speaker'],'text':turn['text'],
        'language':voice['language'],'voice_id':voice['voice_id'],'voice_kind':voice['kind'],
        'utterance_id':identity,'audio_sha256':receipt['audio_sha256'],
        'samples':receipt['samples'],'duration_seconds':receipt['duration_seconds'],'quality':qc['status']}
    save(f'sessions/{session["id"]}/parts/{part:03d}/turn-{index:03d}.json',result)
    return result


def assemble(session, parts):
    relative=Path('sessions')/session['id']
    ready=read(relative/'receipt.json')
    if ready:
        for name,key in [('audio.flac','audio_sha256'),('labels.json','labels_sha256'),('composition.json','composition_sha256')]:
            verified(relative/name,ready[key])
        return ready
    rng=random.Random(session['seed'])
    offset=0; composition=[]; segments=[]; words=[]; events=[]; qc_warnings=[]
    directory=DATA/relative; directory.mkdir(parents=True,exist_ok=True)
    wav_path=directory/'.assembly.wav'
    with wave.open(str(wav_path),'wb') as out:
        out.setparams((1,2,RATE,0,'NONE','not compressed'))
        previous_speaker=None
        for item in [t for p in parts for t in p['turns']]:
            identity=item['utterance_id']; root=Path('utterances')/identity
            gap=round(RATE*(rng.uniform(.12,.48) if previous_speaker is not None and item['speaker']!=previous_speaker else rng.uniform(.04,.14))) if offset else round(RATE*.16)
            out.writeframes(b'\0\0'*gap)
            events.append({'start':offset/RATE,'end':(offset+gap)/RATE,'kind':'silence','provenance':'inserted zero PCM samples'})
            offset+=gap; start=offset/RATE
            audio=verified(root/'audio.wav',item['audio_sha256'])
            with wave.open(io.BytesIO(audio),'rb') as w:
                pcm=w.readframes(w.getnframes())
            out.writeframes(pcm)
            count=len(pcm)//2
            if count!=item['samples']: raise ValueError('Wrong paired TTS sample count')
            native=read(root/'words.json')
            lexical=[w for w in native if any(c.isalnum() for c in w['text'])]
            if not lexical: raise ValueError('No lexical TTS timestamps')
            a=start+lexical[0]['start']; b=start+lexical[-1]['end']
            if not start<=a<=b<=(offset+count)/RATE+.05: raise ValueError('Invalid composed turn bounds')
            # Adjacent chunks of the same speaking turn are merged in the ASR target.
            if segments and segments[-1]['Speaker']==item['speaker'] and a-segments[-1]['End']<1.5:
                segments[-1]['Content']+=' '+item['text']; segments[-1]['End']=round(b,6)
            else:
                segments.append({'Start':round(a,6),'End':round(b,6),'Speaker':item['speaker'],'Content':item['text']})
            words.extend({'text':w['text'],'start':round(start+w['start'],6),'end':round(start+w['end'],6),
                          'speaker':item['speaker'],'utterance_id':identity} for w in native)
            composition.append({**item,'offset_samples':offset,'end_samples':offset+count,'sample_rate':RATE})
            qc=read(root/'quality.json')
            if qc['warnings']: qc_warnings.append({'utterance_id':identity,'warnings':qc['warnings']})
            offset+=count; previous_speaker=item['speaker']
    flac_path=directory/'.assembly.flac'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(wav_path),'-ar','16000','-ac','1',
                    '-c:a','flac',str(flac_path)],check=True)
    audio=flac_path.read_bytes(); put(relative/'audio.flac',audio)
    # Verify the encoded training asset actually decodes and preserves duration.
    checked=subprocess.run(['ffmpeg','-v','error','-i',str(directory/'audio.flac'),'-f','s16le','-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True)
    decoded_samples=len(checked.stdout)//2
    if abs(decoded_samples/16000-offset/RATE)>1/16000: raise ValueError('Encoded audio duration mismatch')
    if len({s['Speaker'] for s in segments})!=session['speaker_count']: raise ValueError('Composed speaker count mismatch')
    labels={'version':'cartesia-qda-v7-native-v1','session_id':session['id'],'language':session['language'],
        'duration':decoded_samples/16000,'audio_path':str(relative/'audio.flac'),
        'segments':segments,'words':words,'customized_context':session['customized_context'],
        'context_provenance':session['context_provenance'],'audio_events':events,
        'timestamp_source':'Cartesia native word timestamps; original-text mode for vocabulary-v4, legacy normalized events retained',
        'alignment_recipe':read('alignment.json'),'label_status':'synthetic_native_timestamps',
        'quality_warnings':qc_warnings,'split':'train'}
    save(relative/'labels.json',labels);save(relative/'composition.json',composition)
    receipt={'status':'complete','session_id':session['id'],'cohort':session['cohort'],
        'language':session['language'],'speaker_count':session['speaker_count'],'planned_seconds':session['planned_seconds'],
        'duration_seconds':decoded_samples/16000,'audio_sha256':digest(audio),'labels_sha256':digest(encode(labels)),
        'composition_sha256':digest(encode(composition)),'utterances':len(composition),
        'quality_review_utterances':len(qc_warnings),'created_at':time.time(),
        'training_ready':not qc_warnings,'training_autostart':False}
    save(relative/'receipt.json',receipt)
    wav_path.unlink();flac_path.unlink()
    print(json.dumps({'event':'session_complete',**receipt},ensure_ascii=False),flush=True)
    return receipt


def run_session(session, tts_pool):
    control()
    relative=Path('sessions')/session['id']
    existing=read(relative/'receipt.json')
    if existing: return existing
    session=dict(session)
    contract_files=sorted((DATA/'vocabulary/contracts'/session['id']).glob('part-*.json'))
    contracts={int(p.stem.split('-')[1]):json.loads(p.read_text()) for p in contract_files}
    vocabulary_plan=read('vocabulary/plan.json',{})
    legacy_parts=vocabulary_plan.get('legacy_parts',{}).get(session['id'],[])
    fixed_parts=sorted(set(contracts)|set(legacy_parts))
    if vocabulary_plan and not fixed_parts:raise ValueError('No vocabulary obligations or retained paid parts for this session')
    if contracts and session.get('customized_context'):
        names=list(dict.fromkeys(t['name'] for c in contracts.values() for t in c['terms']))
        rng=random.Random(session['seed']);rng.shuffle(names)
        session['customized_context']=', '.join(names[:4])
        session['context_provenance']='Vocabulary obligations selected before generating this recording'
    voices=assign(session)
    save(relative/'spec.json',session)
    parts=[]; elapsed=0.; total_words=0; previous=None; part=0
    # Timings from already generated parts guide subsequent script length.
    while (part<=max(fixed_parts) if vocabulary_plan else elapsed<session['planned_seconds']*.97):
        control()
        remaining=max(20,session['planned_seconds']-elapsed)
        final=remaining<=360
        seconds=remaining if final else 300
        vc=contracts.get(part)
        if vocabulary_plan:
            final=part==max(fixed_parts)
            if vc:
                weight=sum(c['planned_seconds'] for index,c in contracts.items() if index>=part)
                seconds=max(20,remaining*vc['planned_seconds']/weight)
            elif part not in legacy_parts:raise ValueError('Unplanned unpaid script part')
        rate=min(190,max(110,total_words/elapsed*60)) if elapsed else 145
        script=make_part(session,part,seconds,final,previous,rate)
        if vc:
            if script.get('vocabulary_contract_sha256')!=vc['sha256']:raise ValueError('Cached script has the wrong vocabulary contract')
            validate_script(script,{**session,'required_terms':[],'minimum_brand_mentions':0,'vocabulary_contract':vc},script['target_words'])
            counts=validate_contract(script,vc)
            save(relative/f'parts/{part:03d}/vocabulary-verified.json',{'contract_sha256':vc['sha256'],
                'script_sha256':digest(encode(script)),'counts':counts,'verified_before_tts':True,'at':time.time()})
        futures=[tts_pool.submit(tts_one,session,part,i,t,voices[t['speaker']],bool(vc)) for i,t in enumerate(script['turns'])]
        results=[future.result() for future in futures]
        record={'part':part,'script_path':script.get('script_path',f'scripts/{session["id"]}/part-{part:03d}.json'),
                'script_sha256':digest(encode(script)),'turns':results,
                'duration_seconds':sum(t['duration_seconds'] for t in results)}
        save(relative/f'parts/{part:03d}/complete.json',record)
        parts.append(record);elapsed+=record['duration_seconds'];total_words+=script['words']
        previous=script;part+=1
        progress()
        if final: break
        if part>math.ceil(session['planned_seconds']/120)+3: raise RuntimeError('Session failed to reach its duration target')
    result=assemble(session,parts)
    save(relative/'state.json',{'status':'complete','updated_at':time.time()})
    progress()
    return result


def progress(force=False):
    global LAST_PROGRESS,PROGRESS_CACHE
    with STATUS_LOCK:
        if not force and PROGRESS_CACHE is not None and time.time()-LAST_PROGRESS<15:return PROGRESS_CACHE
        p=plan();completed=[];failed=[];waiting=[];parts=0;utterance_seconds=0.;by_lang=defaultdict(float);by_speakers=defaultdict(float)
        voice_kind=Counter()
        for path in (DATA/'sessions').glob('*/receipt.json') if (DATA/'sessions').exists() else []:
            r=json.loads(path.read_text());completed.append(r)
            by_lang[','.join(r['language'])]+=r['duration_seconds']/3600
            by_speakers[str(r['speaker_count'])]+=r['duration_seconds']/3600
        for path in (DATA/'sessions').glob('*/state.json') if (DATA/'sessions').exists() else []:
            d=json.loads(path.read_text())
            if d.get('status')=='error': failed.append({'session_id':path.parent.name,**d})
            if d.get('status')=='waiting_for_script_quota':waiting.append(path.parent.name)
        for path in (DATA/'sessions').glob('*/parts/*/complete.json') if (DATA/'sessions').exists() else []:
            d=json.loads(path.read_text());parts+=1;utterance_seconds+=d['duration_seconds']
            for t in d['turns']: voice_kind[t['voice_kind']]+=t['duration_seconds']/3600
        budget=read('budget/reservations.json',{})
        control_status=read('control.json',{}).get('status')
        result={'status':'paused' if STOP.is_set() or control_status=='paused' else 'waiting_for_script_quota' if control_status=='waiting_for_script_quota' else 'preparing','updated_at':time.time(),
            'target_hours':60,'planned_sessions':len(p['sessions']),'complete_sessions':len(completed),
            'complete_hours':sum(r['duration_seconds'] for r in completed)/3600,
            'tts_completed_part_hours':utterance_seconds/3600,'complete_parts':parts,
            'hours_by_language':dict(by_lang),'hours_by_speaker_count':dict(by_speakers),
            'tts_hours_by_voice_kind':dict(voice_kind),'reserved_characters':budget.get('characters',0),
            'character_ceiling':p['character_ceiling'],'errors':failed,
            'script_generation':read('scripts-control.json',{}),'sessions_waiting_for_script_quota':len(waiting),
            'training_autostart':False,'timestamp_source':'cartesia-native',
            'configured_script_workers':p.get('script_workers'),
            'tts_concurrency':read('concurrency.json',{}),'vocabulary_revision':p.get('vocabulary_revision'),
            'diarization_source':'known voice assignments','selected_alignment_policy':read('alignment.json')}
        save('progress.json',result)
        entries=[]
        for r in sorted(completed,key=lambda r:r['session_id']):
            entries.append({'id':r['session_id'],'audio':f'sessions/{r["session_id"]}/audio.flac',
                'labels':f'sessions/{r["session_id"]}/labels.json','duration':r['duration_seconds'],
                'audio_sha256':r['audio_sha256'],'labels_sha256':r['labels_sha256'],
                'language':r['language'],'speakers':r['speaker_count'],'cohort':r['cohort'],
                'training_ready':r['training_ready'],'split':'train'})
        put('manifest.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in entries).encode())
        LAST_PROGRESS=time.time();PROGRESS_CACHE=result
        return result


def main():
    global TTS_LIMIT
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12);parser.add_argument('--tts-workers',type=int,default=15)
    parser.add_argument('--limit',type=int);parser.add_argument('--status',action='store_true');args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);REPLICA.mkdir(parents=True,exist_ok=True)
    if args.status: print(json.dumps(progress(force=True),ensure_ascii=False,indent=2));return
    TTS_LIMIT=TTSLimit(args.tts_workers)
    with (REPLICA/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:STOP.set())
        p=plan()
        if p.get('vocabulary_revision'):
            validated=read('vocabulary/validation-ready.json',{})
            if not validated.get('passed') or validated.get('plan_sha256')!=sha(DATA/'vocabulary/plan.json'):
                raise ValueError('The current vocabulary plan must pass validation before paid generation resumes')
        while len(registry())<p.get('minimum_voices_before_resume',100):
            print('Waiting for the expanded private voice bank',flush=True)
            if STOP.wait(10): return
        save('control.json',{'status':'running','started_at':time.time(),'training_autostart':False})
        sessions=[s for s in p['sessions'] if not read(f'sessions/{s["id"]}/receipt.json')]
        sessions.sort(key=lambda s:(not bool(read(f'scripts-v4/{s["id"]}/part-000.json') or read(f'scripts/{s["id"]}/part-000.json')),
            not (s.get('canonical_product') in ('ScioBot','ScioŠkola','ScioLink') and s['planned_seconds']<=300)))
        if args.limit is not None: sessions=sessions[:args.limit]
        def job(s):
            try:
                save(f'sessions/{s["id"]}/state.json',{'status':'running','updated_at':time.time()})
                return run_session(s,tts_pool)
            except Exception as e:
                state={'status':'waiting_for_script_quota' if isinstance(e,ScriptQuotaBlocked) else 'paused' if isinstance(e,InterruptedError) else 'error','error':str(e),'updated_at':time.time()}
                save(f'sessions/{s["id"]}/state.json',state)
                print(json.dumps({'event':'session_error','session_id':s['id'],**state}),flush=True)
                progress(force=not isinstance(e,ScriptQuotaBlocked))
                # Account-wide failures stop further paid work rather than trying every session.
                if any(x in str(e) for x in ['HTTP 401','HTTP 402','HTTP 403','character budget','Insufficient space']): STOP.set()
                return None
        with BoundedPool(args.tts_workers,control) as tts_pool, ThreadPoolExecutor(max_workers=args.workers) as scripts_pool:
            futures={scripts_pool.submit(job,s):s for s in sessions}
            for future in as_completed(futures):
                future.result()
                if STOP.is_set():
                    for f in futures:f.cancel()
                    break
        result=progress(force=True)
        if read('scripts-control.json',{}).get('status')=='quota_blocked':
            result['status']='waiting_for_script_quota';save('progress.json',result)
            save('control.json',{'status':'waiting_for_script_quota','at':time.time(),
                'reason':'Cached approved scripts drained; new scripts require restored Codex quota','training_autostart':False})
            return
        if len(result['errors']): raise SystemExit(1)
        if result['complete_sessions']==result['planned_sessions']:
            result['status']='generation_complete';save('progress.json',result)


if __name__=='__main__': main()
