"""Inventory distinct training speaker references for the ~1,000-voice corpus."""
from array import array
from bisect import bisect_left,bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import wave
from core import ROOT,DATA,read,save,put,sha,digest,encode


def inventory():
    old=read('voices/diverse-candidates.json')
    if old:return old
    sources=json.loads((ROOT/'datasets/releases/qda-recovered-v5/sources.json').read_text())
    candidates=[]
    for index,s in enumerate(sources):
        if s['split']!='train' or s.get('original_split','train')!='train':continue
        lang=s['language']
        if lang not in ('cs','sk'):continue
        corpus='soniox-50h-v2' if s['origin']=='qda-50h-v2' else 'soniox-500h-v1'
        directory=ROOT/'datasets'/corpus/'records'/s['record_id']
        if not all((directory/n).exists() for n in ['transcript.json','vad.json']):continue
        tokens=json.loads((directory/'transcript.json').read_text())['timed_tokens']
        vad=json.loads((directory/'vad.json').read_text())['speech_intervals_seconds']
        ordered=sorted(tokens,key=lambda t:t['start_ms']);starts=[t['start_ms'] for t in ordered];ends=[]
        for t in ordered:ends.append(max(t['end_ms'],ends[-1] if ends else 0))
        groups=[];group=[]
        for t in tokens:
            good=t.get('speaker') is not None and not t.get('is_audio_event') and t.get('language')==lang
            if group and (not good or t['speaker']!=group[-1]['speaker'] or t['start_ms']-group[-1]['end_ms']>1400):
                groups.append(group);group=[]
            if good:group.append(t)
        if group:groups.append(group)
        best={}
        for g in groups:
            start=g[0]['start_ms']/1000
            if start<30 or start>s['audio_duration']-20:continue
            chosen=[t for t in g if t['end_ms']/1000-start<=50]
            if not chosen:continue
            sentence_ends=[i for i,t in enumerate(chosen) if t['text'].rstrip().endswith(('.','?','!')) and t['end_ms']/1000-start>=12]
            if sentence_ends:chosen=chosen[:sentence_ends[-1]+1]
            end=chosen[-1]['end_ms']/1000
            if end-start<12:continue
            conf=statistics.mean(t.get('confidence',0) for t in chosen)
            low=sum(t.get('confidence',0)<.8 for t in chosen)/len(chosen)
            coverage=sum(max(0,min(end,b)-max(start,a)) for a,b in vad)/(end-start)
            if conf<.94 or low>.08 or coverage<.80:continue
            speaker=chosen[0]['speaker']
            overlapping=ordered[bisect_right(ends,start*1000):bisect_left(starts,end*1000)]
            if any(t.get('speaker')!=speaker and t['end_ms']/1000>start for t in overlapping):continue
            score=conf+min(end-start,35)/350+coverage*.05
            if speaker in best and best[speaker]['score']>=score:continue
            best[speaker]={'source_id':s['record_id'],'source_path':s['audio_path'],'source_sha256':s['audio_sha256'],
                'source_split':'train','show':s['show'],'speaker':speaker,'language':[lang],
                'start':max(0,start-.12),'end':end+.12,'transcript':''.join(t['text'] for t in chosen).strip(),
                'confidence':conf,'low_confidence_fraction':low,'speech_fraction':coverage,'score':score,
                'candidate_id':digest((s['record_id']+'/'+str(speaker)).encode())[:24]}
        candidates.extend(best.values())
        if index%25==0:
            save('voices/expansion-status.json',{'phase':'inventory','sources_scanned':index+1,'candidate_speakers':len(candidates),'at':time.time()})
            print('inventory',index+1,len(candidates),flush=True)
    candidates.sort(key=lambda c:(c['language'][0],-c['score'],c['candidate_id']))
    save('voices/diverse-candidates.json',candidates)
    print('candidate speakers',len(candidates),dict(Counter(c['language'][0] for c in candidates)),flush=True)
    return candidates


def extract(c):
    relative=Path('voices/diverse-clips')/c['candidate_id']
    done=read(str(relative)+'.json')
    if done:return done
    # The original full-file hash is checked once per recording by the existing
    # dataset receipts; crop hash below identifies the exact upload/embedding asset.
    audio=ROOT/c['source_path']
    pcm=subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(c['start']),'-i',str(audio),'-t',str(c['end']-c['start']),
                        '-ac','1','-ar','16000','-f','s16le','pipe:1'],check=True,capture_output=True).stdout
    values=array('h');values.frombytes(pcm)
    rms=math.sqrt(sum(x*x for x in values)/len(values))/32768
    clipped=sum(abs(x)>=32760 for x in values)/len(values)
    if rms<.004 or clipped>.003:
        result={**c,'status':'reference_audio_rejected','rms':rms,'clipped_fraction':clipped}
        save(str(relative)+'.json',result);return result
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(pcm)
    data=buf.getvalue();clip_path=str(relative)+'.wav'
    put(clip_path,data)
    result={**c,'status':'reference_ready','clip_path':clip_path,'clip_sha256':digest(data),
            'clip_seconds':len(values)/16000,'sample_rate':16000,'rms_db':20*math.log10(rms),'clipped_fraction':clipped}
    save(str(relative)+'.json',result)
    return result


if __name__=='__main__':
    rows=inventory();results=[]
    pools={l:[c for c in rows if c['language']==[l]] for l in ('cs','sk')}
    rows=[]
    while any(pools.values()):
        for lang in ('cs','sk'):
            if pools[lang]:rows.append(pools[lang].pop(0))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(extract,c) for c in rows]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%25==0:
                save('voices/expansion-status.json',{'phase':'extracting','processed':len(results),'total':len(rows),'at':time.time()})
                print('clips',len(results),'/',len(rows),flush=True)
    ready=[r for r in results if r['status']=='reference_ready']
    ready.sort(key=lambda r:r['candidate_id'])
    save('voices/diverse-references.json',ready)
    save('voices/expansion-status.json',{'phase':'references_ready','references':len(ready),
        'by_language':dict(Counter(c['language'][0] for c in ready)),'at':time.time()})
    print('references ready',len(ready),flush=True)
