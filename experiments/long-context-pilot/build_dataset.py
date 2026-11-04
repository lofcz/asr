import json,pathlib,subprocess,sys,hashlib
BASE=pathlib.Path(__file__).resolve().parent
from transcript_utils import merge_words

def make_segments(tokens,offset):
    result=[];mapping={}
    for t in tokens:
        speaker=mapping.setdefault(str(t['speaker']),len(mapping));start=t['start_ms']/1000-offset;end=t['end_ms']/1000-offset
        prev=result[-1] if result else None
        split=not prev or prev['speaker']!=speaker or start-prev['end']>.8 or end-prev['start']>30 or (end-prev['start']>8 and prev['text'].rstrip().endswith(('.','?','!')))
        if split:result.append({'speaker':speaker,'session_speaker':str(t['speaker']),'text':t['text'],'start':round(start,3),'end':round(end,3)})
        else:prev['text']+=t['text'];prev['end']=round(end,3)
    return [{**s,'text':s['text'].strip()} for s in result if s['text'].strip()]

def main():
    out=BASE/'dataset';(out/'audio').mkdir(parents=True,exist_ok=True);rows=[]
    for source in json.loads((BASE/'sample/sources.json').read_text()):
        data=json.loads((BASE/'sample'/source['id']/'soniox.json').read_text());words=merge_words(data['tokens'])
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(BASE/'sample'/source['id']/'audio.flac')],text=True))
        sizes=(600,1800,3600) if source['split']=='train' else (600,)
        for size in sizes:
            bounds=[0.0]
            if size<duration:
                target=size
                while target<duration-30:
                    candidates=[(w['start_ms']/1000+words[i-1]['end_ms']/1000)/2 for i,w in enumerate(words) if i and abs(w['start_ms']/1000-target)<5 and w['start_ms']>=words[i-1]['end_ms'] and (w['speaker']!=words[i-1]['speaker'] or words[i-1]['text'].rstrip().endswith(('.','?','!')))]
                    bounds.append(min(candidates,key=lambda x:abs(x-target)) if candidates else target);target+=size
            bounds.append(duration)
            for part,(start,end) in enumerate(zip(bounds,bounds[1:])):
                tokens=[w for w in words if w['start_ms']/1000>=start and w['end_ms']/1000<=end]
                if not tokens:continue
                uid=f"{source['id']}-{size}-{part:02}";rel=f'audio/{uid}.flac';path=out/rel
                if not path.exists():
                    if start==0 and end==duration:path.symlink_to('../../sample/'+source['id']+'/audio.flac')
                    else:subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',str(BASE/'sample'/source['id']/'audio.flac'),'-t',str(end-start),'-ar','24000','-ac','1','-sample_fmt','s16','-c:a','flac','-frame_size','2304',str(path)],check=True)
                segs=make_segments(tokens,start)
                assert all(0<=s['start']<=s['end']<=end-start+.002 for s in segs)
                rows.append({'id':uid,'source_id':source['id'],'show_id':source['show_id'],'split':source['split'],'language':source['language'],'bucket':size,'audio_path':rel,'audio_duration':round(end-start,3),'source_start':round(source['source_start']+start,3),'segments':segs,'speaker_count':len({s['speaker'] for s in segs}),'mean_confidence':sum(t['confidence'] for t in tokens)/len(tokens)})
    assert not {r['show_id'] for r in rows if r['split']=='train'} & {r['show_id'] for r in rows if r['split']=='validation'}
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary={'unique_minutes':sum(s['duration'] for s in json.loads((BASE/'sample/sources.json').read_text()))/60,'note':'Same training session at 10/30/60-minute resolutions. Do not count variants as independent hours. Full-hour training-source inference is a capacity/coverage probe, not held-out accuracy.','rows':[{'id':r['id'],'seconds':r['audio_duration'],'speakers':r['speaker_count'],'split':r['split'],'segments':len(r['segments'])} for r in rows]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
