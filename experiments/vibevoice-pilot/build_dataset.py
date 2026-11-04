import json,pathlib,subprocess,collections,hashlib
BASE=pathlib.Path(__file__).resolve().parent

def merge_words(tokens):
    words=[]
    for t in tokens:
        if t.get('speaker') is None or t.get('is_audio_event') or t.get('text','').strip() in ('<end>','<fin>') or t.get('start_ms',-1)<0:
            continue
        if words and not t['text'][:1].isspace() and t['speaker']==words[-1]['speaker']:
            w=words[-1];w['text']+=t['text'];w['end_ms']=max(w['end_ms'],t['end_ms']);w['confidence_sum']+=t.get('confidence',0);w['pieces']+=1
        else:words.append({**t,'confidence_sum':t.get('confidence',0),'pieces':1})
    for w in words:w['confidence']=w['confidence_sum']/w['pieces']
    return words

def segments(tokens,offset):
    out=[];speakers={}
    for t in tokens:
        speaker=str(t['speaker']);sid=speakers.setdefault(speaker,len(speakers));start=t['start_ms']/1000-offset;end=t['end_ms']/1000-offset
        if out and out[-1]['speaker']==sid and start-out[-1]['end']<.8 and end-out[-1]['start']<=20:
            out[-1]['text']+=t['text'];out[-1]['end']=round(end,3)
        else:out.append({'speaker':sid,'text':t['text'],'start':round(start,3),'end':round(end,3)})
    for s in out:s['text']=s['text'].strip()
    return [s for s in out if s['text']]

def main():
    out=BASE/'dataset';(out/'audio').mkdir(parents=True,exist_ok=True);rows=[]
    for source in json.loads((BASE/'sample/sources.json').read_text()):
        raw=json.loads((BASE/'sample'/source['id']/'soniox.json').read_text())
        tokens=[t for t in merge_words(raw['tokens']) if t.get('speaker') is not None and not t.get('is_audio_event') and t.get('text','').strip() not in ('<end>','<fin>') and t.get('start_ms',-1)>=1000 and t.get('end_ms',1e9)<=179000]
        assert all(t['start_ms']<=t['end_ms'] for t in tokens)
        for target in (30,60,120):
            i=0;part=0
            while i<len(tokens):
                start=max(0,tokens[i]['start_ms']/1000-.1);j=i+1
                while j<len(tokens) and tokens[j]['end_ms']/1000-start<target:j+=1
                # Prefer a pause or speaker turn near the target, without splitting a token.
                candidates=[k for k in range(max(i+1,j-25),min(len(tokens),j+25)) if abs(tokens[k]['start_ms']/1000-start-target)<5 and tokens[k]['start_ms']>=tokens[k-1]['end_ms'] and (tokens[k]['speaker']!=tokens[k-1]['speaker'] or tokens[k]['start_ms']-tokens[k-1]['end_ms']>=150)]
                if candidates:j=min(candidates,key=lambda k:abs(tokens[k]['start_ms']/1000-start-target))
                ts=tokens[i:j];end=min(180,max(t['end_ms'] for t in ts)/1000+.1);duration=end-start
                if duration<8:break
                uid=f"{source['id']}-{target}-{part:02d}";rel=f'audio/{uid}.flac';path=out/rel
                if not path.exists() or path.stat().st_size==0:subprocess.run(['ffmpeg','-y','-v','error','-ss',str(start),'-i',str(BASE/'sample'/source['id']/'audio.flac'),'-t',str(duration),'-ar','24000','-ac','1','-c:a','flac','-frame_size','2304',str(path)],check=True)
                segs=segments(ts,start)
                assert segs and all(0<=s['start']<=s['end']<=duration+.001 for s in segs)
                rows.append({'id':uid,'source_id':source['id'],'show_id':source['show_id'],'split':source['split'],'language':source['language'],'bucket':target,'audio_path':rel,'audio_duration':round(duration,3),'source_start':round(source['source_start']+start,3),'segments':segs,'mean_confidence':sum(t.get('confidence',0) for t in ts)/len(ts),'speaker_count':len({s['speaker'] for s in segs})})
                i=j;part+=1
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary={'unique_source_minutes':36,'note':'30/60/120s variants overlap; use one bucket per experiment, never combine as independent hours','rows':len(rows),'counts':dict(collections.Counter(f"{r['split']}/{r['language']}/{r['bucket']}" for r in rows)),'segmentation_version':'word-boundary-v2', 'multispeaker_rows':sum(r['speaker_count']>1 for r in rows),'low_confidence_rows':sum(r['mean_confidence']<.8 for r in rows)}
    train={r['show_id'] for r in rows if r['split']=='train'};val={r['show_id'] for r in rows if r['split']=='validation'};assert not train&val
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
