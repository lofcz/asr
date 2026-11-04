"""Read-only investigation of raw provider labels, actual crops, and exported PCM."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import wave

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'datasets/releases/qda-full-continuous-v8'
SYNTH=ROOT/'datasets/cartesia-qda-v7'


def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_bytes())


def real_source(item):
    rid,rows=item;r=rows[0]
    try:
        origin='soniox-50h-v2' if r['origin']=='qda-50h-v2' else 'soniox-500h-v1'
        raw_bytes=(ROOT/'datasets'/origin/r['raw_transcript_path']).read_bytes()
        label_bytes=(DATA/r['label_path']).read_bytes()
        assert digest(raw_bytes)==r['raw_transcript_sha256'], 'raw SHA'
        assert digest(label_bytes)==r['label_sha256'], 'label SHA'
        raw=json.loads(raw_bytes);label=json.loads(label_bytes);tokens=raw['tokens']
        speech=[t for t in tokens if isinstance(t.get('text'),str) and t['text'] and t['text'].strip() not in ('<end>','<fin>')]
        assert ''.join(t['text'] for t in speech)==''.join(s['text'] for s in label['segments']), 'raw text differs'
        for segment in label['segments']:
            parts=[tokens[i] for i in segment['token_indices']]
            assert segment['text']==''.join(t['text'] for t in parts), 'segment text differs'
            assert {str(t['speaker']) for t in parts}=={str(segment['speaker'])}, 'speaker reassignment'
            assert abs(segment['start']-min(t['start_ms'] for t in parts)/1000)<1e-6, 'start shifted'
            assert abs(segment['end']-max(t['end_ms'] for t in parts)/1000)<1e-6, 'end shifted'
        for row in rows:
            a,b=row['source_start'],row['source_end'];expected=[];speakers={}
            for segment in label['segments']:
                start,end=segment['start'],segment['end']
                included=start<b and end>a if start!=end else a<=start<b or (b==row['audio_duration'] and start==b)
                if not included:continue
                assert start>=a-1e-6 and end<=b+1e-6, 'crop cuts labelled speech'
                if not segment['text'].strip():continue
                expected.append({'Start':round(start-a,2),'End':round(end-a,2),
                    'Speaker':speakers.setdefault(segment['speaker'],len(speakers)),'Content':segment['text']})
            assert expected==row['target'], 'target differs from raw-source crop'
        return {'record_id':rid,'windows':len(rows),'passed':True}
    except Exception as e:return {'record_id':rid,'windows':len(rows),'passed':False,'error':str(e)}


def synthetic_pcm(sid):
    try:
        composition=read(SYNTH/'sessions'/sid/'composition.json')
        receipt=read(SYNTH/'training/receipts'/(sid+'.json'))
        voice_map={voice:int(speaker) for speaker,voice in receipt['known_speaker_voice_ids'].items()}
        words=read(SYNTH/receipt['words_path']);last=0;checked=[]
        pcm=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(SYNTH/receipt['audio_path']),'-f','s16le','-ac','1','-ar','24000','-'])
        assert len(pcm)//2==composition[-1]['end_samples'],'final PCM duration'
        for i in sorted({0,len(composition)//2,len(composition)-1}):
            item=composition[i]
            with wave.open(str(SYNTH/'utterances'/item['utterance_id']/'audio.wav'),'rb') as f:
                assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(24000,1,2),'source PCM format'
                original=f.readframes(f.getnframes())
            actual=pcm[2*item['offset_samples']:2*item['end_samples']]
            assert actual==original,'exported PCM differs from original utterance'
            timed=[w for w in words if w['utterance_id']==item['utterance_id'] and item['offset_samples']/24000<=w['start']<item['end_samples']/24000]
            assert timed and all(w['speaker']==voice_map[item['voice_id']] for w in timed),'word speaker differs from assigned voice'
            native=read(SYNTH/'utterances'/item['utterance_id']/'words.json')
            assert abs(timed[0]['start']-(native[0]['start']+item['offset_samples']/24000))<.001,'native timestamp offset differs'
            checked.append({'turn':i,'utterance_id':item['utterance_id'],'samples':len(original)//2,'exact_pcm':True})
        return {'record_id':sid,'passed':True,'duration':len(pcm)/48000,'checked_utterances':checked}
    except Exception as e:return {'record_id':sid,'passed':False,'error':str(e)}


def main():
    plan=read(ROOT/'experiments/qda-full-continuous-v8/plan.json')
    rows=[r for pair in plan['probe_pairs'] for r in pair]
    by_real=defaultdict(list);by_synth={}
    for r in rows:
        if r['dataset_kind']=='real':by_real[r['record_id']].append(r)
        else:by_synth[r['record_id']]=r
    real=[]
    with ThreadPoolExecutor(max_workers=4) as workers:
        for result in workers.map(real_source,sorted(by_real.items())):
            real.append(result)
            if len(real)%100==0:print(json.dumps({'real_sources_checked':len(real),'failures':sum(not r['passed'] for r in real)}),flush=True)
    choices=[]
    for lang in ('cs','sk'):
        for cohort in ('core','single_speaker_extra'):
            candidates=sorted((r for r in by_synth.values() if r['language']==lang and r['cohort']==cohort),key=lambda r:r['audio_duration'])
            choices.extend(candidates[i]['record_id'] for i in (0,len(candidates)//2,len(candidates)-1))
    synthetic=[]
    for sid in dict.fromkeys(choices):
        result=synthetic_pcm(sid);synthetic.append(result)
        print(json.dumps({'synthetic_recording':sid,'passed':result['passed'],'error':result.get('error')}),flush=True)
    report={'completed_at':time.time(),'read_only':True,'new_api_calls':0,'training_modified':False,
        'real_sources':len(real),'real_windows':sum(r['windows'] for r in real),
        'real_failures':[r for r in real if not r['passed']],
        'synthetic_recordings':len(synthetic),'synthetic_checks':synthetic,
        'limits':'Checks original provider text/times/speakers and exact source PCM composition, not independent recognition of every spoken word.'}
    (OUT/'source-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='synthetic_checks'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
