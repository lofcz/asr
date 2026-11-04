"""A different ASR screens label disagreements; its outputs never overwrite Soniox."""
import os
import glob
import sys
os.environ['LD_LIBRARY_PATH']=':'.join(glob.glob('/data/work/asr-pilot/venv/lib/python3.12/site-packages/nvidia/*/lib'))+':'+os.environ.get('LD_LIBRARY_PATH','')
if __name__=='__main__' and not os.environ.get('ASR_AUDIT_LIBS_READY'):
    os.environ['ASR_AUDIT_LIBS_READY']='1'
    os.execv(sys.executable,[sys.executable,*sys.argv])
import argparse
import importlib.metadata
import json
import pathlib
import re
import sys
import time
import unicodedata
import subprocess
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio
from rapidfuzz.distance import Levenshtein
from contracts import atomic_json, sha
from events import TAGS

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/data/work/asr-evaluation/periodic/code')
from number_wer import tokens as number_tokens
REVISION='0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'


def words(text,language):
    return number_tokens(text,language)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=2);a=p.parse_args()
    policy=json.loads((HERE/'policy.json').read_text());selection=json.loads((a.data/'screening-selection.json').read_text())
    rows=selection['train']+selection['development']
    model=WhisperModel(str(HERE/'secondary-model'),device='cuda',device_index=a.rank,compute_type='float16',cpu_threads=2)
    pipeline=BatchedInferencePipeline(model)
    for row in rows[a.rank::a.world]:
        dest=a.data/'secondary'/(row['window_id']+'.json')
        if dest.exists():
            old=json.loads(dest.read_text())
            if old['label_sha256']==row['label_sha256'] and old['audio_sha256']==row['audio_sha256'] and old['code_sha256']==sha(pathlib.Path(__file__)):continue
        start=time.monotonic();assert sha(row['audio_file'])==row['audio_sha256']
        ffmpeg=glob.glob('/data/work/asr-pilot/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-*')[0]
        decoded=subprocess.run([ffmpeg,'-nostdin','-v','error','-threads','1','-i',row['audio_file'],'-ss',str(row['source_start']),'-t',str(row['duration']),
                                '-vn','-ar','16000','-ac','1','-f','f32le','-'],capture_output=True,check=True)
        if decoded.stderr:raise RuntimeError('Secondary decode diagnostics: '+decoded.stderr.decode(errors='replace')[:300])
        wave=np.frombuffer(decoded.stdout,dtype='<f4').copy()
        if not np.isfinite(wave).all() or abs(len(wave)/16000-row['duration'])>.03:raise RuntimeError('Secondary crop duration/nonfinite mismatch')
        # Word timestamps allow comparisons at Soniox segment boundaries without imposing its text.
        label=json.loads((a.data/row['label_path']).read_text());ws=[w for w in label['words'] if w['start']>=row['source_start'] and w['end']<=row['source_end']]
        boundaries=[0.]+[(x['end']+y['start'])/2-row['source_start'] for x,y in zip(ws,ws[1:]) if y['start']>=x['end']]+[row['duration']]
        clips=[];cursor=0.
        while cursor<row['duration']-.001:
            possible=[b for b in boundaries if cursor+.001<b<=min(cursor+30,row['duration'])]
            if not possible:raise RuntimeError('No safe secondary ASR boundary under 30 seconds')
            end=max(possible);clips.append({'start':cursor,'end':end});cursor=end
        segments,info=pipeline.transcribe(wave,language=row['language'],beam_size=5,batch_size=8,clip_timestamps=clips,
                                         word_timestamps=True,vad_filter=False,condition_on_previous_text=False,
                                         temperature=0.,no_speech_threshold=.6,compression_ratio_threshold=2.4)
        hypothesis=[dict(start=s.start,end=s.end,text=s.text,words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability) for w in (s.words or [])]) for s in segments]
        speech_target=[s for s in row['target'] if s['Content'] not in TAGS]
        reference=' '.join(s['Content'] for s in speech_target);hyp=''.join(s['text'] for s in hypothesis)
        refwords=words(reference,row['language']);hypwords=words(hyp,row['language'])
        errors=Levenshtein.distance(refwords,hypwords);wer=errors/max(1,len(refwords))
        # Group reference turns into ~30s contiguous spans; assign hypothesis words by midpoint.
        groups=[];current=[]
        for s in speech_target:
            if current and s['End']-current[0]['Start']>30:
                groups.append(current);current=[]
            current.append(s)
        if current:groups.append(current)
        timed=[w for s in hypothesis for w in s['words']];checks=[]
        for i,g in enumerate(groups):
            left=0 if i==0 else (groups[i-1][-1]['End']+g[0]['Start'])/2
            right=row['duration'] if i==len(groups)-1 else (g[-1]['End']+groups[i+1][0]['Start'])/2
            r=words(''.join(s['Content'] for s in g),row['language']);h=words(''.join(w['word'] for w in timed if left<=(w['start']+w['end'])/2<right),row['language'])
            e=Levenshtein.distance(r,h);checks.append({'start':left,'end':right,'reference_words':len(r),'errors':e,'wer':e/max(1,len(r))})
        bad=sum(x['wer']>policy['secondary_bad_segment_wer'] for x in checks)/max(1,len(checks))
        passed=wer<=policy['maximum_secondary_asr_wer'] and bad<=policy['maximum_secondary_bad_segment_fraction']
        atomic_json(dest,{'status':'passed' if passed else 'quarantined','window_id':row['window_id'],'record_id':row['record_id'],
                          'audio_sha256':row['audio_sha256'],'label_sha256':row['label_sha256'],'source_start':row['source_start'],'source_end':row['source_end'],
                          'model':'dropbox-dash/faster-whisper-large-v3-turbo','model_revision':REVISION,
                          'reference_words':len(refwords),'errors':errors,'number_normalized_disagreement_wer':wer,
                          'bad_segment_fraction':bad,'segments':checks,'hypothesis':hypothesis,'elapsed_seconds':time.monotonic()-start,
                          'policy_sha256':sha(HERE/'policy.json'),'code_sha256':sha(pathlib.Path(__file__)),
                          'package_versions':{n:importlib.metadata.version(n) for n in ('faster-whisper','ctranslate2')},
                          'quality':'Independent machine disagreement signal, not measured reference accuracy or human gold.'})
        print(json.dumps({'window':row['window_id'],'status':'passed' if passed else 'quarantined','disagreement_wer':wer,'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
