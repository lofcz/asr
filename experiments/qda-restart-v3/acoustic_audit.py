"""Full decode + clipping/dropout/energy + independent AST evidence, no generated labels."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import argparse
import json
import pathlib
import subprocess
import time
import numpy as np
import torch
import imageio_ffmpeg
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from contracts import atomic_json, sha

MODEL='MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION='f826b80d28226b62986cc218e5cec390b1096902'
CODE_SHA=sha(pathlib.Path(__file__))
# Same numerical audit, before only source-pool selection options were added.
COMPATIBLE_AUDITS={'99f65513914aee4fcefe39094261bc57d4793f8346da8a85543610458efd8925'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--world',type=int,default=2);p.add_argument('--all',action='store_true');p.add_argument('--original-budget-hours',type=float,default=0)
    p.add_argument('--development',action='store_true');p.add_argument('--long-only',action='store_true');a=p.parse_args()
    torch.cuda.set_device(a.rank);torch.set_num_threads(2)
    rows=json.loads((a.data/'sources.json').read_text())
    if a.development:rows=[r for r in rows if r['split']=='development']
    elif a.long_only:rows=[r for r in rows if r['split']=='train' and r['audio_duration']>=2700 and r['speaker_count'] in (2,3,4) and r['low_confidence_fraction']<=.03 and r['target_language_fraction']>=.95]
    elif a.original_budget_hours:
        pool=[r for r in rows if r['origin']=='qda-496h-v1' and r['split']!='confirmation' and r['speaker_count'] in (2,3,4) and r['low_confidence_fraction']<=.03 and r['target_language_fraction']>=.95]
        pool.sort(key=lambda r:(r['low_confidence_fraction'],r['record_id']))
        selected=[];hours={'cs':0.,'sk':0.}
        for r in pool:
            if hours[r['language']]>=a.original_budget_hours*({'cs':.65,'sk':.35}[r['language']]):continue
            selected.append(r);hours[r['language']]+=r['audio_duration']/3600
        rows=selected
    else:rows=[r for r in rows if a.all or r['origin']=='qda-50h-v2']
    extractor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForAudioClassification.from_pretrained(MODEL,revision=REVISION,use_safetensors=True,local_files_only=True).cuda().eval()
    lookup={v:int(k) for k,v in model.config.id2label.items()}
    groups={'music':['Music'],'noise':['Noise','Static','White noise','Pink noise','Hiss','Distortion'],
            'human':['Laughter','Cough','Crying, sobbing','Scream'],'speech':['Speech']}
    for row in rows[a.rank::a.world]:
        dest=a.data/'acoustic'/(row['record_id']+'.json')
        if dest.exists():
            old=json.loads(dest.read_text())
            if old['audio_sha256']==row['audio_sha256'] and old['audit_code_sha256'] in COMPATIBLE_AUDITS|{CODE_SHA}:continue
        start=time.monotonic()
        try:
            assert sha(row['audio_file'])==row['audio_sha256']
            proc=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',row['audio_file'],
                                 '-vn','-ar','16000','-ac','1','-f','f32le','-'],capture_output=True,check=True)
            assert not proc.stderr, 'decoder diagnostics: '+proc.stderr[:300].decode(errors='replace')
            wave=np.frombuffer(proc.stdout,dtype='<f4');duration=len(wave)/16000
            assert wave.size and np.isfinite(wave).all() and abs(duration-row['audio_duration'])<.1
            windows=[];reused=None
            existing=pathlib.Path(row['audio_file']).parents
            oldpath=pathlib.Path('/data/work/asr-datasets')/row['origin']/'acoustic'/(row['record_id']+'.json')
            if oldpath.exists():
                evidence=json.loads(oldpath.read_text())
                if evidence.get('audio_sha256')==row['audio_sha256'] and evidence.get('revision')==REVISION and not evidence.get('decoder_diagnostics'):
                    old=evidence['windows']
                    if len(old)==int(np.ceil(duration/10)) and all(abs(w['start']-i*10)<.01 and abs(w['end']-min((i+1)*10,duration))<.01 for i,w in enumerate(old)):
                        reused=old
            for offset in range(0,len(wave),160000*16):
                chunks=[wave[i:min(i+160000,len(wave))] for i in range(offset,min(offset+160000*16,len(wave)),160000)]
                # A final <25ms tail cannot form one AST frame. Pad only the classifier
                # input; PCM statistics and interval duration always use original samples.
                if reused is None:
                    inp=extractor([np.pad(c,(0,max(0,400-len(c)))) for c in chunks],sampling_rate=16000,return_tensors='pt')
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        scores=model(**{k:v.cuda() for k,v in inp.items()}).logits.sigmoid().float().cpu().numpy()
                else:scores=[None]*len(chunks)
                for j,(clip,score) in enumerate(zip(chunks,scores)):
                    s=(offset+j*160000)/16000;e=s+len(clip)/16000
                    rms=float(np.sqrt(np.mean(clip.astype(np.float64)**2)))
                    if reused is not None:
                        old=reused[offset//160000+j];group_scores={'music':old['group_scores']['[Music]'],'noise':old['group_scores']['[Noise]'],'human':old['group_scores']['[Human Sounds]'],'speech':old['speech_score']}
                    else:group_scores={k:max(float(score[lookup[n]]) for n in names if n in lookup) for k,names in groups.items()}
                    windows.append({'start':s,'end':e,'scores':group_scores,
                                    'rms_dbfs':20*np.log10(max(rms,1e-12)), 'clipped_fraction':float(np.mean(np.abs(clip)>=.999)),
                                    'zero_fraction':float(np.mean(clip==0)), 'peak':float(np.max(np.abs(clip)))})
            atomic_json(dest,{'status':'passed_decode','record_id':row['record_id'],'audio_sha256':row['audio_sha256'],
                              'decoded_seconds':duration,'windows':windows,'model':MODEL,'model_revision':REVISION,
                              'audit_code_sha256':CODE_SHA,'elapsed_seconds':time.monotonic()-start,'reused_classifier_evidence_sha256':sha(oldpath) if reused is not None else None,
                              'policy':'Classifier evidence only, never verified event labels or measured SNR.'})
            print(json.dumps({'event':'acoustic_checked','record':row['record_id'],'hours':duration/3600,'seconds':time.monotonic()-start}),flush=True)
        except Exception as error:
            atomic_json(a.data/'acoustic-errors'/(row['record_id']+'.json'),{'error':str(error),'time':time.time(),'audio_sha256':row['audio_sha256']})
            print(json.dumps({'event':'quarantined_audio','record':row['record_id'],'error':str(error)}),flush=True)


if __name__=='__main__':main()
