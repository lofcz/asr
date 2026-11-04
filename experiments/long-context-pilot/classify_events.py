"""AudioSet evidence for non-speech review; model scores are not verified labels."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import json,pathlib,time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification

BASE=pathlib.Path(__file__).resolve().parent
MODEL='MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION='f826b80d28226b62986cc218e5cec390b1096902'
GROUPS={
    '[Music]':['Music'],
    '[Human Sounds]':['Laughter','Cough','Sneeze','Breathing','Sigh','Gasp','Throat clearing','Hiccup','Crying, sobbing'],
    '[Noise]':['Noise','Static','White noise','Pink noise','Hiss','Distortion'],
    '[Environmental Sounds]':['Wind','Rain','Traffic noise, roadway noise','Vehicle','Bird','Dog','Door','Water'],
    '[Silence]':['Silence'],
}


def main():
    torch.set_num_threads(1)
    extractor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForAudioClassification.from_pretrained(MODEL,revision=REVISION,
                                                         use_safetensors=True).to('cuda').eval()
    labels=model.config.id2label;lookup={v:int(k) for k,v in labels.items()}
    for source in json.loads((BASE/'sample/sources.json').read_text()):
        started=time.monotonic()
        audit_path=BASE/'annotation-audit'/f"{source['id']}.json"
        audit=json.loads(audit_path.read_text())
        audio,sr=sf.read(BASE/'sample'/source['id']/'audio.flac',dtype='float32');assert sr==24000
        audio=resample_poly(audio,2,3).astype(np.float32)
        windows=[{'start':float(s),'end':min(float(s+10),audit['duration']),'kind':'full_timeline_scan'}
                 for s in range(0,int(np.ceil(audit['duration'])),10)]
        windows += [{**c,'kind':'non_speech_candidate'} for c in audit['candidate_non_speech']]
        windows += [{**c,'kind':'untranscribed_vad'} for c in audit['untranscribed_vad']]
        results=[]
        for i in range(0,len(windows),4):
            batch=windows[i:i+4]
            samples=[audio[int(w['start']*16000):min(int(w['end']*16000),int((w['start']+10)*16000))] for w in batch]
            inputs=extractor(samples,sampling_rate=16000,return_tensors='pt')
            with torch.inference_mode():scores=model(**{k:v.to('cuda') for k,v in inputs.items()}).logits.sigmoid().cpu().numpy()
            for w,p in zip(batch,scores):
                top=np.argsort(p)[-8:][::-1]
                grouped={tag:max((float(p[lookup[n]]) for n in names if n in lookup),default=0.) for tag,names in GROUPS.items()}
                speech=float(p[lookup['Speech']])
                proposed=max(grouped,key=grouped.get)
                # These are triage thresholds, deliberately not an auto-accept rule.
                if grouped[proposed]<.6 or speech>.2:proposed=None
                if proposed=='[Silence]' and w.get('candidate_tag')!='[Silence]':proposed=None
                results.append({**w,'audioset_top':[{'label':labels[int(j)],'score':round(float(p[j]),5)} for j in top],
                                'group_scores':grouped,'speech_score':speech,'proposed_tag':proposed,
                                'speech_and_music':speech>.3 and grouped['[Music]']>.3,
                                'status':'needs_acoustic_verification'})
            if (i//4)%25==0:print(json.dumps({'source_id':source['id'],'windows_done':min(i+4,len(windows)),'total':len(windows)}),flush=True)
        result={'source_id':source['id'],'model':MODEL,'revision':REVISION,
                'policy':'AudioSet class scores are uncalibrated evidence. No automatic acceptance. Background music does not replace speech labels.',
                'windows':results,'elapsed_seconds':time.monotonic()-started}
        (BASE/'annotation-audit'/f"{source['id']}.acoustic.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({'source_id':source['id'],'windows':len(results),'music_windows':sum(w['group_scores']['[Music]']>.6 for w in results if w['kind']=='full_timeline_scan'),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
