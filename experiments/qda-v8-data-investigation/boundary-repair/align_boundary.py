import json,sys,pathlib,time
import numpy as np,torch
sys.path.insert(0,'/data/work/asr-word-alignment')
from align_canary import load_model,align_segments
from align_existing import decode,atomic_json,sha
out=pathlib.Path(__file__).resolve().parent
audio=pathlib.Path('/data/work/asr-datasets/qda-496h-v1/data/audio/cs/bd1f434aac9d0865/17572540824a74d61b30b038.mp3')
torch.set_num_threads(3);torch.manual_seed(0)
wave=decode(audio)[1396*16000:1407*16000]
model=load_model('cpu')
tests=[
 {'Start':4.,'End':7.,'Speaker':'1','Content':'Tak to pro mě bylo.'},
 {'Start':4.,'End':9.,'Speaker':'1','Content':'Tak to pro mě bylo. Za hodinu, když vystříkne z penisu.'},
 {'Start':0.,'End':10.5,'Speaker':None,'Content':'Tak to pro mě jo, teda. Pro mě teda zase nebyla novinka, že rychlost spermatů je 45 km/h. Tak to pro mě bylo. Za hodinu, když vystříkne z penisu. To pro mě teda.'}
]
result=align_segments(tests,wave,model,'cpu',padding=0.)
result.update(audio_file=str(audio),audio_sha256=sha(audio),crop_source_start=1396.,source_text_unchanged=True)
atomic_json(out/'canary-boundary.json',result)
for s in result['segments']:
 print(json.dumps([(w['word'],None if w['start'] is None else round(w['start']+1396,3),None if w['end'] is None else round(w['end']+1396,3),w['issues']) for w in s['words']],ensure_ascii=False),flush=True)
