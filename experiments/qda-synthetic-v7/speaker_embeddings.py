"""Source-speaker diversity preprocessing, independent of TTS word alignment."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi,hf_hub_download
from nemo.collections.asr.models import EncDecSpeakerLabelModel

ROOT=Path('/data/work/asr-voice-diversity-v7')

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def main():
    repo='nvidia/speakerverification_en_titanet_large';info=HfApi().model_info(repo)
    filename=next(s.rfilename for s in info.siblings if s.rfilename.endswith('.nemo'))
    model_path=hf_hub_download(repo,filename,revision=info.sha,cache_dir=str(ROOT/'cache'))
    save(ROOT/'model.json',{'repository':repo,'revision':info.sha,'file':filename,
        'purpose':'Detect repeated source voices and select acoustically diverse clone references; no word alignment.'})
    model=EncDecSpeakerLabelModel.restore_from(model_path,map_location='cpu').eval().to('cuda')
    model.freeze()
    while True:
        for path in sorted((ROOT/'clips').glob('*.json')):
            row=json.loads(path.read_text())
            if row.get('status')!='reference_ready':continue
            out=ROOT/'embeddings'/(row['candidate_id']+'.json')
            if out.exists():continue
            clip=ROOT/'clips'/(row['candidate_id']+'.wav')
            if not clip.exists():continue
            try:
                if hashlib.sha256(clip.read_bytes()).hexdigest()!=row['clip_sha256']:raise ValueError('Reference clip hash mismatch')
                with torch.inference_mode():v=model.get_embedding(str(clip)).flatten().float().cpu().numpy()
                v=v/np.linalg.norm(v)
                save(out,{'candidate_id':row['candidate_id'],'clip_sha256':row['clip_sha256'],
                          'embedding':v.tolist(),'model_revision':info.sha})
            except Exception as e:save(ROOT/'errors'/(row['candidate_id']+'.json'),{'error':str(e)})
        count=len(list((ROOT/'embeddings').glob('*.json')))
        save(ROOT/'progress.json',{'embeddings':count,'at':time.time()})
        time.sleep(5)

if __name__=='__main__':main()
