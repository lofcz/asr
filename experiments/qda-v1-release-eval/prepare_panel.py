"""Reuse verified frozen features; encode each additional held-out crop once."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import argparse,hashlib,json,math,pathlib,subprocess
from common import HERE,BASE,sha,atomic_json

def exact_decode(row):
    import numpy as np,imageio_ffmpeg
    assert sha(row['audio_file'])==row['audio_sha256']
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',row['audio_file'],'-vn','-ac','1','-ar','24000','-f','f32le','-'],capture_output=True,check=True)
    assert not result.stderr
    pcm=np.frombuffer(result.stdout,dtype='<f4');first=round(row['source_start']*24000);count=round(row['duration']*24000)
    wave=pcm[first:first+count].copy()
    assert first>=0 and count>0 and 0<=count-len(wave)<=3 and np.isfinite(wave).all()
    return wave

def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);args=p.parse_args()
    import torch,prepare
    from model_paths import model_paths
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    torch.set_num_threads(2);torch.cuda.set_device(args.rank)
    selection=json.loads((HERE/'selection.json').read_text());contract=json.loads((HERE/'feature-contract.json').read_text())
    contract_sha=sha(HERE/'feature-contract.json');plan={'cache_contract_sha256':contract_sha}
    (HERE/'cache').mkdir(exist_ok=True);(HERE/'ready').mkdir(exist_ok=True)
    prepare.HERE=HERE;prepare.decode=exact_decode
    path,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok);encoder=prepare.FeatureEncoder()
    for row in (selection['rows']+selection['long_rows'])[args.rank::2]:
        wid=row['window_id'];cache=pathlib.Path(row['cache_root'])
        if cache==BASE/'cache':
            ip=cache/(wid+'.input.pt');fm=json.loads((cache/(wid+'.features.json')).read_text())
            assert sha(ip)==row['input_sha256']==fm['input_sha256']
            assert sha(cache/(wid+'.features.npz'))==fm['sha256'] and fm['contract_sha256']==row['feature_contract_sha256']
            atomic_json(HERE/'ready'/(wid+'.json'),row);continue
        body,target,cm=prepare.tokenized_target(proc,row)
        row={**row,'target_tokens':len(target),'content_tokens':sum(cm),'target_json_sha256':hashlib.sha256(body.encode()).hexdigest(),
             'total_tokens':len(target)+math.ceil(row['duration']*7.5)+128,'feature_contract_sha256':contract_sha}
        assert row['total_tokens']<=65536
        encoder.write(row,plan)
        actual=prepare.checked_input(row,plan)
        atomic_json(HERE/'ready'/(wid+'.json'),actual)
        print(json.dumps({'event':'prepared','window_id':wid,'rank':args.rank}),flush=True)
if __name__=='__main__':main()
