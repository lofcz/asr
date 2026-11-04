"""Exact target tokenization and frozen features for the screened Soniox restart."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import argparse
import collections
import gc
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from contracts import atomic_json, sha, target_with_spans, content_token_mask
from gates import check_data, demand, development_reference

HERE=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path('/data/work/asr-datasets/qda-clean-v3')
sys.path[:0]=['/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice']


def require_old_jobs_stopped():
    for name in ('qda496-training.service','qda496-periodic-eval.service','qda50-v2-experiments.service','qda250-unseen-eval.service'):
        status=subprocess.run(['systemctl','--user','is-active',name],capture_output=True,text=True).stdout.strip()
        demand(status in ('inactive','failed','unknown'),'Conflicting job is active: '+name)
    demand(pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16/paused-by-user.json').exists(),'Old training pause marker was removed')


def decode(row):
    import numpy as np
    import imageio_ffmpeg
    demand(sha(row['audio_file'])==row['audio_sha256'],'Audio identity changed')
    p=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',row['audio_file'],
                      '-ss',str(row['source_start']),'-t',str(row['duration']),'-vn','-ac','1','-ar','24000','-f','f32le','-'],capture_output=True,check=True)
    demand(not p.stderr,'Decoder reported an audio error')
    wave=np.frombuffer(p.stdout,dtype='<f4').copy()
    demand(np.isfinite(wave).all() and abs(len(wave)/24000-row['duration'])<.03,'Crop duration/nonfinite mismatch')
    return wave


def arrange(rows):
    pool=sorted(rows,key=lambda r:(r['total_tokens'],r['window_id']));pairs=[pool[i:i+2] for i in range(0,len(pool),2)]
    ordered=[];counts=collections.Counter();lang=collections.Counter();seen=0
    while pairs:
        def score(pair):
            d=counts.copy();l=lang.copy();n=seen
            for r in pair:d[r['duration_class']]+=r['duration'];l[r['language']]+=r['duration'];n+=r['duration']
            return sum((d[k]/n-v)**2/v for k,v in {'short':.25,'medium':.5,'long':.25}.items())+(l['sk']/n-.3)**2/.3
        index=min(range(len(pairs)),key=lambda i:(score(pairs[i]),i));pair=pairs.pop(index);ordered.append(pair)
        for r in pair:counts[r['duration_class']]+=r['duration'];lang[r['language']]+=r['duration'];seen+=r['duration']
    prefix=[r for p in ordered[:max(4,len(ordered)//4)] for r in p]
    demand(any(r['duration']>=2700 for r in prefix),'Long sessions absent from early training')
    return ordered


def inputs():
    import torch
    from model_paths import model_paths, MODEL_REVISION, TOKENIZER_REVISION
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    admission=check_data(DATA,reviewed=True)
    selection=json.loads((DATA/'screening-selection.json').read_text());cache=HERE/'cache';cache.mkdir(exist_ok=True)
    contract={'model_revision':MODEL_REVISION,'tokenizer_revision':TOKENIZER_REVISION,'dataset_admission_sha256':sha(DATA/'admission.json'),
              'prepare_code_sha256':sha(pathlib.Path(__file__)),'target_code_sha256':sha(HERE/'targets.py'),'contract_code_sha256':sha(HERE/'contracts.py'),
              'feature_dtype':'bfloat16','feature_seed':'window id first 8 hex digits','context':[],'max_tokens':65536}
    if (cache/'contract.json').exists():demand(json.loads((cache/'contract.json').read_text())==contract,'Cache contract changed: use a fresh cache directory')
    else:atomic_json(cache/'contract.json',contract)
    path,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok);rows=[]
    for source in selection['train']+selection['development']:
        row=dict(source);wid=row['window_id']
        row['reference_quality']='Soniox pseudo-reference'
        if row['split']=='development':row['target'],row['reference_quality'],_=development_reference(DATA,row)
        wave=decode(row);enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
        sample={'segments':[dict(start=s['Start'],end=s['End'],speaker=s.get('Speaker'),text=s['Content']) for s in row['target']]}
        body,spans=target_with_spans(sample)
        demand(json.loads(body)==row['target'],'Serialized targets differ from approved labels')
        template=proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=False,add_generation_prompt=False)
        target=proc.tokenizer(template,add_special_tokens=False,return_offsets_mapping=True);offset=template.index(body)
        demand(target['input_ids']==proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=True,add_generation_prompt=False),'Tokenizer/template mismatch')
        cm=content_token_mask(target['offset_mapping'],[(a+offset,b+offset) for a,b in spans])
        n=len(enc['input_ids']);ids=enc['input_ids']+target['input_ids'];mask=enc['acoustic_input_mask']+[False]*len(target['input_ids'])
        demand(len(ids)<=65536 and sum(cm)>0,'Token overflow or empty content; no truncation allowed')
        payload={'ids':torch.tensor(ids),'mask':torch.tensor(mask,dtype=torch.bool),'labels':torch.tensor([-100]*n+target['input_ids']),
                 'content_mask':torch.tensor([False]*n+cm,dtype=torch.bool)}
        dest=cache/(wid+'.input.pt');tmp=dest.with_suffix('.building');torch.save(payload,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        tmp.replace(dest)
        row.update(total_tokens=len(ids),target_tokens=len(target['input_ids']),content_tokens=sum(cm),speech_features=sum(mask),input_sha256=sha(dest),target_json_sha256=hashlib.sha256(body.encode()).hexdigest())
        atomic_json(cache/(wid+'.input.json'),row);rows.append(row)
        print(json.dumps({'event':'tokenized','window':wid,'tokens':len(ids)}),flush=True)
    pairs=arrange([r for r in rows if r['split']=='train']);dev=[r for r in rows if r['split']=='development']
    small=[next(r for r in dev if r['language']==l) for l in ('cs','sk')]
    plan={'version':3,'seed':20261023,'initialization':'vanilla; zero B; fresh optimizer','dataset':str(DATA),
          'dataset_admission_sha256':sha(DATA/'admission.json'),'cache_contract_sha256':sha(cache/'contract.json'),
          'probe_pairs':pairs,'calibration_pairs':pairs[:6],'validation':dev,'validation_small':small,
          'train_hours':sum(r['duration'] for p in pairs for r in p)/3600,'cached_windows_initial':[r['window_id'] for r in rows],
          'probe_target_tokens':sum(r['target_tokens'] for p in pairs for r in p),'model_revision':MODEL_REVISION,
          'long_stress':[max((r for p in pairs for r in p if r['language']==l),key=lambda r:r['duration']) for l in ('cs','sk')],
          'fixed_markers':{'root':'/data/work/asr-evaluation/production-reference-audit-v2','languages':['cs','sk'],
                           'role':'Required longitudinal markers, separate from independent held-out interviews'}}
    atomic_json(HERE/'plan.json',plan)


def features(rank,world):
    import torch,numpy as np
    from model_paths import model_paths
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    require_old_jobs_stopped();plan=json.loads((HERE/'plan.json').read_text());cache=HERE/'cache'
    demand(sha(DATA/'admission.json')==plan['dataset_admission_sha256'],'Admission changed')
    torch.cuda.set_device(rank);torch.set_num_threads(2);path,tok=model_paths()
    proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
    model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    model.model.language_model=None;model.lm_head=None;model.requires_grad_(False);gc.collect();model.cuda()
    rows=[r for p in plan['probe_pairs'] for r in p]+plan['validation']
    for row in rows[rank::world]:
        wid=row['window_id'];dest=cache/(wid+'.features.npy');receipt=cache/(wid+'.features.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            demand(old['contract_sha256']==plan['cache_contract_sha256'] and old['input_sha256']==row['input_sha256'] and old['sha256']==sha(dest),'Stale feature cache');continue
        wave=decode(row);enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
        with torch.random.fork_rng(devices=[rank]),torch.inference_mode():
            torch.manual_seed(int(wid[:8],16));f=model.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).cuda()).squeeze(0).bfloat16().cpu()
        demand(torch.isfinite(f).all() and f.shape[0]==row['speech_features'],'Speech feature mismatch')
        tmp=dest.with_suffix('.building')
        with tmp.open('wb') as h:np.save(h,f.view(torch.uint16).numpy(),allow_pickle=False);h.flush();os.fsync(h.fileno())
        tmp.replace(dest);atomic_json(receipt,{'sha256':sha(dest),'input_sha256':row['input_sha256'],'contract_sha256':plan['cache_contract_sha256'],'shape':list(f.shape)})
        print(json.dumps({'event':'features','rank':rank,'window':wid}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inputs','features']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=2);a=p.parse_args()
    if a.command=='inputs':inputs()
    else:features(a.rank,a.world)
