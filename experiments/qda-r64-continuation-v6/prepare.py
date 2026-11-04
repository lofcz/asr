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
DATA=pathlib.Path('/data/work/asr-datasets/qda-unseen-v6')
sys.path[:0]=['/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice']


def require_old_jobs_stopped():
    for name in ('qda496-training.service','qda496-periodic-eval.service','qda50-v2-experiments.service','qda250-unseen-eval.service','qda-v3-experiments.service','qda-lr-calibration.service','qda-v4-training.service'):
        status=subprocess.run(['systemctl','--user','is-active',name],capture_output=True,text=True).stdout.strip()
        demand(status in ('inactive','failed','unknown'),'Conflicting job is active: '+name)
    demand(pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16/paused-by-user.json').exists(),'Old training pause marker was removed')


def pcm_crop(wave,start,duration,sample_rate=24000):
    """Crop the decoded sample timeline, independent of container timestamps."""
    first=round(start*sample_rate);count=round(duration*sample_rate)
    demand(first>=0 and count>0,'Invalid PCM crop interval')
    result=wave[first:first+count]
    demand(abs(len(result)-count)<=3,'Decoded source does not contain the requested crop')
    return result.copy()


def decode(row):
    import numpy as np
    import imageio_ffmpeg
    demand(sha(row['audio_file'])==row['audio_sha256'],'Audio identity changed')
    p=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',row['audio_file'],
                      '-ss',str(row['source_start']),'-t',str(row['duration']),'-vn','-ac','1','-ar','24000','-f','f32le','-'],capture_output=True,check=True)
    demand(not p.stderr,'Decoder reported an audio error')
    wave=np.frombuffer(p.stdout,dtype='<f4').copy()
    demand(np.isfinite(wave).all(),'Nonfinite decoded crop')
    if abs(len(wave)/24000-row['duration'])>=.03:
        timestamp_seconds=len(wave)/24000
        # Some AAC/M4A timestamp discontinuities make output-side -ss/-t
        # return the wrong sample count. Soniox/VAD use the decoded timeline.
        full=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',row['audio_file'],
                             '-vn','-ac','1','-ar','24000','-f','f32le','-'],capture_output=True,check=True)
        demand(not full.stderr,'Decoder reported an audio error')
        source=np.frombuffer(full.stdout,dtype='<f4')
        demand(np.isfinite(source).all() and sha(row['audio_file'])==row['audio_sha256'],'Invalid decoded source or changed audio identity')
        wave=pcm_crop(source,row['source_start'],row['duration'])
        atomic_json(HERE/'decode-repairs'/(row['window_id']+'.json'),{
            'window_id':row['window_id'],'audio_sha256':row['audio_sha256'],'source_start':row['source_start'],
            'expected_seconds':row['duration'],'timestamp_crop_seconds':timestamp_seconds,'decoded_source_seconds':len(source)/24000,
            'exact_crop_samples':len(wave),'method':'full decoded PCM, integer sample-index slice; no padding or label changes',
            'prepare_sha256':sha(pathlib.Path(__file__))})
        print(json.dumps({'event':'timestamp_crop_recovered','window':row['window_id'],'previous_seconds':timestamp_seconds,'exact_seconds':len(wave)/24000}),flush=True)
    demand(np.isfinite(wave).all() and abs(len(wave)/24000-row['duration'])<.03,'Crop duration/nonfinite mismatch')
    return wave


def arrange(rows):
    # Neighbouring token lengths balance both GPUs; shuffle whole pairs so the
    # actual remaining length/language mix is represented throughout the pass.
    # Do not force the old full-corpus length fractions onto a different remainder.
    import random
    pool=sorted(rows,key=lambda r:(r['total_tokens'],r['window_id']))
    pairs=[pool[i:i+2] for i in range(0,len(pool),2)]
    random.Random(20261023).shuffle(pairs)
    return pairs


def tokenized_target(proc,row):
    sample={'segments':[dict(start=s['Start'],end=s['End'],speaker=s.get('Speaker'),text=s['Content']) for s in row['target']]}
    body,spans=target_with_spans(sample)
    demand(json.loads(body)==row['target'],'Serialized targets differ from approved labels')
    template=proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=False,add_generation_prompt=False)
    target=proc.tokenizer(template,add_special_tokens=False,return_offsets_mapping=True);offset=template.index(body)
    demand(target['input_ids']==proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=True,add_generation_prompt=False),'Tokenizer/template mismatch')
    cm=content_token_mask(target['offset_mapping'],[(a+offset,b+offset) for a,b in spans])
    return body,target['input_ids'],cm


def checked_input(row,plan):
    cache=HERE/'cache';saved=json.loads((cache/(row['window_id']+'.input.json')).read_text())
    demand(all(saved[k]==row[k] for k in ('window_id','audio_sha256','label_sha256','source_start','source_end','target_tokens','target_json_sha256')),'Cached input differs from immutable row')
    demand(saved['input_contract_sha256']==plan['cache_contract_sha256'],'Input preparation contract changed')
    demand(sha(cache/(row['window_id']+'.input.pt'))==saved['input_sha256'],'Cached input checksum mismatch')
    return saved


def materialize_input(row,proc,contract,return_audio=False):
    import torch,math
    cache=HERE/'cache';wid=row['window_id'];receipt=cache/(wid+'.input.json');dest=cache/(wid+'.input.pt')
    enc=None
    if receipt.exists():
        saved=checked_input(row,{'cache_contract_sha256':contract})
        if return_audio:
            wave=decode(row);enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
        return saved,enc
    body,target,cm=tokenized_target(proc,row)
    demand(len(target)==row['target_tokens'] and hashlib.sha256(body.encode()).hexdigest()==row['target_json_sha256'],'Token plan differs from exact target')
    if row['split']=='development':
        old_root=pathlib.Path('/data/work/asr-experiments/qda-restart-v3')
        old_plan=json.loads((old_root/'plan.json').read_text())
        old=next(r for r in old_plan['validation'] if r['window_id']==wid)
        demand(all(old[k]==row[k] for k in ('audio_sha256','label_sha256','source_start','source_end','target')),'Held-out input provenance differs')
        original=old_root/'cache'/(wid+'.input.pt');demand(sha(original)==old['input_sha256'],'Old held-out input changed')
        payload=torch.load(original,weights_only=True)
        n=len(payload['ids'])-len(target)
        demand(torch.equal(payload['ids'][n:],torch.tensor(target)) and torch.equal(payload['content_mask'][n:],torch.tensor(cm,dtype=torch.bool)),'Held-out target tokens differ')
    else:
        wave=decode(row);enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
        n=len(enc['input_ids'])
        payload={'ids':torch.tensor(enc['input_ids']+target),'mask':torch.tensor(enc['acoustic_input_mask']+[False]*len(target),dtype=torch.bool),
                 'labels':torch.tensor([-100]*n+target),'content_mask':torch.tensor([False]*n+cm,dtype=torch.bool)}
    demand(len(payload['ids'])<=65536 and sum(cm)>0,'Exact token overflow or empty content; no truncation allowed')
    tmp=dest.with_suffix('.building');torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(dest)
    saved={**row,'total_tokens':len(payload['ids']),'speech_features':int(payload['mask'].sum()),
           'input_sha256':sha(dest),'input_contract_sha256':contract,'input_mode':'exact-decoded-or-verified-reuse',
           'prepare_runtime_sha256':sha(pathlib.Path(__file__))}
    atomic_json(receipt,saved)
    return saved,enc if return_audio else None


def inputs(rank=0,world=1):
    import math
    from model_paths import model_paths, MODEL_REVISION, TOKENIZER_REVISION
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    check_data(DATA,reviewed=True)
    selection=json.loads((DATA/'screening-selection.json').read_text());cache=HERE/'cache';cache.mkdir(exist_ok=True)
    contract={'model_revision':MODEL_REVISION,'tokenizer_revision':TOKENIZER_REVISION,'dataset_admission_sha256':sha(DATA/'admission.json'),
              'prepare_code_sha256':sha(pathlib.Path(__file__)),'target_code_sha256':sha(HERE/'targets.py'),'contract_code_sha256':sha(HERE/'contracts.py'),
              'feature_dtype':'bfloat16-lossless-npz','feature_seed':'window id first 8 hex digits','context':[],'max_tokens':65536,
              'input_mode':'exact-target-plan; decode and validate each waveform before its first update'}
    if (cache/'contract.json').exists():demand(json.loads((cache/'contract.json').read_text())==contract,'Cache contract changed: use a fresh cache directory')
    else:atomic_json(cache/'contract.json',contract)
    identity=sha(cache/'contract.json')
    path,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
    for source in (selection['train']+selection['development'])[rank::world]:
        row=dict(source);wid=row['window_id'];row['reference_quality']='Soniox pseudo-reference'
        if row['split']=='development':row['target'],row['reference_quality'],_=development_reference(DATA,row)
        body,target,cm=tokenized_target(proc,row)
        row.update(total_tokens=len(target)+math.ceil(row['duration']*proc.target_sample_rate/proc.speech_tok_compress_ratio)+128,
                   target_tokens=len(target),content_tokens=sum(cm),input_sha256=None,
                   target_json_sha256=hashlib.sha256(body.encode()).hexdigest(),input_mode='verified-on-demand')
        demand(row['total_tokens']<=65536 and row['content_tokens']>0,'Planned token overflow or empty content')
        if row['split']=='development':row,_=materialize_input(row,proc,identity)
        atomic_json(cache/(wid+'.input-plan.json'),row)
        print(json.dumps({'event':'tokenized-target','window':wid,'tokens':row['total_tokens']}),flush=True)
    if world==1:finalize_inputs()


def finalize_inputs():
    from model_paths import MODEL_REVISION
    selection=json.loads((DATA/'screening-selection.json').read_text());cache=HERE/'cache'
    rows=[]
    for source in selection['train']+selection['development']:
        row=json.loads((cache/(source['window_id']+'.input-plan.json')).read_text())
        demand(row['audio_sha256']==source['audio_sha256'] and row['label_sha256']==source['label_sha256'] and
               row['target']==source['target'],'Input plan differs from selected data')
        rows.append(row)
    pairs=arrange([r for r in rows if r['split']=='train']);dev=[r for r in rows if r['split']=='development']
    small=[next(r for r in dev if r['language']==l) for l in ('cs','sk')]
    plan={'version':6,'seed':20261023,'initialization':'parent checkpoint-001314; restored Adam moments; new measured LR schedule','dataset':str(DATA),
          'dataset_admission_sha256':sha(DATA/'admission.json'),'cache_contract_sha256':sha(cache/'contract.json'),
          'probe_pairs':pairs,'calibration_pairs':pairs[:6],'validation':dev,'validation_small':small,
          'train_hours':sum(r['duration'] for p in pairs for r in p)/3600,'cached_windows_initial':[r['window_id'] for r in rows],
          'probe_target_tokens':sum(r['target_tokens'] for p in pairs for r in p),'model_revision':MODEL_REVISION,
          'long_stress':[max((r for p in pairs for r in p if r['language']==l),key=lambda r:r['duration']) for l in ('cs','sk')],
          'fixed_markers':{'root':'/data/work/asr-evaluation/production-reference-audit-v2','languages':['cs','sk'],
                           'role':'Required longitudinal markers, separate from independent held-out interviews'}}
    atomic_json(HERE/'plan.json',plan)
    if (HERE/'run-spec.json').exists():
        from schedule import write_preview
        write_preview(HERE)


class FeatureEncoder:
    """Frozen encoder kept on CPU between steps; each completed feature is durable."""
    def __init__(self):self.model=None;self.proc=None

    def write(self,row,plan):
        import fcntl
        with (HERE/'cache'/(row['window_id']+'.features.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._write(row,plan)

    def _write(self,row,plan):
        import torch,numpy as np,shutil
        from model_paths import model_paths
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        cache=HERE/'cache';wid=row['window_id'];dest=cache/(wid+'.features.npz');receipt=cache/(wid+'.features.json')
        if receipt.exists():row=checked_input(row,plan)
        if receipt.exists():
            old=json.loads(receipt.read_text())
            demand(old['contract_sha256']==plan['cache_contract_sha256'] and old['input_sha256']==row['input_sha256'] and old['sha256']==sha(dest),'Stale feature cache')
            return
        demand(shutil.disk_usage(cache).free>8*2**30,'Less than 8 GiB free; preserve completed data and stop')
        old_root=pathlib.Path('/data/work/asr-experiments/qda-restart-v3')
        if row['split']=='development':
            row=checked_input(row,plan)
            old_plan=json.loads((old_root/'plan.json').read_text())
            old=next(r for r in old_plan['validation'] if r['window_id']==wid)
            old_fp=old_root/'cache'/(wid+'.features.npy');old_meta=json.loads((old_root/'cache'/(wid+'.features.json')).read_text())
            demand(all(row[k]==old[k] for k in ('audio_sha256','source_start','source_end','target')) and plan['model_revision']==old_plan['model_revision'],'Held-out feature provenance differs')
            demand(sha(old_fp)==old_meta['sha256'],'Old held-out features changed')
            a=torch.load(cache/(wid+'.input.pt'),weights_only=True);b=torch.load(old_root/'cache'/(wid+'.input.pt'),weights_only=True)
            demand(a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'Held-out input tensors changed')
            f=np.load(old_fp,allow_pickle=False)
            extra={'reused_from':str(old_fp),'reused_sha256':old_meta['sha256']}
        else:
            if self.model is None:
                path,tok=model_paths()
                self.proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
                with torch.random.fork_rng(devices=[]):
                    self.model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
                self.model.model.language_model=None;self.model.lm_head=None;self.model.requires_grad_(False);gc.collect()
            row,enc=materialize_input(row,self.proc,plan['cache_contract_sha256'],return_audio=True)
            inputs=torch.load(cache/(wid+'.input.pt'),map_location='cpu',weights_only=True)
            n=len(enc['input_ids'])
            demand(torch.equal(inputs['ids'][:n],torch.tensor(enc['input_ids'])) and
                   torch.equal(inputs['mask'][:n],torch.tensor(enc['acoustic_input_mask'],dtype=torch.bool)),
                   'Real decoded audio prefix differs from the immutable training input')
            device=torch.cuda.current_device();self.model.cuda()
            try:
                with torch.random.fork_rng(devices=[device]),torch.inference_mode():
                    torch.manual_seed(int(wid[:8],16));features=self.model.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).cuda()).squeeze(0).bfloat16().cpu()
                demand(torch.isfinite(features).all() and features.shape[0]==row['speech_features'],'Speech feature mismatch')
                f=features.view(torch.uint16).numpy()
            finally:
                self.model.cpu();torch.cuda.empty_cache()
            extra={'computed_on_demand':True}
        tmp=dest.with_suffix('.building')
        with tmp.open('wb') as h:np.savez_compressed(h,features=f);h.flush();os.fsync(h.fileno())
        tmp.replace(dest)
        atomic_json(receipt,{'sha256':sha(dest),'input_sha256':row['input_sha256'],
                   'contract_sha256':plan['cache_contract_sha256'],'shape':list(f.shape),**extra})
        print(json.dumps({'event':'features','window':wid}),flush=True)


def features(rank,world,validation_only=False):
    import torch
    require_old_jobs_stopped();torch.cuda.set_device(rank);torch.set_num_threads(2)
    plan=json.loads((HERE/'plan.json').read_text())
    demand(sha(DATA/'admission.json')==plan['dataset_admission_sha256'],'Admission changed')
    rows=plan['validation'] if validation_only else [r for p in plan['probe_pairs'] for r in p]+plan['validation']
    encoder=FeatureEncoder()
    for row in rows[rank::world]:encoder.write(row,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inputs','features','plan']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);p.add_argument('--validation-only',action='store_true');a=p.parse_args()
    if a.command=='inputs':inputs(a.rank,a.world)
    elif a.command=='plan':finalize_inputs()
    else:features(a.rank,a.world,a.validation_only)
