"""Resumable dual-GPU QLoRA on immutable source-offset pseudo-labels."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import sys,pathlib,json,hashlib,random,time,datetime,gc,math,shutil
ROOT=pathlib.Path('/data/work/asr-datasets/qda-496h-v1')
PILOT=pathlib.Path('/data/work/asr-pilot/long-context-pilot')
OUT=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
sys.path[:0]=[str(ROOT/'code'),str(PILOT)]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from qda_release_dataset import ReleaseDataset,target_json
from train import load_model
from optimized_kernels import loss_for,AdaptiveOffload
from concurrent.futures import ThreadPoolExecutor
import threading
from model_paths import model_paths,MODEL_REVISION
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
sys.path.insert(0,'/data/work/asr-evaluation/periodic/code')
from policy import due as evaluation_due
EVAL=pathlib.Path('/data/work/asr-evaluation/periodic')
SEED=20261022

def atomic_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n')
 with tmp.open('rb') as f:os.fsync(f.fileno())
 os.replace(tmp,path)

def make_plan(datasets):
 override=OUT/'plan-override.json'
 if override.exists():
  plan=json.loads(override.read_text())
  for r in plan:
   actual=datasets[r['bucket']].rows[r['index']]
   assert actual['window_id']==r['window_id'] and actual['record_id']==r['record_id']
  return plan
 ids=sorted({r['record_id'] for r in datasets[600].rows});random.Random(SEED).shuffle(ids)
 assignment={rid:600 if i<len(ids)*.2 else 1800 if i<len(ids)*.5 else 3600 for i,rid in enumerate(ids)}
 plan=[]
 for duration in (600,1800,3600):
  rows=[{'bucket':duration,'index':i,'window_id':r['window_id'],'record_id':r['record_id'],'seconds':r['duration']} for i,r in enumerate(datasets[duration].rows) if assignment[r['record_id']]==duration]
  random.Random(SEED+duration).shuffle(rows);plan.extend(rows)
 assert set(x['record_id'] for x in plan)==set(ids)
 assert len({x['window_id'] for x in plan})==len(plan)
 return plan

class Objective(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model;self.offload=AdaptiveOffload()
 def forward(self,item):return loss_for(self.model,item,self.offload)

def main():
 local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
 dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=datetime.timedelta(minutes=30))
 rank=dist.get_rank();world=dist.get_world_size();assert world==2
 torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
 verified=json.loads((ROOT/'verification.json').read_text());assert verified['status']=='verified'
 assert hashlib.sha256((ROOT/'training/catalog.sqlite3').read_bytes()).hexdigest()==verified['training_catalog_sha256']
 datasets={d:ReleaseDataset(ROOT,'train',d) for d in (600,1800,3600)}
 validation=ReleaseDataset(ROOT,'validation',600)
 plan=make_plan(datasets);plan_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();steps=math.ceil(len(plan)/world)
 OUT.mkdir(parents=True,exist_ok=True)
 def log(event):
  if rank==0:
   event={'time':time.time(),**event};print(json.dumps(event),flush=True)
   with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
 config={'model_revision':MODEL_REVISION,'training_manifest_sha256':verified['training_manifest_sha256'],'plan_sha256':plan_hash,'seed':SEED,'world_size':world,'optimizer':'AdamW','lr':1e-5,'weight_decay':.01,'lora_rank':16,'lora_alpha':32,'dropout':.05,'max_tokens':65536,'views':[600,1800,3600],'source_fractions':[.2,.3,.5],'samples':len(plan),'steps':steps,'planned_audio_hours':sum(r['seconds'] for r in plan)/3600,'profile':'speech-diarization-pseudo-v1','encoders_and_connectors':'frozen; online features, no overlapping cache duplication','checkpoint_every':25,'validation_every':25,'validation_note':'Fixed held-out pseudo-label loss; test set unused'}
 if rank==0:
  if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==config
  else:atomic_json(OUT/'config.json',config);atomic_json(OUT/'plan.json',plan)
 dist.barrier();log({'event':'loading','config':config})
 path,tok=model_paths();processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
 encoder=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
 encoder.model.language_model=None;encoder.lm_head=None;encoder.requires_grad_(False);gc.collect()
 model=load_model('qlora',16,2048);model.train();trainable=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(trainable,lr=1e-5,weight_decay=.01)
 start=0
 if (OUT/'latest.json').exists():
  saved=OUT/json.loads((OUT/'latest.json').read_text())['checkpoint']
  state=torch.load(saved/'state.pt',map_location='cpu',weights_only=False);assert state['plan_sha256']==plan_hash
  from peft import set_peft_model_state_dict
  from safetensors.torch import load_file
  set_peft_model_state_dict(model,load_file(str(saved/'adapter_model.safetensors')))
  optimizer.load_state_dict(state['optimizer']);start=state['step']
  rng=state['rng'][rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy']);del state,rng
 else:torch.manual_seed(SEED+rank)
 objective=Objective(model);ddp=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True)
 processor_lock=threading.Lock()
 def prepare(ds,index):
  begin=time.monotonic();sample,wave=ds.decode(index)
  with processor_lock:
   enc=processor._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
   target=processor.tokenizer.apply_chat_template([{'role':'assistant','content':target_json(sample)}],tokenize=True,add_generation_prompt=False)
  ids=enc['input_ids']+target;mask=enc['acoustic_input_mask']+[0]*len(target);labels=[-100]*len(enc['input_ids'])+target
  if len(ids)>65536:raise ValueError(f"Token budget exceeded: {sample['window_id']} {len(ids)}; requires a reviewed smaller-window plan")
  return enc,ids,mask,labels,sample,len(target),time.monotonic()-begin
 def item(ds,index,prepared=None):
  enc,ids,mask,labels,sample,n,prep_seconds=prepare(ds,index) if prepared is None else prepared
  begin=time.monotonic()
  # Free encoder VRAM before the long-context language-model pass.
  encoder.to('cuda')
  with torch.random.fork_rng(devices=[local]),torch.inference_mode():
   torch.manual_seed(int(hashlib.sha256(sample['window_id'].encode()).hexdigest()[:8],16))
   features=encoder.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).to('cuda')).squeeze(0).to(torch.bfloat16)
  encoder.to('cpu');torch.cuda.empty_cache()
  assert features.shape[0]==sum(mask) and torch.isfinite(features).all()
  return (torch.tensor(ids,device='cuda').unsqueeze(0),torch.tensor(mask,device='cuda',dtype=torch.bool).unsqueeze(0),features,torch.tensor(labels,device='cuda').unsqueeze(0)),sample,n,{'cpu_prepare_seconds':prep_seconds,'encode_and_transfer_seconds':time.monotonic()-begin}
 # Stable small monitoring panel, separate shows on the two ranks.
 val_sources=sorted({r['record_id'] for r in validation.rows});random.Random(SEED).shuffle(val_sources)
 val_indices=[max((i for i,r in enumerate(validation.rows) if r['record_id']==rid),key=lambda i:validation.rows[i]['duration']) for rid in val_sources[:4]]
 def evaluate(step):
  model.eval();total=torch.zeros(2,device='cuda',dtype=torch.float64)
  for i in val_indices[rank::world]:
   batch,_,n,_=item(validation,i)
   with torch.no_grad():value=objective(batch)
   total+=torch.tensor([value.item()*n,n],device='cuda');del batch
  dist.all_reduce(total);model.train();log({'event':'validation','step':step,'loss':(total[0]/total[1]).item(),'target_tokens':int(total[1]),'windows':len(val_indices)})
 def save(step):
  rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state()};states=[None]*world;dist.all_gather_object(states,rng)
  if rank==0:
   name=f'checkpoint-{step:06d}';tmp=OUT/(name+'.building');tmp.mkdir(exist_ok=True)
   model.save_pretrained(tmp,safe_serialization=True)
   torch.save({'step':step,'optimizer':optimizer.state_dict(),'rng':states,'plan_sha256':plan_hash},tmp/'state.pt')
   for p in tmp.iterdir():
    if p.is_file():
     with p.open('rb') as f:os.fsync(f.fileno())
   os.replace(tmp,OUT/name);atomic_json(OUT/'latest.json',{'checkpoint':name,'step':step})
   for old in sorted(OUT.glob('checkpoint-[0-9]*'))[:-3]:
    if old.is_dir() and not old.name.endswith('.building'):shutil.rmtree(old)
   log({'event':'checkpoint','step':step,'path':str(OUT/name)})
  dist.barrier()
 if start==0:evaluate(0)
 lengths=json.loads(pathlib.Path('/data/work/asr-optimization/lengths.json').read_text())
 pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='audio-prepare')
 def submit(step):
  r=plan[min(step*world+rank,len(plan)-1)]
  return pool.submit(prepare,datasets[r['bucket']],r['index'])
 pending=submit(start) if start<steps else None
 for step in range(start,steps):
  pos=step*world+rank;active=pos<len(plan);r=plan[min(pos,len(plan)-1)]
  log({'event':'step_start','step':step+1,'buckets':[x['bucket'] for x in plan[step*world:(step+1)*world]]})
  t=time.monotonic();prepared=pending.result();prefetch_wait=time.monotonic()-t
  pending=submit(step+1) if step+1<steps else None
  batch,sample,n,timing=item(datasets[r['bucket']],r['index'],prepared);n_effective=n if active else 0
  assert n==lengths[r['window_id']]['target_tokens']
  count=sum(lengths[x['window_id']]['target_tokens'] for x in plan[step*world:(step+1)*world])
  timing['prefetch_wait_seconds']=prefetch_wait
  warmup=max(1,math.ceil(steps*.05));factor=min((step+1)/warmup,1.)*min(1.,max(.1,(steps-step)/max(1,steps-warmup)))
  for group in optimizer.param_groups:group['lr']=1e-5*factor
  optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();compute_start=time.monotonic();loss=ddp(batch);assert torch.isfinite(loss)
  timing['forward_seconds']=time.monotonic()-compute_start;backward_start=time.monotonic()
  (loss*(world*n_effective/count)).backward();torch.cuda.synchronize();timing['backward_and_sync_seconds']=time.monotonic()-backward_start
  norm=torch.nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm) and norm>0
  optimizer.step();del batch;torch.cuda.synchronize()
  timing.update(offloaded_gib=objective.offload.total_bytes/2**30,activation_offload=objective.offload.enabled)
  record={'timing':timing,'rank':rank,'window_id':r['window_id'],'record_id':r['record_id'],'active':active,'seconds':sample['audio_duration'] if active else 0,'target_tokens':n_effective,'loss':loss.item(),'grad_norm':norm.item(),'wall_seconds':time.monotonic()-t,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
  records=[None]*world;dist.all_gather_object(records,record);log({'event':'step','step':step+1,'lr':optimizer.param_groups[0]['lr'],'ranks':records})
  if (step+1)%25==0 or step+1==steps:evaluate(step+1)
  hours=sum(x['seconds'] for x in plan[:(step+1)*world])/3600
  decision=[False]
  if rank==0 and (EVAL/'policy.json').exists():
   decision[0]=evaluation_due(hours,step+1,steps,json.loads((EVAL/'policy.json').read_text()))
  dist.broadcast_object_list(decision,src=0)
  if step==0 or (step+1)%25==0 or step+1==steps or decision[0]:save(step+1)
  if decision[0]:
   if rank==0:
    atomic_json(EVAL/'request.json',{'id':f'checkpoint-{step+1:06d}','step':step+1,'hours':hours,'final':step+1==steps,'created_at':time.time()})
    log({'event':'evaluation_handoff','step':step+1,'hours':hours})
   pool.shutdown(wait=True);dist.barrier();dist.destroy_process_group();return
 pool.shutdown(wait=True)
 if rank==0:atomic_json(OUT/'completed.json',{'steps':steps,'finished_at':time.time(),'plan_sha256':plan_hash})
 dist.destroy_process_group()
if __name__=='__main__':main()
