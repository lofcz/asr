"""Two-GPU, resumable, token-weighted continuation with cached frozen features."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,datetime,gc,hashlib,json,math,pathlib,random,sys,time
import numpy as np,torch,torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot','/data/work/asr-pilot/VibeVoice']
from train import load_model
from optimized_kernels import AdaptiveOffload
from peft import set_peft_model_state_dict,get_peft_model_state_dict
from safetensors.torch import load_file
from contracts import sha,atomic_json,wsd_lr
from optimizers import build_optimizer
from runtime_state import runtime_hashes

class Cache:
 def __init__(self,contract):self.verified=set();self.contract=contract
 def item(self,row):
  wid=row['window_id'];folder=HERE/'cache';ip=folder/(wid+'.input.pt');fp=folder/(wid+'.features.npy');meta=json.loads((folder/(wid+'.features.json')).read_text())
  if wid not in self.verified:
   assert sha(ip)==row['input_sha256'] and sha(fp)==meta['sha256'] and meta['contract_sha256']==self.contract and meta['input_sha256']==row['input_sha256'];self.verified.add(wid)
  x=torch.load(ip,map_location='cpu',weights_only=True);f=torch.from_numpy(np.load(fp,allow_pickle=False).copy()).view(torch.bfloat16)
  assert f.shape[0]==int(x['mask'].sum()) and int((x['labels']!=-100).sum())==row['target_tokens']
  return tuple(x[k].unsqueeze(0).to('cuda') for k in ('ids','mask'))+(f.to('cuda'),x['labels'].unsqueeze(0).to('cuda'),x['content_mask'].unsqueeze(0).to('cuda'))

class Objective(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model;self.offload=AdaptiveOffload()
 def forward(self,item):
  ids,mask,features,labels,content=item;base=self.model.get_base_model()
  with self.offload.context(ids.shape[1]),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   embeddings=base.get_input_embeddings()(ids).clone();embeddings[mask]=features
   hidden=base.model(inputs_embeds=embeddings,use_cache=False,return_dict=True).last_hidden_state
  targets=labels[:,1:].reshape(-1);states=hidden[:,:-1].reshape(-1,hidden.shape[-1]);valid=targets!=-100;states=states[valid];targets=targets[valid];content=content[:,1:].reshape(-1)[valid]
  def chunk(x,y,c):
   losses=F.cross_entropy(base.lm_head(x.to(base.lm_head.weight.dtype)).float(),y,reduction='none')
   return torch.stack((losses.sum(),losses[c].sum(),losses[~c].sum()))
  sums=torch.zeros(3,device=ids.device,dtype=torch.float32)
  for i in range(0,len(targets),512):
   inputs=(states[i:i+512],targets[i:i+512],content[i:i+512])
   sums=sums+(checkpoint(chunk,*inputs,use_reentrant=False) if self.training and torch.is_grad_enabled() else chunk(*inputs))
  counts=torch.tensor([len(targets),int(content.sum()),int((~content).sum())],device=ids.device,dtype=torch.float64)
  return sums[0]/len(targets),torch.cat((sums.detach().double(),counts))

def main():
 p=argparse.ArgumentParser();p.add_argument('--optimizer',choices=['adamw','polora'],required=True);p.add_argument('--stage',choices=['calibration','probe'],required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--name',required=True);args=p.parse_args()
 assert args.lr>0 and '/' not in args.name and '..' not in args.name
 baseline=HERE/'baseline';provenance=json.loads((baseline/'provenance.json').read_text());assert sha(baseline/'adapter_model.safetensors')==provenance['adapter_sha256']
 assert (pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')/'completed.json').exists(),'Baseline incomplete'
 local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=datetime.timedelta(minutes=30));rank=dist.get_rank();world=dist.get_world_size();assert world==2
 plan=json.loads((HERE/'plan.json').read_text());assert sha(HERE/'cache/contract.json')==plan['cache_contract_sha256']
 pairs=plan['calibration_pairs'] if args.stage=='calibration' else plan['probe_pairs'];steps=len(pairs);total_tokens=sum(r['target_tokens'] for pair in pairs for r in pair)
 warmup=sum(r['target_tokens'] for pair in pairs[:min(6,max(1,steps//10))] for r in pair)
 out=HERE/'runs'/args.name;out.mkdir(parents=True,exist_ok=True)
 config={**vars(args),'rank':16,'alpha':32,'fresh_optimizer':True,'weight_decay':0.,'dropout':.05,'seed':plan['seed'],'world_size':2,'plan_sha256':sha(HERE/'plan.json'),'baseline_sha256':provenance['adapter_sha256'],'code_sha256':sha(pathlib.Path(__file__)),'optimizer_code_sha256':sha(HERE/'optimizers.py'),'steps':steps,'target_tokens':total_tokens,'warmup_tokens':warmup,'schedule':'constant calibration / token-clock WSD probe','cooldown_fraction':.2,'min_lr_ratio':.1,'data_hours':sum(r['duration'] for pair in pairs for r in pair)/3600}
 config['runtime_sha256']=runtime_hashes()
 def log(event):
  if rank==0:
   event={'time':time.time(),'run':args.name,**event};line=json.dumps(event,allow_nan=False)
   with (out/'events.jsonl').open('a') as f:f.write(line+'\n');f.flush();os.fsync(f.fileno())
   print(line,flush=True)
 if rank==0:
  if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'Run config changed'
  else:atomic_json(out/'config.json',config)
 dist.barrier();torch.manual_seed(plan['seed']);random.seed(plan['seed']);np.random.seed(plan['seed'])
 model=load_model('qlora',16,2048);weights=load_file(str(baseline/'adapter_model.safetensors'));result=set_peft_model_state_dict(model,weights)
 assert not result.unexpected_keys,'Adapter mismatch'
 actual=get_peft_model_state_dict(model);assert actual.keys()==weights.keys(),'Incomplete baseline adapter'
 assert all(torch.equal(v.detach().cpu(),weights[k].to(v.dtype)) for k,v in actual.items()),'Baseline adapter values changed'
 del weights,actual
 model.train();params=[p for p in model.parameters() if p.requires_grad];optimizer=build_optimizer(model,args.optimizer,args.lr)
 objective=Objective(model);ddp=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True);cache=Cache(plan['cache_contract_sha256'])
 start=0;seen=0;bad_checks=0
 if (out/'latest.json').exists():
  checkpoint_path=out/json.loads((out/'latest.json').read_text())['checkpoint'];state=torch.load(checkpoint_path/'state.pt',map_location='cpu',weights_only=False)
  assert state['config']==config and sha(checkpoint_path/'adapter_model.safetensors')==state['adapter_sha256']
  set_peft_model_state_dict(model,load_file(str(checkpoint_path/'adapter_model.safetensors')));optimizer.load_state_dict(state['optimizer']);start=state['step'];seen=state['tokens_seen'];bad_checks=state.get('bad_checks',0)
  rng=state['rng'][rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy']);del state,rng
  log({'event':'resumed','step':start})
 else:torch.manual_seed(plan['seed']+rank)
 def evaluate(step,full=False):
  model.eval();objective.eval();sums=torch.zeros(6,device='cuda',dtype=torch.float64)
  rows=plan['validation'] if full else plan['validation_small']
  with torch.no_grad():
   for r in rows[rank::world]:
    item=cache.item(r);_,s=objective(item);sums+=s;del item
  dist.all_reduce(sums);values=sums.cpu().tolist();model.train();objective.train()
  result={'event':'validation','step':step,'full_panel':full,'loss':values[0]/values[3],'content_loss':values[1]/values[4],'structure_loss':values[2]/values[5],'target_tokens':int(values[3]),'content_tokens':int(values[4]),'windows':len(rows)};log(result);return result
 def save(step,tag=None):
  rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state()};states=[None]*world;dist.all_gather_object(states,rng)
  if rank==0:
   name=f'checkpoint-{step:06d}';tmp=out/(name+'.building');dest=out/name
   if not dest.exists():
    tmp.mkdir(exist_ok=True);model.save_pretrained(tmp,safe_serialization=True)
    torch.save({'step':step,'tokens_seen':seen,'config':config,'optimizer':optimizer.state_dict(),'rng':states,'bad_checks':bad_checks,'adapter_sha256':sha(tmp/'adapter_model.safetensors')},tmp/'state.pt')
    for f in tmp.iterdir():
     if f.is_file():
      with f.open('rb') as h:os.fsync(h.fileno())
    os.replace(tmp,dest)
   atomic_json(out/'latest.json',{'checkpoint':name,'step':step})
   if tag:atomic_json(out/(tag+'.json'),{'checkpoint':name,'step':step,'tokens_seen':seen})
   log({'event':'checkpoint','step':step,'checkpoint':name,'tag':tag})
  dist.barrier()
 initial_file=out/'initial-validation.json'
 if start==0:
  initial=evaluate(0)
  if rank==0:atomic_json(initial_file,initial)
 else:initial=json.loads(initial_file.read_text())
 if args.stage=='probe' and start==0:evaluate(0,full=True)
 for step in range(start,steps):
  pair=pairs[step];active=rank<len(pair);row=pair[min(rank,len(pair)-1)];n=sum(r['target_tokens'] for r in pair);begin=time.monotonic();item=cache.item(row);prep=time.monotonic()-begin
  lr=args.lr if args.stage=='calibration' else wsd_lr(args.lr,seen+n,total_tokens,warmup)
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=time.monotonic();loss,stats=ddp(item)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; preserving prior checkpoint')
  (loss*(world*row['target_tokens']/n if active else 0.)).backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
  if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
  torch.cuda.synchronize();compute=time.monotonic()-t;t=time.monotonic();optimizer.step();torch.cuda.synchronize();opt_seconds=time.monotonic()-t;seen+=n
  if not active:stats.zero_()
  dist.all_reduce(stats);s=stats.cpu().tolist();elapsed=time.monotonic()-begin
  timing=torch.tensor([elapsed,prep,compute,opt_seconds,torch.cuda.max_memory_allocated()/2**30],device='cuda',dtype=torch.float64);dist.all_reduce(timing,op=dist.ReduceOp.MAX)
  log({'event':'step','step':step+1,'tokens_seen':seen,'audio_hours':sum(r['duration'] for r in pair)/3600,'audio_hours_seen':sum(r['duration'] for p in pairs[:step+1] for r in p)/3600,'lr':lr,'loss':s[0]/s[3],'content_loss':s[1]/s[4],'structure_loss':s[2]/s[5],'gradient_norm':float(norm),'wall_seconds':float(timing[0]),'cache_load_seconds':float(timing[1]),'compute_seconds':float(timing[2]),'optimizer_seconds':float(timing[3]),'peak_gpu_gib':float(timing[4]),'target_tokens':n})
  del item,loss,stats
  stable_end=total_tokens*.8
  stable_boundary=args.stage=='probe' and seen<=stable_end and (step+1==steps or seen+sum(r['target_tokens'] for r in pairs[step+1])>stable_end)
  validation_due=(step+1)%10==0 or step+1==steps
  if validation_due:
   val=evaluate(step+1);bad_checks=bad_checks+1 if val['content_loss']>initial['content_loss']*1.15 else 0
  if validation_due or stable_boundary:save(step+1,'stable-end' if stable_boundary else None)
  if validation_due and bad_checks>=2:
   if rank==0:atomic_json(out/'stopped.json',{'reason':'content loss >15% above start on two consecutive checks','step':step+1})
   dist.destroy_process_group();return
 if args.stage=='probe':evaluate(steps,full=True)
 if rank==0:atomic_json(out/'completed.json',{'steps':steps,'finished_at':time.time(),'config_sha256':sha(out/'config.json'),'tokens_seen':seen,'training_audio_hours':config['data_hours']})
 dist.destroy_process_group()
if __name__=='__main__':main()
