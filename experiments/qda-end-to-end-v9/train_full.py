"""Portable full-weight pilot. Requires a measured LR decision and enough RAM.

Uses DeepSpeed ZeRO-3 + CPU Adam, FP32 master weights/moments and BF16 compute.
The running v8 code, its cached features and its optimizer are never modified.
"""
import argparse,hashlib,json,math,os,pathlib,random,re,shutil,time
import torch
import torch.distributed as dist
from runtime import HERE,atomic,sha,load_parent,prepare_item,Objective
from audio_path import trainable_groups


def required_memory(parameters):
 # DeepSpeed's optimizer-offload estimator: CPU ~16 bytes per parameter,
 # plus 1.5 safety factor. GPU BF16 parameter shards + largest-layer gather
 # and activations are accounted for separately at actual preflight.
 return math.ceil(parameters*16*1.5)


def main():
 p=argparse.ArgumentParser();p.add_argument('--decision',type=pathlib.Path,required=True)
 p.add_argument('--branch',choices=['end-to-end','decoder-only-control'],required=True)
 p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--resume',action='store_true')
 p.add_argument('--max-steps',type=int,default=96);p.add_argument('--microbatches',type=int,default=2)
 p.add_argument('--checkpoint-every',type=int,default=24);p.add_argument('--local_rank',type=int,default=0)
 args=p.parse_args()
 import psutil
 # Refuse before loading a model or allocating an optimizer on the 64GB VM.
 estimated_parameters=8330000000
 if psutil.virtual_memory().total<required_memory(estimated_parameters):
  raise RuntimeError('This CPU-offloaded full Adam configuration needs approximately 187 GiB host RAM with safety margin; provision 256 GiB. This host has insufficient RAM. No training started.')
 disk_parent=args.out.resolve()
 while not disk_parent.exists():disk_parent=disk_parent.parent
 if shutil.disk_usage(disk_parent).free<300*2**30 and not args.resume:
  raise RuntimeError('Provision at least 300 GiB free checkpoint storage for this full-Adam pilot. No training started.')
 decision=json.loads(args.decision.read_text());manifest=json.loads((HERE/'manifest.json').read_text())
 assert decision['parent_adapter_sha256']==manifest['parent_adapter_sha256'] and decision['manifest_sha256']==sha(HERE/'manifest.json')
 assert decision['parameterization']=='full-bf16' and decision['optimizer']=='adamw' and decision['lr_finder_completed'] is True
 assert decision['restoration_verified'] is True and decision['learning_rate']>0
 assert sha(pathlib.Path(decision['history_path']))==decision['history_sha256']
 assert 1<=args.max_steps<=96,'Pilot only; a full-corpus continuation is a separate reviewed configuration'
 import deepspeed
 local=int(os.getenv('LOCAL_RANK',args.local_rank));torch.cuda.set_device(local);torch.set_num_threads(2)
 deepspeed.init_distributed();rank=dist.get_rank();world=dist.get_world_size()
 torch.manual_seed(manifest['seed']);random.seed(manifest['seed'])
 model,processor,counts=load_parent();groups=trainable_groups(model)
 if args.branch=='decoder-only-control':
  for group,parameters in groups.items():
   if group not in ['language_model','lm_head']:
    for _,parameter in parameters:parameter.requires_grad_(False)
 model.train();objective=Objective(model)
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 config={'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':args.microbatches,
  'train_batch_size':world*args.microbatches,'bf16':{'enabled':True},'gradient_clipping':1.0,
  'optimizer':{'type':'Adam','params':{'lr':decision['learning_rate'],'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'adam_w_mode':True}},
  'zero_optimization':{'stage':3,'offload_optimizer':{'device':'cpu','pin_memory':True},
   'overlap_comm':True,'contiguous_gradients':True,'stage3_gather_16bit_weights_on_model_save':True,
   'reduce_bucket_size':25000000,'stage3_prefetch_bucket_size':25000000,'stage3_param_persistence_threshold':100000},
  'steps_per_print':1000000,'wall_clock_breakdown':False}
 engine,optimizer,_,_=deepspeed.initialize(model=objective,model_parameters=[p for p in model.parameters() if p.requires_grad],config=config)
 args.out.mkdir(parents=True,exist_ok=True)
 identity={'branch':args.branch,'parent_adapter_sha256':manifest['parent_adapter_sha256'],'manifest_sha256':sha(HERE/'manifest.json'),
  'decision_sha256':sha(args.decision),'max_steps':args.max_steps,'microbatches':args.microbatches,'world_size':world,
  'code_sha256':{n:sha(HERE/n) for n in ['runtime.py','audio_path.py','train_full.py']},'trainable_parameters':trainable,'counts':counts}
 start=0
 if args.resume:
  loaded,client=engine.load_checkpoint(str(args.out/'checkpoints'))
  if not loaded or client['identity']!=identity:raise ValueError('Resume identity changed')
  start=client['step'];rng=client['rng'][rank];torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
 elif (args.out/'events.jsonl').exists():raise ValueError('Use a new run directory or explicit resume')
 if rank==0:atomic(args.out/'config.json',identity)
 def log(event):
  if rank==0:
   event={'time':time.time(),**event}
   with (args.out/'events.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(event),flush=True)
 def save(step):
  rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()};states=[None]*world;dist.all_gather_object(states,rng)
  engine.save_checkpoint(str(args.out/'checkpoints'),tag=f'step-{step:06d}',client_state={'identity':identity,'step':step,'rng':states})
  dist.barrier()
  if rank==0:
   atomic(args.out/'latest.json',{'step':step,'tag':f'step-{step:06d}'})
   # Retain two complete checkpoints from this run. Never touch parent/v8 files.
   prior=sorted(p for p in (args.out/'checkpoints').iterdir() if p.is_dir() and re.fullmatch(r'step-\d{6}',p.name) and int(p.name[5:])<=step)
   for obsolete in prior[:-2]:shutil.rmtree(obsolete)
  dist.barrier()
 # Decoded waveform/token preparation can be cached, but never trainable features.
 prepared={}
 def item(row):
  if row['window_id'] not in prepared:prepared[row['window_id']]=prepare_item(row,processor)
  return prepared[row['window_id']]
 pool=manifest['train'];global_batch=world*args.microbatches
 for step in range(start,args.max_steps):
  begin=time.monotonic();rows=[pool[(step*global_batch+micro*world+rank)%len(pool)] for micro in range(args.microbatches)]
  items=[item(row) for row in rows]
  denominator=torch.tensor(sum(x['target_tokens'] for x in items),device='cuda',dtype=torch.float64);dist.all_reduce(denominator)
  aggregate=torch.zeros(2,device='cuda',dtype=torch.float64)
  for x in items:
   loss=engine(x)
   finite=torch.isfinite(loss).to(dtype=torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
   if not finite.item():raise RuntimeError('Nonfinite loss; previous checkpoint preserved')
   # DeepSpeed divides by accumulation count; distributed gradients are averaged.
   scaled=loss*(world*args.microbatches*x['target_tokens']/denominator)
   engine.backward(scaled);engine.step()
   aggregate[0]+=loss.detach().double()*x['target_tokens'];aggregate[1]+=x['target_tokens']
  dist.all_reduce(aggregate)
  norm=float(engine.get_global_grad_norm())
  if not math.isfinite(norm):raise RuntimeError('Nonfinite gradients; resume the previous completed checkpoint')
  log({'event':'step','step':step+1,'loss':float(aggregate[0]/aggregate[1]),'target_tokens':int(aggregate[1]),
       'lr':optimizer.param_groups[0]['lr'],'gradient_norm':norm,'wall_seconds':time.monotonic()-begin,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
  if (step+1)%args.checkpoint_every==0 or step+1==args.max_steps:save(step+1)
 engine.save_16bit_model(str(args.out/'export'),save_filename='pytorch_model.bin')
 if rank==0:atomic(args.out/'completed.json',{'steps':args.max_steps,'identity':identity,'evaluation_pending':True})
 dist.destroy_process_group()
if __name__=='__main__':main()
