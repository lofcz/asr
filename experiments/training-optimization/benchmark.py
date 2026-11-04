import sys,pathlib,os,json,time,gc,argparse
sys.path.insert(0,'/data/work/asr-pilot/long-context-pilot')
import torch
from train import Cache,load_model,loss_for as old_loss,BoundaryOffload
from optimized_kernels import AdaptiveOffload,loss_for as new_loss
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['old','new'],required=True);p.add_argument('--bucket',type=int,required=True);p.add_argument('--stress',action='store_true');p.add_argument('--stress-tokens',type=int,default=65147);a=p.parse_args()
out=pathlib.Path('/data/work/asr-optimization');out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.manual_seed(20261022)
model=load_model('qlora',16,1024 if a.variant=='old' else 2048);root=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16');latest=json.loads((root/'latest.json').read_text());set_peft_model_state_dict(model,load_file(str(root/latest['checkpoint']/'adapter_model.safetensors')))
# Disable dropout for comparable loss/gradients across different chunk layouts.
for m in model.modules():
 if isinstance(m,torch.nn.Dropout) and not a.stress:m.p=0
model.train();params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
cache=Cache();row=next(r for r in cache.rows if r['split']=='train' and r['bucket']==a.bucket);item=cache.item(row)
if a.stress:
 ids,mask,features,labels=item;extra=a.stress_tokens-ids.shape[1];assert extra>=0
 item=(torch.cat([ids,ids[:,-1:].expand(1,extra)],1),torch.cat([mask,torch.zeros((1,extra),dtype=torch.bool,device='cuda')],1),features,torch.cat([labels,labels[:,-1:].expand(1,extra)],1))
 row={**row,'token_length':a.stress_tokens}
offload=BoundaryOffload(True,pin_memory=False) if a.variant=='old' else AdaptiveOffload();fn=old_loss if a.variant=='old' else new_loss
results=[]
for i in range(1 if a.stress else 2):
 optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();value=fn(model,item,offload);value.backward();torch.cuda.synchronize();elapsed=time.monotonic()-t
 assert torch.isfinite(value) and all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
 if i==0 and not a.stress:
  torch.save({name:p.grad.detach().cpu() for name,p in model.named_parameters() if p.requires_grad and p.grad is not None},out/f'grads-{a.variant}-{a.bucket}.pt')
 # Initialize optimizer state without changing the weights used for equivalence.
 for group in optimizer.param_groups:group['lr']=0
 optimizer.step();torch.cuda.synchronize()
 r={'variant':a.variant,'bucket':a.bucket,'iteration':i,'tokens':row['token_length'],'loss':value.item(),'forward_backward_seconds':elapsed,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'offloaded_gib':offload.total_bytes/2**30};results.append(r);print(json.dumps(r),flush=True)
(out/f'bench-{a.variant}-{a.bucket}{"-stress" if a.stress else ""}-gpu{os.environ.get("CUDA_VISIBLE_DEVICES","0")}.json').write_text(json.dumps(results,indent=2))
