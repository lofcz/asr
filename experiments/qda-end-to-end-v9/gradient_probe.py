"""One disposable full-weight Adam step, streamed parameter-by-parameter on CPU.

This checks gradients/weight changes on the real 8B model without persisting Adam
moments. It is NOT a resumable trainer or evidence of WER improvement. Production
training uses ordinary sharded Adam states on a sufficiently provisioned host.
"""
import argparse,gc,json,math,pathlib,time
import torch
from runtime import HERE,atomic,load_parent,prepare_item,Objective
from audio_path import trainable_groups,receptive_field

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
 torch.cuda.set_device(args.gpu);torch.set_num_threads(2);torch.manual_seed(20261025)
 start=time.time();model,processor,counts=load_parent();groups=trainable_groups(model)
 manifest=json.loads((HERE/'manifest.json').read_text());rows=[min((r for r in manifest['train'] if r['language']==lang),key=lambda r:r['duration']) for lang in ['cs','sk']]
 items=[prepare_item(r,processor) for r in rows]
 atomic(HERE/'probe-progress.json',{'stage':'loading','parameters':counts,'training_clips':[{k:r[k] for k in ['window_id','duration','language']} for r in rows]})
 model.cuda().train();objective=Objective(model,tile_seconds=4.).train()
 stats={k:{'parameters':counts[k],'gradient_tensors':0,'nonzero_gradient_tensors':0,'gradient_squared_norm':0.,'updated_tensors':0,'changed_elements':0} for k in groups}
 membership={n:k for k,ps in groups.items() for n,p in ps};grads={};hooks=[]
 # Accumulate exact gradients on the host so one GPU can check the complete graph.
 # No parameter update happens until both backwards finish.
 def capture(name):
  def hook(p):
   if p.grad is None:raise ValueError('Missing accumulated gradient')
   g=p.grad.detach().cpu()
   if name in grads:grads[name].add_(g)
   else:grads[name]=g
   p.grad=None
  return hook
 for name,p in model.named_parameters():
  if p.requires_grad:hooks.append(p.register_post_accumulate_grad_hook(capture(name)))
 torch.cuda.reset_peak_memory_stats();before=[]
 for r,item in zip(rows,items):
  torch.manual_seed(20261025);loss=objective(item);before.append(float(loss.detach()));(loss/len(items)).backward()
  atomic(HERE/'probe-progress.json',{'stage':'backward','language':r['language'],'loss':before[-1],'gradient_tensors':len(grads),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
  del loss;gc.collect();torch.cuda.empty_cache()
 for hook in hooks:hook.remove()
 # One-step test with FP32 master values. Discarding moments is legal only for
 # this disposable verification; never use this loop for multi-step learning.
 lr=1e-5
 for name,p in model.named_parameters():
  if not p.requires_grad:continue
  if name not in grads:raise ValueError('No gradient for trainable parameter: '+name)
  g=grads.pop(name).float();s=stats[membership[name]]
  if not torch.isfinite(g).all():raise ValueError('Nonfinite gradient: '+name)
  s['gradient_tensors']+=1;s['nonzero_gradient_tensors']+=int(torch.count_nonzero(g)>0);s['gradient_squared_norm']+=float(g.square().sum())
  master=torch.nn.Parameter(p.detach().cpu().float());old=master.detach().to(torch.bfloat16);master.grad=g
  optimizer=torch.optim.AdamW([master],lr=lr,weight_decay=0.)
  optimizer.step();new=master.detach().to(torch.bfloat16);changed=int(torch.count_nonzero(new!=old))
  s['updated_tensors']+=int(changed>0);s['changed_elements']+=changed
  with torch.no_grad():p.copy_(new.to(p.device))
  del master,old,new,g,optimizer
 for name,s in stats.items():
  s['gradient_norm']=math.sqrt(s.pop('gradient_squared_norm'))
  if s['gradient_norm']<=0 or s['changed_elements']==0:raise ValueError('Disconnected or unchanging group: '+name)
 after=[];model.eval();objective.eval()
 with torch.no_grad():
  for item in items:
   torch.manual_seed(20261025);after.append(float(objective(item)))
 result={'status':'passed','parent_step':manifest['parent_step'],'parent_adapter_sha256':manifest['parent_adapter_sha256'],
  'full_model_unquantized':True,'merged_lora_no_remaining_adapters':True,'raw_audio_to_text_loss':True,
  'groups':stats,'loss_before':before,'loss_after_one_disposable_step':after,'verification_lr':lr,
  'optimizer_steps':1,'persistent_checkpoints_written':0,'cpu_grad_accumulation':True,'optimizer_moments_discarded_after_verification':True,
  'production_training_feasibility_not_established':True,'wer_improvement_not_measured':True,
  'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'wall_seconds':time.time()-start,
  'receptive_fields':{k:receptive_field(getattr(model.model,k).encoder) for k in ['acoustic_tokenizer','semantic_tokenizer']}}
 atomic(HERE/'gradient-probe-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
