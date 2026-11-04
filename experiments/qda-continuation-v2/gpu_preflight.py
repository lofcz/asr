"""One real longest-window update, discarded after checking memory and finite values."""
import argparse,json,pathlib,sys,time
import torch
HERE=pathlib.Path(__file__).resolve().parent
from train_continuation import Cache,Objective,load_model,set_peft_model_state_dict,load_file
from contracts import sha,atomic_json
from optimizers import build_optimizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--optimizer',choices=['adamw','polora'],required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
 assert pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16/completed.json').exists(),'Baseline incomplete'
 torch.cuda.set_device(args.gpu);torch.set_num_threads(2);torch.manual_seed(20261023)
 plan=json.loads((HERE/'plan.json').read_text());row=max((r for pair in plan['probe_pairs'] for r in pair),key=lambda r:r['total_tokens']);model=load_model('qlora',16,2048)
 set_peft_model_state_dict(model,load_file(str(HERE/'baseline/adapter_model.safetensors')));model.train();opt=build_optimizer(model,args.optimizer,3e-5 if args.optimizer=='adamw' else .001);objective=Objective(model);cache=Cache(plan['cache_contract_sha256']);item=cache.item(row)
 started=time.monotonic();torch.cuda.reset_peak_memory_stats();loss,stats=objective(item);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step()
 assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
 atomic_json(HERE/f'gpu-preflight-{args.optimizer}.json',{'status':'passed','optimizer':args.optimizer,'window_id':row['window_id'],'total_tokens':row['total_tokens'],'duration_seconds':row['duration'],'loss':float(loss),'content_loss':float(stats[1]/stats[4]),'peak_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-started,'adapter_sha256':sha(HERE/'baseline/adapter_model.safetensors'),'plan_sha256':sha(HERE/'plan.json'),'updates_discarded':True})
if __name__=='__main__':main()
