"""Two-GPU loader/backward check; no optimizer updates or model checkpoints."""
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contracts import atomic_json,sha
from gates import check_data,demand
from model import fresh_model
from prepare import DATA,require_old_jobs_stopped
from train_restart import Cache,Objective

HERE=Path(__file__).resolve().parent


def main():
    require_old_jobs_stopped();check_data(DATA)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    plan=json.loads((HERE/'plan.json').read_text());rows=[r for p in plan['probe_pairs'] for r in p]
    # Exercise actual synthetic labels/context on one GPU and the longest real
    # recording on the other. The normal training objective is unchanged.
    candidates=[r for r in rows if r['dataset_kind']==('real' if rank==0 else 'synthetic')]
    row=max(candidates,key=lambda r:r['total_tokens'])
    model=fresh_model(64);model.train();objective=Objective(model)
    ddp=DDP(objective,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True)
    item=Cache(plan['cache_contract_sha256']).item(row)
    # Reserve the actual Adam moment footprint without taking an optimizer step.
    moments=[torch.zeros_like(p) for p in model.parameters() if p.requires_grad for _ in range(2)]
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    loss,stats=ddp(item);demand(torch.isfinite(loss),'Nonfinite loader-check loss');loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    demand(torch.isfinite(norm) and norm>0,'Invalid gradient in loader check')
    for module in model.modules():
        if hasattr(module,'lora_B') and 'default' in module.lora_B:
            demand(module.lora_B['default'].weight.detach().count_nonzero()==0,'Preflight mutated the fresh adapter')
    torch.cuda.synchronize()
    result={'rank':rank,'dataset_kind':row['dataset_kind'],'window_id':row['window_id'],
        'total_tokens':len(item[0][0]),'duration_seconds':row['duration'],
        'peak_gib':torch.cuda.max_memory_allocated()/2**30,'loss':float(loss),
        'gradient_norm':float(norm),'wall_seconds':time.monotonic()-begin}
    results=[None,None];dist.all_gather_object(results,result)
    if rank==0:atomic_json(HERE/'preflight-r64.json',{'status':'passed','rank':64,'world_size':2,
        'optimizer_updates':0,'adapter_unchanged':True,'plan_sha256':sha(HERE/'plan.json'),
        'admission_sha256':sha(DATA/'admission.json'),'results':results})
    del moments
    dist.destroy_process_group()


if __name__=='__main__':main()
