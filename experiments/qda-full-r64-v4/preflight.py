"""Discarded longest-context optimizer step, only after all dataset gates pass."""
import argparse
import json
import pathlib
import time
import torch
from contracts import atomic_json, sha
from gates import check_data, demand, lease_data
from prepare import DATA, require_old_jobs_stopped
from model import fresh_model
from train_restart import Cache, Objective

HERE=pathlib.Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[16,64],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    require_old_jobs_stopped();check_data(DATA,reviewed=True)
    torch.cuda.set_device(a.gpu);torch.set_num_threads(2);torch.manual_seed(20261023)
    plan=json.loads((HERE/'plan.json').read_text());row=max((r for p in plan['probe_pairs'] for r in p),key=lambda r:r['total_tokens'])
    demand(plan['dataset_admission_sha256']==sha(DATA/'admission.json'),'Preflight plan contains stale data/reviews')
    model=fresh_model(a.rank);objective=Objective(model);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=0.)
    item=Cache(plan['cache_contract_sha256']).item(row);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    loss,stats=objective(item);demand(torch.isfinite(loss),'Nonfinite preflight loss');loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);demand(torch.isfinite(norm) and norm>0,'Invalid gradient');opt.step()
    demand(all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad),'Nonfinite adapter')
    atomic_json(HERE/f'preflight-r{a.rank}.json',{'status':'passed','rank':a.rank,'alpha':32,'mode':'nf4-qlora','world':'one discarded update per rank option',
                'window_id':row['window_id'],'plan_sha256':sha(HERE/'plan.json'),'admission_sha256':sha(DATA/'admission.json'),
                'peak_gib':torch.cuda.max_memory_allocated()/2**30,'wall_seconds':time.monotonic()-start,'loss':float(loss),'updates_discarded':True})


if __name__=='__main__':
    with lease_data(DATA):main()
