"""Discarded rank64 LR range calibration using torch-lr-finder's published test.

Keep the original trainer, audited data and checkpoints immutable. Only the DDP
loss/validation hooks and adapter-only temporary state handling are customized.
The upstream finder controls LR, exponential sweep, smoothing and divergence.
"""
import os
os.environ.setdefault('HF_HOME', '/data/work/asr-pilot/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse, datetime, fcntl, json, math, pathlib, random, sys, time
HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path('/data/work/asr-experiments/qda-restart-v3')
sys.path[:0] = [str(HERE/'vendor'), str(BASE)]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from contracts import sha, atomic_json
from gates import demand, check_data, lease_data
from prepare import DATA, require_old_jobs_stopped
from train_restart import Cache, Objective
from model import fresh_model
from runtime_state import runtime_hashes


class AdapterState(torch.nn.Module):
    """Only adapters can change; do not copy 7B frozen quantized weights."""
    def __init__(self, ddp):
        super().__init__(); self.ddp = ddp

    def state_dict(self, *args, **kwargs):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_state_dict(self, state, *args, **kwargs):
        params = dict(self.named_parameters())
        demand(set(state) == {n for n,p in params.items() if p.requires_grad}, 'Adapter restore mismatch')
        with torch.no_grad():
            for n, value in state.items(): params[n].copy_(value)

    def to(self, *args, **kwargs):
        # DDP and the quantized model already live on this worker's device.
        return self


class DistributedFinder(LRFinder):
    def __init__(self, model, optimizer, pairs, validation, cache, rank, output, max_seconds):
        super().__init__(model, optimizer, torch.nn.Identity(), device=f'cuda:{rank}')
        self.pairs = pairs; self.validation = validation; self.cache = cache
        self.rank = rank; self.output = output; self.iteration = 0
        self.started = time.monotonic(); self.max_seconds = max_seconds
        self.records = []; self.latest = {}; self.before = None

    def persist(self):
        if self.rank == 0:
            atomic_json(self.output/'history.json', {'records':self.records, 'running':True,
                        'updated_at':time.time(), 'upstream_history':self.history})

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        demand(accumulation_steps == 1, 'Range test must preserve two-worker batch size')
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        pair = self.pairs[self.iteration % len(self.pairs)]
        row = pair[self.rank]; tokens = sum(x['target_tokens'] for x in pair)
        start = time.monotonic(); item = self.cache.item(row)
        loss, stats = self.model.ddp(item)
        finite = torch.isfinite(loss).int(); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError('Nonfinite loss')
        (loss * (2*row['target_tokens']/tokens)).backward()
        params = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        finite = torch.isfinite(norm).int(); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError('Nonfinite gradients')
        self.optimizer.step(); dist.all_reduce(stats)
        values = stats.cpu().tolist(); self.iteration += 1
        self.latest = {'step':self.iteration, 'lr':self.optimizer.param_groups[0]['lr'],
            'train_loss':values[0]/values[3], 'train_content_loss':values[1]/values[4],
            'train_structure_loss':values[2]/values[5], 'gradient_norm':float(norm),
            'target_tokens':tokens, 'window_ids':[x['window_id'] for x in pair],
            'train_seconds':time.monotonic()-start}
        del item, loss, stats
        return self.latest['train_loss']

    def _validate(self, val_iter, non_blocking_transfer=True):
        self.model.eval(); sums = torch.zeros(6, device='cuda', dtype=torch.float64)
        with torch.no_grad():
            for row in self.validation[self.rank::2]:
                item = self.cache.item(row); _, stats = self.model.ddp.module(item)
                sums += stats; del item, stats
        dist.all_reduce(sums); v = sums.cpu().tolist()
        metrics = {'validation_loss':v[0]/v[3], 'validation_content_loss':v[1]/v[4],
                   'validation_structure_loss':v[2]/v[5]}
        if self.iteration:
            row = {**self.latest, **metrics, 'elapsed_seconds':time.monotonic()-self.started}
            self.records.append(row); self.persist()
            if self.rank == 0:
                with (self.output/'events.jsonl').open('a') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
                print(json.dumps(row), flush=True)
        else: self.before = metrics
        if time.monotonic()-self.started > self.max_seconds:
            raise TimeoutError('Predeclared LR-finder wall-time budget reached')
        return metrics['validation_content_loss']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--start-lr', type=float, default=1e-6)
    p.add_argument('--end-lr', type=float, default=1e-2)
    p.add_argument('--max-seconds', type=int, default=2700)
    a = p.parse_args()
    demand(10 <= a.iterations <= 100 and 0 < a.start_lr < a.end_lr, 'Invalid finder bounds')
    require_old_jobs_stopped()
    import subprocess
    status=subprocess.run(['systemctl','--user','is-active','qda-v3-experiments.service'],capture_output=True,text=True).stdout.strip()
    demand(status in ('inactive','failed','unknown'), 'Capacity probes are still running')
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank); torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.device('cuda',rank), timeout=datetime.timedelta(minutes=15))
    demand(dist.get_world_size()==2, 'This test requires the same two DDP workers')
    lock=None
    if rank==0:
        lock=(BASE/'gpu-work.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dist.barrier()
    if rank==0: check_data(DATA,reviewed=True)
    dist.barrier()
    plan=json.loads((BASE/'plan.json').read_text()); ready=json.loads((BASE/'ready.json').read_text())
    demand(ready['runtime_sha256']==runtime_hashes() and ready['plan_sha256']==sha(BASE/'plan.json'), 'Audited runtime changed')
    demand(plan['dataset_admission_sha256']==sha(DATA/'admission.json'), 'Dataset admission changed')
    # Short clips give enough range-test points within the bounded GPU budget.
    # Preserve original token-matched DDP pairs; never use dev examples to train.
    pairs=[pair for pair in plan['probe_pairs'] if len(pair)==2 and all(r['duration_class']=='short' for r in pair)]
    demand(len(pairs)>=10, 'Insufficient diverse matched short pairs')
    random.Random(plan['seed']).shuffle(pairs)
    demand({r['language'] for pair in pairs for r in pair}=={'cs','sk'}, 'Missing training language')
    output=HERE/'result'; output.mkdir(exist_ok=True)
    demand(not (output/'history.json').exists(), 'Finder already ran; preserve results and use a new version')
    config={**vars(a),'rank':64,'alpha':32,'optimizer':'AdamW','weight_decay':0.,'gradient_clip_norm':1.,
      'initialization':'fresh vanilla, zero B, fresh optimizer; all calibration weights discarded',
      'implementation':'torch-lr-finder==0.2.2 LRFinder.range_test(exp, smooth_f=.05, diverge_th=5)',
      'selection_metric':'same held-out CS/SK content loss after every update; train total CE drives gradients',
      'plan_sha256':sha(BASE/'plan.json'),'runtime_sha256':runtime_hashes(),'finder_code_sha256':sha(__file__),
      'pairs':[[r['window_id'] for r in pair] for pair in pairs],
      'batch_limitation':'Short-window LR range is diagnostic; it does not prove the full length mix or a decay schedule optimal.',
      'reference':'https://github.com/davidtvs/pytorch-lr-finder','full_training_started':False}
    if rank==0: atomic_json(output/'config.json',config)
    torch.manual_seed(plan['seed']); np.random.seed(plan['seed']); random.seed(plan['seed'])
    model=fresh_model(64); objective=Objective(model)
    ddp=DDP(objective,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True)
    wrapped=AdapterState(ddp)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.start_lr,weight_decay=0.)
    finder=DistributedFinder(wrapped,optimizer,pairs,plan['validation_small'],Cache(plan['cache_contract_sha256']),rank,output,a.max_seconds)
    finder._validate(None)
    dummy=DataLoader(torch.utils.data.TensorDataset(torch.arange(a.iterations),torch.arange(a.iterations)),batch_size=1)
    stopped=None
    try:
        finder.range_test(dummy,val_loader=dummy,start_lr=a.start_lr,end_lr=a.end_lr,num_iter=a.iterations,
                          step_mode='exp',smooth_f=.05,diverge_th=5,accumulation_steps=1)
    except (FloatingPointError,TimeoutError) as error: stopped=str(error)
    finally:
        # Restore adapters/optimizer for verification, then exit the disposable process.
        finder.reset()
        restored=wrapped.state_dict(); original=finder.state_cacher.retrieve('model')
        demand(all(torch.equal(restored[k],original[k]) for k in original),'Finder did not restore adapter state')
    if rank==0:
        atomic_json(output/'history.json',{'records':finder.records,'running':False,'updated_at':time.time(),
                    'upstream_history':finder.history,'before':finder.before,'stop_reason':stopped,
                    'calibration_discarded':True,'restoration_verified':True})
        atomic_json(output/'completed.json',{'steps':len(finder.records),'stop_reason':stopped,
                    'calibration_discarded':True,'full_training_started':False})
    dist.destroy_process_group()


if __name__=='__main__':
    with lease_data(DATA): main()
