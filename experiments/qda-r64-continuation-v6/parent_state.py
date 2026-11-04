"""Warm start weights and Adam moments; reset only the continuation LR schedule."""
import json
from pathlib import Path
from contracts import sha
from gates import demand
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'qda-full-r64-v4/runs/r64-full/checkpoint-001314'


def retarget_optimizer(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr']=lr
        group['initial_lr']=lr


def restore_parent(model, optimizer, rank, lr, restore_rng=True):
    import random, numpy as np, torch
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    from safetensors.torch import load_file
    provenance=json.loads((Path('/data/work/asr-datasets/qda-unseen-v6')/'provenance.json').read_text())
    for name in ('adapter_config.json','adapter_model.safetensors','state.pt'):
        demand(sha(PARENT/name)==provenance['parent_checkpoint_files'][name],'Parent checkpoint changed: '+name)
    state=torch.load(PARENT/'state.pt',map_location='cpu',mmap=True,weights_only=False)
    demand(state['step']==1314 and state['config']['rank']==64,'Unexpected parent checkpoint')
    saved=load_file(str(PARENT/'adapter_model.safetensors'))
    current=get_peft_model_state_dict(model)
    demand(set(current)==set(saved) and all(current[k].shape==saved[k].shape for k in saved),'Parent adapter architecture mismatch')
    set_peft_model_state_dict(model,saved)
    demand(all(torch.equal(v.detach().cpu(),saved[k]) for k,v in get_peft_model_state_dict(model).items()),'Parent adapter restore failed')
    optimizer.load_state_dict(state['optimizer'])
    demand(bool(optimizer.state),'Parent Adam moments missing')
    # The parent finished at LR zero. Build the new HF scheduler after restoring
    # Adam's moments, with its new measured peak as both lr and initial_lr.
    retarget_optimizer(optimizer,lr)
    if restore_rng:
        rng=state['rng'][rank]
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    return {'parent_step':state['step'],'parent_tokens_seen':state['tokens_seen'],
            'adapter_sha256':state['adapter_sha256'],'optimizer_restored':True,
            'rng_restored':restore_rng,'parent_audio_hours':state['config']['data_hours']}
