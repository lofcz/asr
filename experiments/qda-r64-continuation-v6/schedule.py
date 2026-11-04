"""Only an explicit, evidence-linked scheduler decision can launch the full run."""
import json, math, pathlib
from contracts import sha, atomic_json
from gates import demand
HERE=pathlib.Path(__file__).resolve().parent


def settings(spec,steps):
    demand(isinstance(steps,int) and steps>0,'Scheduler requires a positive optimizer-step budget')
    scheduler=spec.get('scheduler')
    demand(isinstance(scheduler,dict),'Full-run scheduler is unset; calibration must be reviewed first')
    demand(scheduler.get('name') in ('linear','cosine','constant','constant_with_warmup'),
           'Only the explicitly selected standard Hugging Face scheduler is supported')
    warmup=scheduler.get('warmup_ratio')
    demand(isinstance(warmup,(int,float)) and 0<=warmup<1,'Explicit warmup ratio required')
    warmup_steps=math.ceil(steps*warmup)
    demand(warmup_steps<steps,'Warmup would consume the entire training budget')
    return scheduler['name'],warmup_steps


def make_scheduler(optimizer,spec,steps):
    from transformers import get_scheduler
    name,warmup=settings(spec,steps)
    return get_scheduler(name,optimizer,num_warmup_steps=warmup,num_training_steps=steps)


def require_decision(spec):
    demand(isinstance(spec.get('learning_rate'),(int,float)) and spec['learning_rate']>0,'Full-run LR is not finalized')
    settings(spec,100)
    evidence=spec.get('lr_evidence')
    demand(isinstance(evidence,dict) and evidence.get('rationale'),'Measured LR decision is missing')
    result=pathlib.Path(evidence['history_path'])
    demand(sha(result)==evidence['history_sha256'],'LR range-test evidence changed')
    history=json.loads(result.read_text())
    demand(not history['running'] and history.get('restoration_verified'),'LR calibration is incomplete')


def write_preview(root=HERE):
    """Sample the same HF scheduler as training; never load model weights."""
    import torch
    import transformers
    root=pathlib.Path(root)
    spec=json.loads((root/'run-spec.json').read_text())
    plan_path=root/'plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else None
    # Before tokenization, show a normalized preview, not an invented run length.
    steps=len(plan['probe_pairs']) if plan else 1000
    name,warmup=settings(spec,steps)
    param=torch.nn.Parameter(torch.zeros(1))
    optimizer=torch.optim.AdamW([param],lr=spec['learning_rate'],weight_decay=0.)
    scheduler=make_scheduler(optimizer,spec,steps)
    hours=0.
    points=[{'completed_steps':0,'progress':0.,'lr':optimizer.param_groups[0]['lr'],'audio_hours_seen':0. if plan else None}]
    for i in range(steps):
        optimizer.step();scheduler.step()
        if plan:hours+=sum(r['duration'] for r in plan['probe_pairs'][i])/3600
        points.append({'completed_steps':i+1,'progress':(i+1)/steps,
                       'lr':optimizer.param_groups[0]['lr'],'audio_hours_seen':hours if plan else None})
    preview={'name':name,'warmup_ratio':spec['scheduler']['warmup_ratio'],
             'peak_lr':spec['learning_rate'],'terminal_lr':points[-1]['lr'],
             'clock':'completed optimizer updates; evaluations do not advance the scheduler',
             'exact_plan':plan is not None,'optimizer_steps':steps if plan else None,
             'warmup_steps':warmup if plan else None,'audio_hours':hours if plan else None,
             'run_spec_sha256':sha(root/'run-spec.json'),'plan_sha256':sha(plan_path) if plan else None,
             'implementation':'transformers.get_scheduler','transformers_version':transformers.__version__,
             'note':'Each point is the LR available for the next update; the terminal zero is reached after the final update.',
             'points':points}
    atomic_json(root/'schedule-preview.json',preview)
    return preview


if __name__=='__main__':
    result=write_preview()
    print(json.dumps({k:v for k,v in result.items() if k!='points'}))
