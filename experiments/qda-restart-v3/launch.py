"""Bounded fresh rank comparison. Every training entry point rechecks admission."""
import argparse
import fcntl
import json
import pathlib
import shutil
import subprocess
import sys
import time
import contextlib
from contracts import atomic_json,sha
from gates import check_data,demand,lease_data,validate_reference
from prepare import DATA,require_old_jobs_stopped
from runtime_state import runtime_hashes
from analyze import compare

HERE=pathlib.Path(__file__).resolve().parent
PY='/data/work/asr-pilot/venv/bin/python'
MONITOR='/data/work/asr-training/monitor-venv/bin/python'


class TrialRejected(RuntimeError):
    """A measured trajectory rejects one capacity without cancelling its peer."""


def progress(stage,**details):
    atomic_json(HERE/'experiment-status.json',{'stage':stage,'updated_at':time.time(),'running':True,**details})


def run(command,name):
    progress(name)
    (HERE/'logs').mkdir(exist_ok=True)
    with (HERE/'logs'/(name+'.log')).open('a') as f:subprocess.run(command,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)


def parallel(commands,name):
    progress(name)
    files=[];children=[];(HERE/'logs').mkdir(exist_ok=True)
    try:
        for i,cmd in enumerate(commands):
            f=(HERE/'logs'/f'{name}-{i}.log').open('a');files.append(f);children.append(subprocess.Popen(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in children]
        demand(all(c==0 for c in codes),'Worker failed: '+name)
    finally:
        for p in children:
            if p.poll() is None:p.terminate();p.wait()
        for f in files:f.close()


def prepare():
    require_old_jobs_stopped();check_data(DATA,reviewed=True)
    run([PY,'prepare.py','inputs'],'inputs')
    parallel([[PY,'prepare.py','features','--rank',str(i)] for i in (0,1)],'features')
    parallel([[PY,'preflight.py','--rank',str(rank),'--gpu',str(i)] for i,rank in enumerate((16,64))],'preflight')
    atomic_json(HERE/'ready.json',{'plan_sha256':sha(HERE/'plan.json'),'runtime_sha256':runtime_hashes(),'admission_sha256':sha(DATA/'admission.json')})


def evaluate(name,adapter=None,small=False,stress=False):
    out=HERE/'evaluations'/name;out.mkdir(parents=True,exist_ok=True)
    flags=(['--adapter',str(adapter)] if adapter else [])+(['--small'] if small else [])+(['--stress'] if stress else [])
    parallel([[PY,'evaluate_panel.py','--out',str(out),'--rank',str(i),*flags] for i in (0,1)],'evaluation-'+name)
    plan=json.loads((HERE/'plan.json').read_text());panel=plan['long_stress'] if stress else plan['validation_small'] if small else plan['validation']
    results=[json.loads((out/r['window_id']/'result.json').read_text()) for r in panel]
    if adapter and not all(r.get('complete') for r in results):raise TrialRejected('Adapted evaluation has invalid/incomplete raw output: '+name)
    demand(all(r.get('complete') for r in results),'Evaluation invalid/incomplete; inspect raw output: '+name)
    return results


def markers(name,adapter=None):
    """Both original approved CS/SK fixtures remain visible, separately, at every trajectory checkpoint."""
    out=HERE/'markers'/name;out.mkdir(parents=True,exist_ok=True);variant='finetuned' if adapter else 'vanilla'
    if adapter:
        dest=out/'adapter';dest.mkdir(exist_ok=True)
        for filename in ('adapter_model.safetensors','adapter_config.json'):
            if (dest/filename).exists():demand(sha(dest/filename)==sha(adapter/filename),'Marker checkpoint changed')
            else:shutil.copy2(adapter/filename,dest/filename)
    stamp=out/'completed-v3.json';identity=sha(adapter/'adapter_model.safetensors') if adapter else 'vanilla'
    from reuse_markers import contract as marker_contract
    if not adapter and not stamp.exists():run([PY,'reuse_markers.py'],'reuse-vanilla-markers')
    if stamp.exists():
        saved=json.loads(stamp.read_text());demand(saved['adapter_sha256']==identity and saved['contract']==marker_contract(),'Marker identity/contract changed');return saved
    step=int(adapter.name.split('-')[-1]) if adapter else 0;hours=0.
    if adapter:
        events=[json.loads(l) for l in (adapter.parent/'events.jsonl').read_text().splitlines()]
        hours=next(e['audio_hours_seen'] for e in events if e['event']=='step' and e['step']==step)
    parallel([['env',f'CUDA_VISIBLE_DEVICES={i}',PY,'/data/work/asr-evaluation/periodic/code/evaluate.py','--variant',variant,'--out',str(out),'--language',lang,'--step',str(step),'--hours',str(hours)] for i,lang in enumerate(('cs','sk'))],'markers-'+name)
    rows={l:json.loads((out/variant/l/'result.json').read_text()) for l in ('cs','sk')}
    demand(all(r.get('termination')=='eos' and 'der_250ms' in r.get('metrics',{}) for r in rows.values()),'CS/SK marker evaluation failed')
    for lang,row in rows.items():
        raw=json.loads((out/variant/lang/'output.txt').read_text())
        demand(raw==json.loads((out/variant/lang/'hypothesis.json').read_text()),'Marker raw JSON differs from scored output')
        validate_reference(raw,row['duration_seconds'])
    provenance={l:{k:r[k] for k in ('audio_sha256','reference_sha256','feature_sha256')} for l,r in rows.items()}
    baseline=json.loads((HERE/'markers/vanilla/completed-v3.json').read_text())
    demand(provenance==baseline['provenance'],'Marker input/features differ from vanilla')
    result={'adapter_sha256':identity,'languages':{l:r['metrics'] for l,r in rows.items()},'provenance':provenance,'contract':marker_contract(),
            'step':step,'audio_hours_seen':hours,'role':'Required fixed markers; user-approved generated dialogues; report separately'}
    atomic_json(stamp,result);return result


def train(rank,stage,lr,name):
    out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'stopped.json').exists():raise TrialRejected('Training quality guard stopped '+name)
    if (out/'completed.json').exists():
        c=json.loads((out/'config.json').read_text());demand(c['rank']==rank and c['lr']==lr and c['stage']==stage and c['runtime_sha256']==runtime_hashes(),'Completed run contract changed');return out
    monitor=None;handle=None
    if stage=='calibration':
        (HERE/'logs').mkdir(exist_ok=True);handle=(HERE/'logs'/(name+'-wandb.log')).open('a')
        monitor=subprocess.Popen([MONITOR,'wandb_sync.py',str(out)],cwd=HERE,stdout=handle,stderr=subprocess.STDOUT)
    try:
        run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_restart.py','--optimizer','adamw','--rank',str(rank),'--stage',stage,'--lr',str(lr),'--name',name],name)
        if monitor and (out/'completed.json').exists():
            atomic_json(out/'monitor-completed.json',{'calibration/complete':True})
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:print('W&B sync delayed; local calibration metrics are preserved.',flush=True)
    finally:
        if monitor and monitor.poll() is None:monitor.terminate();monitor.wait()
        if handle:handle.close()
    if (out/'stopped.json').exists():raise TrialRejected('Training quality guard stopped '+name)
    return out


def option(rank,baseline,baseline_markers):
    stopped=HERE/f'r{rank}-probe-stopped.json'
    if stopped.exists():raise TrialRejected(json.loads(stopped.read_text())['reason'])
    calibrations=[]
    for lr in (3e-5,1e-4):
        out=train(rank,'calibration',lr,f'r{rank}-cal-{lr:g}')
        demand((out/'completed.json').exists(),'Incomplete calibration')
        vals=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines() if json.loads(line)['event']=='validation']
        first,last=vals[0],vals[-1]
        if last['content_loss']<=first['content_loss']*1.05 and last['loss']<=first['loss']*1.05:calibrations.append((last['content_loss'],lr))
    if not calibrations:raise TrialRejected('Neither bounded LR probe passed; no full run')
    lr=min(calibrations)[1];atomic_json(HERE/f'lr-r{rank}.json',{'rank':rank,'lr':lr,'trials':calibrations,'selection':'lowest held-out content loss among two safe probes; fresh restart afterward'})
    if baseline is None:baseline=evaluate('vanilla')
    name=f'r{rank}-probe';out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    f=(HERE/'logs'/(name+'-wandb.log')).open('a');monitor=subprocess.Popen([MONITOR,'wandb_sync.py',str(out)],cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
    try:
        # If interrupted at/after the midpoint, score that saved checkpoint before
        # any further update. Resume can use a later durable training checkpoint.
        if not (out/'midpoint.json').exists() and not (out/'completed.json').exists():out=train(rank,'probe',lr,name)
        if not (out/'completed.json').exists():
            demand((out/'evaluation-due.json').exists(),'Unexpected training exit')
            adapter=out/json.loads((out/'midpoint.json').read_text())['checkpoint'];current=evaluate(name+'-10h',adapter,small=True);marker=markers(name+'-10h',adapter)
            ids={r['window_id'] for r in current};mid=compare([r for r in baseline if r['window_id'] in ids],current)
            mid['markers']=marker;atomic_json(HERE/(name+'-10h.json'),mid)
            healthy=mid['finetuned']['number_normalized_wer']<=mid['vanilla']['number_normalized_wer']*1.15 and mid['finetuned']['der_250ms']<=mid['vanilla']['der_250ms']+.01
            healthy=healthy and all(marker['languages'][l]['number_normalized_wer']<=baseline_markers['languages'][l]['number_normalized_wer']+.005 for l in ('cs','sk'))
            if not healthy:raise TrialRejected('10h trajectory regressed; preserving checkpoint and ending this capacity probe')
            out=train(rank,'probe',lr,name)
        demand((out/'completed.json').exists(),'Probe incomplete')
        adapter=out/json.loads((out/'latest.json').read_text())['checkpoint'];current=evaluate(name+'-20h',adapter);marker=markers(name+'-20h',adapter);stress=evaluate(name+'-long',adapter,stress=True)
        result=compare(baseline,current);result['markers']=marker;result['long_stress']=[{'record_id':r['record_id'],'duration':r['duration'],'complete':r['complete']} for r in stress]
        result['marker_deltas']={l:{k:marker['languages'][l][k]-baseline_markers['languages'][l][k] for k in ('number_normalized_wer','der_250ms')} for l in ('cs','sk')}
        result['markers_nonregressing']=all(d['number_normalized_wer']<=.001 and d['der_250ms']<=.005 for d in result['marker_deltas'].values())
        result['eligible_for_confirmatory_test']=result['screening_goal_met'] and result['markers_nonregressing'];result['automatic_scale_up']=False
        atomic_json(HERE/(name+'-result.json'),result)
        summary={'development/'+k:v for k,v in result['finetuned'].items() if isinstance(v,(int,float))}
        summary.update({f'marker/{l}/{k}':v for l,m in marker['languages'].items() for k,v in m.items() if isinstance(v,(int,float))})
        atomic_json(out/'monitor-completed.json',summary)
        try:monitor.wait(timeout=30)
        except subprocess.TimeoutExpired:print('W&B sync is delayed; durable local metrics are preserved.',flush=True)
    finally:
        if monitor.poll() is None:monitor.terminate();monitor.wait()
        f.close()
    return baseline


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','prepare','both','r16','r64']);a=p.parse_args()
    if a.command=='status':
        for name in ('ready.json','r16-probe-result.json','r64-probe-result.json'):
            print(name, (HERE/name).read_text() if (HERE/name).exists() else 'pending')
        return
    with contextlib.ExitStack() as stack:
        lock=stack.enter_context((HERE/'gpu-work.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_old_jobs_stopped();check_data(DATA,reviewed=True)
        stack.enter_context(lease_data(DATA));check_data(DATA,reviewed=True)
        if a.command=='prepare':prepare();return
        demand((HERE/'ready.json').exists(),'Run prepare after data admission passes')
        ready=json.loads((HERE/'ready.json').read_text());demand(ready['runtime_sha256']==runtime_hashes() and ready['plan_sha256']==sha(HERE/'plan.json') and ready['admission_sha256']==sha(DATA/'admission.json'),'Readiness contract stale')
        baseline=None;baseline_markers=markers('vanilla')
        for rank in ((16,64) if a.command=='both' else (int(a.command[1:]),)):
            try:baseline=option(rank,baseline,baseline_markers)
            except TrialRejected as error:
                atomic_json(HERE/f'r{rank}-probe-stopped.json',{'rank':rank,'reason':str(error),'updated_at':time.time()})
                print(f'Rank {rank} stopped: {error}',flush=True)


if __name__=='__main__':main()
