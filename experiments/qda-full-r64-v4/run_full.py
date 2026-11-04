"""One fresh full-corpus run, resumable across periodic evaluations. Never a sweep."""
import contextlib, fcntl, json, os, pathlib, subprocess, time
from contracts import sha,atomic_json
from gates import check_data,demand,lease_data
from prepare import DATA,require_old_jobs_stopped
from runtime_state import runtime_hashes
from schedule import require_decision
from evaluation import evaluate,markers,run,compare,PY,MONITOR
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/data/work/asr-experiments/qda-restart-v3')


def main():
    spec=json.loads((HERE/'run-spec.json').read_text());require_decision(spec)
    require_old_jobs_stopped()
    ready=json.loads((HERE/'prepared.json').read_text())
    check_data(DATA,reviewed=True)
    demand(ready['runtime_sha256']==runtime_hashes() and ready['plan_sha256']==sha(HERE/'plan.json') and
           ready['admission_sha256']==sha(DATA/'admission.json'),'Full data/runtime preparation is stale')
    preflight=json.loads((HERE/'preflight-r64.json').read_text())
    demand(preflight['status']=='passed' and preflight['updates_discarded'] and
           preflight['plan_sha256']==ready['plan_sha256'] and
           preflight['admission_sha256']==ready['admission_sha256'],'Rank64 memory check is missing or stale')
    plan=json.loads((HERE/'plan.json').read_text())
    out=HERE/'runs/r64-full';out.mkdir(parents=True,exist_ok=True)
    demand(not (out/'stopped.json').exists(),'Prior quality guard stopped this run; inspect evidence')
    baseline=[json.loads((HERE/'evaluations/vanilla'/r['window_id']/'result.json').read_text()) for r in plan['validation']]
    baseline_markers=markers('vanilla')
    with (HERE/'wandb-full.log').open('a') as log:
        monitor=subprocess.Popen([MONITOR,'wandb_sync.py',str(out)],stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
        try:
            while True:
                pending=out/'evaluation-due.json'
                if pending.exists():
                    due=json.loads(pending.read_text());adapter=out/due['checkpoint'];name=f'r64-full-{due["boundary_hours"]}h'
                    current=evaluate(name,adapter,small=True);marker=markers(name,adapter)
                    ids={r['window_id'] for r in current};result=compare([r for r in baseline if r['window_id'] in ids],current)
                    result['markers']=marker;result['audio_hours_seen']=due['hours'];result['step']=due['step']
                    atomic_json(HERE/(name+'-result.json'),result)
                    event={'event':'decoded_validation','step':due['step'],'time':time.time(),
                        **{'development/'+k:v for k,v in result['finetuned'].items() if isinstance(v,(int,float))},
                        **{f'marker/{l}/{k}':v for l,m in marker['languages'].items() for k,v in m.items() if isinstance(v,(int,float))}}
                    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
                    # Same predeclared regression guards as the short capacity screen.
                    healthy=result['finetuned']['number_normalized_wer']<=result['vanilla']['number_normalized_wer']*1.15
                    healthy=healthy and result['finetuned']['der_250ms']<=result['vanilla']['der_250ms']+.01
                    healthy=healthy and all(marker['languages'][l]['number_normalized_wer']<=baseline_markers['languages'][l]['number_normalized_wer']+.005 for l in ('cs','sk'))
                    if not healthy:
                        atomic_json(out/'stopped.json',{'reason':'Periodic evaluation regressed beyond predeclared limits','evaluation':name})
                        return
                    pending.unlink()
                if (out/'completed.json').exists():break
                demand(not (out/'stopped.json').exists(),'Training loss guard stopped the run')
                run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_restart.py',
                     '--optimizer','adamw','--stage','full','--rank','64','--lr',str(spec['learning_rate']),'--name','r64-full'],'train-full')
            adapter=out/json.loads((out/'latest.json').read_text())['checkpoint']
            current=evaluate('r64-full-final',adapter);marker=markers('r64-full-final',adapter)
            result=compare(baseline,current);result['markers']=marker
            atomic_json(HERE/'r64-full-result.json',result)
            atomic_json(out/'monitor-completed.json',{'full_run_complete':True,**result['finetuned']})
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        finally:
            if monitor.poll() is None:monitor.terminate();monitor.wait()


if __name__=='__main__':
    with (BASE/'gpu-work.lock').open('a') as lock,lease_data(DATA):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main()
            atomic_json(HERE/'experiment-status.json',{'stage':'completed' if (HERE/'r64-full-result.json').exists() else 'stopped',
                'running':False,'updated_at':time.time()})
        except Exception as e:
            atomic_json(HERE/'experiment-status.json',{'stage':'stopped','reason':str(e),'running':False,'updated_at':time.time()})
            raise
