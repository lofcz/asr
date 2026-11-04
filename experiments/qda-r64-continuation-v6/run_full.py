"""Continue the completed r64 adapter through every remaining interval once."""
import fcntl, json, os, pathlib, subprocess, time
from contracts import sha,atomic_json
from gates import demand,lease_data,check_data
from prepare import DATA,require_old_jobs_stopped
from schedule import require_decision
from evaluation import evaluate,markers,run,PY,MONITOR
from final_report import compare_all
HERE=pathlib.Path(__file__).resolve().parent
PARENT=HERE.parent/'qda-full-r64-v4'


def scored(rows):
    return [dict(r,output_issues=r.get('output_issues',[])) for r in rows]


def main():
    status=subprocess.run(['systemctl','--user','is-active','qda-v6-calibration.service'],capture_output=True,text=True).stdout.strip()
    demand(status in ('inactive','failed','unknown'),'LR calibration is still active')
    spec=json.loads((HERE/'run-spec.json').read_text());require_decision(spec)
    require_old_jobs_stopped();check_data(DATA,reviewed=True)
    plan=json.loads((HERE/'plan.json').read_text())
    from runtime_state import runtime_hashes
    ready=json.loads((HERE/'launch-receipt.json').read_text())
    demand(ready['runtime_sha256']==runtime_hashes() and ready['plan_sha256']==sha(HERE/'plan.json') and
           ready['run_spec_sha256']==sha(HERE/'run-spec.json'),'Launch runtime/data/config changed')
    out=HERE/'runs/r64-full';out.mkdir(parents=True,exist_ok=True)
    demand(not (out/'stopped.json').exists(),'Prior quality guard stopped this run; inspect evidence')
    baseline=scored([json.loads((PARENT/'evaluations/vanilla'/r['window_id']/'result.json').read_text()) for r in plan['validation']])
    parent=[json.loads((PARENT/'evaluations/final-audit/r64-full-final'/r['window_id']/'result.json').read_text()) for r in plan['validation']]
    # Reuse baseline marker output; the marker helper validates all provenance.
    import shutil
    if not (HERE/'markers/vanilla/completed-v3.json').exists():
        shutil.copytree(PARENT/'markers/vanilla',HERE/'markers/vanilla',dirs_exist_ok=True)
    baseline_markers=markers('vanilla')
    parent_markers=json.loads((PARENT/'markers/r64-full-final/completed-v3.json').read_text())
    with (HERE/'wandb-full.log').open('a') as log:
        monitor=subprocess.Popen([MONITOR,'wandb_sync.py',str(out)],stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
        try:
            def report(name,adapter,step,hours,small):
                current=scored(evaluate(name,adapter,small=small))
                marker=markers(name,adapter)
                ids={r['window_id'] for r in current}
                base=[r for r in baseline if r['window_id'] in ids]
                previous=[r for r in parent if r['window_id'] in ids]
                result=compare_all(base,current)
                result.update(markers=marker,audio_hours_seen=hours,step=step,
                    cumulative_audio_hours_seen=spec['parent_audio_hours']+hours,
                    parent_comparison=compare_all(previous,current),
                    output_issues=[{'window_id':r['window_id'],'language':r['language'],'issues':r['output_issues']} for r in current if r['output_issues']],
                    report_complete=small,per_recording=[{'window_id':r['window_id'],'language':r['language'],'metrics':r['metrics']} for r in current])
                if not small:
                    panel={r['window_id'] for r in plan['validation_small']}
                    result['trajectory_panel']=compare_all([r for r in base if r['window_id'] in panel],[r for r in current if r['window_id'] in panel])
                atomic_json(HERE/(name.replace('-final','')+'-result.json'),result)
                event={'event':'decoded_validation','step':step,'time':time.time(),
                    **{'development/'+k:v for k,v in result['finetuned'].items() if isinstance(v,(int,float))},
                    **{f'marker/{l}/{k}':v for l,m in marker['languages'].items() for k,v in m.items() if isinstance(v,(int,float))}}
                with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
                return result
            while True:
                pending=out/'evaluation-due.json'
                if pending.exists():
                    due=json.loads(pending.read_text());adapter=out/due['checkpoint'];name=f'r64-full-{due["boundary_hours"]}h'
                    result=report(name,adapter,due['step'],due['hours'],True)
                    current=result['finetuned'];start=result['parent_comparison']['vanilla']
                    healthy=(current['number_normalized_wer']<=start['number_normalized_wer']*1.15 and
                             current['der_250ms']<=start['der_250ms']+.01 and
                             all(result['markers']['languages'][l]['number_normalized_wer']<=parent_markers['languages'][l]['number_normalized_wer']+.005 for l in ('cs','sk')))
                    if not healthy:
                        atomic_json(out/'stopped.json',{'reason':'Periodic WER/DER regressed beyond limits relative to parent checkpoint','evaluation':name})
                        raise RuntimeError('Quality guard: continuation regressed beyond parent limits')
                    pending.unlink()
                if (out/'completed.json').exists():break
                demand(not (out/'stopped.json').exists(),'Training loss guard stopped the run')
                run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_restart.py',
                     '--optimizer','adamw','--stage','full','--rank','64','--lr',str(spec['learning_rate']),'--name','r64-full'],'train-full')
            done=json.loads((out/'completed.json').read_text())
            adapter=out/json.loads((out/'latest.json').read_text())['checkpoint']
            result=report('r64-full-final',adapter,done['steps'],done['training_audio_hours'],False)
            from long_evaluation import evaluate_final
            long_result=evaluate_final(adapter)
            result['long_sessions']=long_result
            result['report_complete']=True
            result['strict_acceptance_passed']=result['strict_acceptance_passed'] and long_result['strict_acceptance_passed']
            atomic_json(HERE/'r64-full-result.json',result)
            event={'event':'decoded_validation','step':done['steps'],'time':time.time(),
                   **{'long_sessions/'+k:v for k,v in long_result['finetuned'].items() if isinstance(v,(int,float))}}
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
            atomic_json(out/'monitor-completed.json',{'full_run_complete':True,'evaluation_report_complete':True,
                        'strict_acceptance_passed':result['strict_acceptance_passed'],**result['finetuned']})
            atomic_json(HERE/'experiment-status.json',{'stage':'completed_with_output_defect' if not result['strict_acceptance_passed'] else 'completed','running':False,'updated_at':time.time()})
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        finally:
            if monitor.poll() is None:monitor.terminate();monitor.wait()


if __name__=='__main__':
    with (HERE.parent/'qda-restart-v3/gpu-work.lock').open('a') as lock,lease_data(DATA):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
