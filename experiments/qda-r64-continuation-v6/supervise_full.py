"""Bounded restart of recoverable worker failures; data and quality failures stop."""
import json,os,pathlib,signal,subprocess,time
from contracts import atomic_json
HERE=pathlib.Path(__file__).resolve().parent
PY='/data/work/asr-pilot/venv/bin/python'

def classify(message,code):
    text=message.lower()
    fatal=('less than 8 gib','no space left','crop duration','nonfinite','hash mismatch','identity changed','contract is stale','runtime repair',
           'run config changed','resume checkpoint','quarantined','quality guard','regressed beyond','does not contain the requested crop')
    if any(s in text for s in fatal):return False
    recoverable=('cuda out of memory','connection reset','connectionerror','connectionreseterror','temporarily unavailable',
                 'connection timed out','nccl error','nccl watchdog','watchdog timeout','exitcode  : -9','exitcode: -9')
    return code in (-9,137) or any(s in text for s in recoverable)

def read(path):return json.loads(path.read_text()) if path.exists() else {}

def main():
    run=HERE/'runs/r64-full';logs=[HERE/'full-training.log',HERE/'logs/train-full.log']
    offsets={p:p.stat().st_size if p.exists() else 0 for p in logs}
    started=time.time();idle_checks=0;forced=False
    child=subprocess.Popen([PY,'-u','run_full.py'],cwd=HERE,start_new_session=True)
    while child.poll() is None:
        try:child.wait(timeout=60);break
        except subprocess.TimeoutExpired:pass
        status=read(HERE/'experiment-status.json');events=run/'events.jsonl'
        progress_at=events.stat().st_mtime if events.exists() else started
        age=time.time()-progress_at;attempt_age=time.time()-max(started,progress_at)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        loads=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit()]
        stalled=status.get('stage')=='train-full' and attempt_age>1200 and len(loads)==2 and max(loads)==0
        idle_checks=idle_checks+1 if stalled else 0
        atomic_json(HERE/'health-status.json',{'checked_at':time.time(),'running':True,'stage':status.get('stage'),
                    'seconds_since_progress':age,'gpu_load_percent':loads,'idle_checks':idle_checks})
        if idle_checks>=3:
            forced=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.returncode
    if code==0 and (run/'monitor-completed.json').exists():
        atomic_json(HERE/'health-status.json',{'checked_at':time.time(),'running':False,'stage':'completed'});return 0
    chunks=[]
    for path,offset in offsets.items():
        if path.exists():
            with path.open() as f:f.seek(offset);chunks.append(f.read()[-20000:])
    detail='\n'.join(chunks);checkpoint=read(run/'latest.json').get('step',0)
    history=read(HERE/'recovery-status.json').get('history',[])
    previous=history[-1] if history else {}
    attempt=previous.get('attempt',0)+1 if previous.get('checkpoint_step')==checkpoint else 1
    recoverable=forced or classify(detail,code)
    if (run/'stopped.json').exists():recoverable=False
    retry=recoverable and attempt<=3
    errors=[line.strip() for line in detail.splitlines() if any(x in line for x in ('RuntimeError:','Error:','Exception:','failed','Killed'))]
    reason='Watchdog: no progress and both GPUs idle for three checks after 20 minutes' if forced else (errors[0] if errors else 'Worker exited with code '+str(code))
    event={'time':time.time(),'checkpoint_step':checkpoint,'attempt':attempt,'recoverable':recoverable,'automatic_restart':retry,'reason':reason}
    history.append(event);atomic_json(HERE/'recovery-status.json',{'history':history,'latest':event})
    atomic_json(HERE/'experiment-status.json',{'stage':'automatic-recovery' if retry else 'needs-attention','running':False,
                'reason':reason,'checkpoint_step':checkpoint,'updated_at':time.time(),'automatic_restart':retry})
    atomic_json(HERE/'health-status.json',{'checked_at':time.time(),'running':False,'alert':True,**event})
    print(json.dumps({'event':'training_failure',**event}),flush=True)
    return 1 if retry else 78

if __name__=='__main__':raise SystemExit(main())
