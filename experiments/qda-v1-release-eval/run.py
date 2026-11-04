"""Durable, restartable two-GPU evaluation. Never starts a training process."""
import fcntl,json,os,pathlib,signal,subprocess,time
from common import HERE,BASE,sha,atomic_json
PY='/data/work/asr-pilot/venv/bin/python'

def parallel(commands,stage):
    children=[];handles=[]
    try:
        for rank,cmd in enumerate(commands):
            log=(HERE/'logs'/f'{stage}-{rank}.log').open('a');handles.append(log)
            children.append(subprocess.Popen(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT))
        while True:
            codes=[p.poll() for p in children]
            atomic_json(HERE/'status.json',{'stage':stage,'running':True,'worker_pids':[p.pid for p in children],'exit_codes':codes,'updated_at':time.time()})
            if (HERE/'report.py').exists():
                subprocess.run([PY,'report.py'],cwd=HERE,stdout=subprocess.DEVNULL,check=True)
            if any(code not in (None,0) for code in codes):raise RuntimeError(f'{stage} failed: {codes}; inspect logs')
            if all(code is not None for code in codes):break
            time.sleep(15)
    finally:
        for process in children:
            if process.poll() is None:process.terminate()
        for process in children:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for handle in handles:handle.close()

def main():
    (HERE/'logs').mkdir(exist_ok=True)
    with (BASE.parent/'qda-restart-v3/gpu-work.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selection=json.loads((HERE/'selection.json').read_text())
        for name,spec in selection['models'].items():
            if spec['adapter']:assert sha(pathlib.Path(spec['adapter'])/'adapter_model.safetensors')==spec['adapter_sha256']
        parallel([[PY,'-u','prepare_panel.py','--rank',str(i)] for i in (0,1)],'prepare')
        for phase in ['main','long']:
            for name in ['300h','450h','final','vanilla']:
                parallel([[PY,'-u','worker.py','--rank',str(i),'--model',name,'--phase',phase] for i in (0,1)],phase+'-'+name)
        subprocess.run([PY,'report.py','--final'],cwd=HERE,check=True)
        atomic_json(HERE/'status.json',{'stage':'completed','running':False,'updated_at':time.time(),'report':str(HERE/'report.md')})

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(HERE/'status.json',{'stage':'failed','running':False,'error':str(exc),'updated_at':time.time()});raise
