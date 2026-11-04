"""Rerun only the two long sessions without the old 20-minute cutoff."""
import fcntl,json,pathlib,subprocess,time
from common import HERE,BASE,sha,atomic_json
from run import parallel,PY

def main():
    (HERE/'logs').mkdir(exist_ok=True)
    selection=json.loads((HERE/'selection.json').read_text())
    with (BASE.parent/'qda-restart-v3/gpu-work.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,spec in selection['models'].items():
            if spec['adapter']:
                assert sha(pathlib.Path(spec['adapter'])/'adapter_model.safetensors')==spec['adapter_sha256']
        for name in ['vanilla','300h','450h','final']:
            parallel([[PY,'-u','worker.py','--rank',str(i),'--model',name,'--phase','long',
                       '--rerun-incomplete','--unbounded-long'] for i in (0,1)],'long-unbounded-'+name)
        subprocess.run([PY,'report.py'],cwd=HERE,check=True)
        complete=all(json.loads((HERE/'evaluations'/name/r['window_id']/'result.json').read_text()).get('complete')
                     for name in selection['models'] for r in selection['long_rows'])
        if complete:subprocess.run([PY,'report.py','--final'],cwd=HERE,check=True)
        atomic_json(HERE/'long-unbounded-status.json',{'status':'completed' if complete else 'completed-with-failures',
            'all_long_outputs_complete':complete,'updated_at':time.time()})

if __name__=='__main__':main()
