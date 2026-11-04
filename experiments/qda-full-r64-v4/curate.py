"""Persistent preparation only. This process cannot launch full training."""
import fcntl, json, pathlib, shutil, subprocess, time
from contracts import atomic_json, sha
from gates import demand, lease_data, check_data
from prepare import DATA, require_old_jobs_stopped

HERE=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/data/work/asr-datasets/qda-clean-v3')
BASE=pathlib.Path('/data/work/asr-experiments/qda-restart-v3')
PY='/data/work/asr-pilot/venv/bin/python'


def status(stage,**values):
    atomic_json(HERE/'preparation-status.json',{'stage':stage,'updated_at':time.time(),'training_started':False,**values})


def run(args,label):
    status(label)
    with (HERE/(label+'.log')).open('a') as f:subprocess.run([PY,*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)


def parallel(commands,label):
    status(label);children=[];handles=[]
    try:
        for i,args in enumerate(commands):
            h=(HERE/f'{label}-{i}.log').open('a');handles.append(h)
            children.append(subprocess.Popen([PY,*args],cwd=HERE,stdout=h,stderr=subprocess.STDOUT))
        demand(all(p.wait()==0 for p in children),'Preparation worker failed: '+label)
    finally:
        for p in children:
            if p.poll() is None:p.terminate();p.wait()
        for h in handles:h.close()


def main():
    if (HERE/'prepared.json').exists():
        from runtime_state import runtime_hashes
        saved=json.loads((HERE/'prepared.json').read_text())
        demand(saved['runtime_sha256']==runtime_hashes() and saved['plan_sha256']==sha(HERE/'plan.json') and
               saved['admission_sha256']==sha(DATA/'admission.json'),'Prepared contract changed; inspect before rebuilding')
        status('data-and-runtime-prepared',train_hours=saved['train_hours'],schedule_status='Requires explicit evidence-linked LR/scheduler decision before launch')
        return
    status('waiting-for-structural-audit')
    while (DATA/'.building').exists() or not (DATA/'structural-audit.json').exists():
        active=subprocess.run(['systemctl','--user','is-active','qda-v4-structural.service'],capture_output=True,text=True).stdout.strip()
        demand(active in ('active','activating'),'Structural build stopped before completion')
        time.sleep(10)
    with lease_data(DATA,exclusive=True) as data_lock:
        for folder in ('acoustic','acoustic-errors'):
            (DATA/folder).mkdir(exist_ok=True)
            for source in (OLD/folder).glob('*.json'):
                dest=DATA/folder/source.name
                if not dest.exists():shutil.copy2(source,dest)
        status('waiting-for-lr-calibration')
        while subprocess.run(['systemctl','--user','is-active','qda-lr-calibration.service'],capture_output=True,text=True).stdout.strip() in ('active','activating'):
            time.sleep(3)
        # Same shared GPU lock as the capacity probes and LR finder.
        with (BASE/'gpu-work.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            require_old_jobs_stopped()
            parallel([['acoustic_audit.py','--all','--data',str(DATA),'--rank',str(i)] for i in (0,1)],'acoustic-audit')
            selected=DATA/'acoustic-selection.json'
            existing=json.loads(selected.read_text()) if selected.exists() else {}
            if not (existing.get('full_selector_sha256')==sha(HERE/'full_dataset.py') and
                    existing.get('sources_sha256')==sha(DATA/'sources.json') and
                    (DATA/'screening-selection.json').exists()):
                run(['full_dataset.py'],'select-full-corpus')
            status('rechecking-data-integrity');admission=check_data(DATA,reviewed=True)
            status('exact-input-preparation',train_hours=admission['train_hours'])
            parallel([['prepare.py','inputs','--rank',str(i),'--world','3'] for i in range(3)],'exact-inputs')
            run(['prepare.py','plan'],'freeze-full-plan')
            parallel([['prepare.py','features','--rank',str(i),'--world','2','--validation-only'] for i in (0,1)],'heldout-features')
            run(['reuse_baseline.py'],'reuse-verified-vanilla')
            # Preparation has finished mutating the dataset. Keep writers out,
            # while permitting the preflight child to acquire its own read lease.
            fcntl.flock(data_lock,fcntl.LOCK_SH)
            run(['preflight.py','--rank','64','--gpu','0'],'rank64-memory-check')
            from runtime_state import runtime_hashes
            atomic_json(HERE/'prepared.json',{'plan_sha256':sha(HERE/'plan.json'),
                'runtime_sha256':runtime_hashes(),'admission_sha256':sha(DATA/'admission.json'),
                'train_hours':admission['train_hours'],'training_started':False})
            status('data-and-runtime-prepared',train_hours=admission['train_hours'],
                   schedule_status='LR and scheduler are revalidated by the training launcher')


if __name__=='__main__':
    try:main()
    except Exception as e:status('preparation-failed',error=str(e));raise
