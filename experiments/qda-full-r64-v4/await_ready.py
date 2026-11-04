"""An authorized launch waits for successful preparation without occupying a GPU."""
import json, pathlib, subprocess, time
from contracts import atomic_json

HERE=pathlib.Path(__file__).resolve().parent


def preparation_complete(root,state):
    status_path=root/'preparation-status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if state in ('active','activating','deactivating'):
        return False
    if state!='inactive' or status.get('stage')!='data-and-runtime-prepared':
        raise RuntimeError('Preparation did not complete successfully: '+str(status.get('error') or status.get('stage') or state))
    if not (root/'prepared.json').exists():
        raise RuntimeError('Preparation has no completed data/runtime contract')
    return True


def main():
    while True:
        state=subprocess.run(['systemctl','--user','is-active','qda-v4-preparation.service'],capture_output=True,text=True,check=False).stdout.strip()
        if preparation_complete(HERE,state):
            atomic_json(HERE/'experiment-status.json',{'stage':'starting-full-training','running':False,'training_requested':True,'updated_at':time.time()})
            return
        atomic_json(HERE/'experiment-status.json',{'stage':'waiting-for-data-preparation','running':False,'training_requested':True,'updated_at':time.time()})
        time.sleep(10)


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(HERE/'experiment-status.json',{'stage':'launch-blocked-by-preparation','running':False,'training_requested':True,'reason':str(e),'updated_at':time.time()})
        raise
