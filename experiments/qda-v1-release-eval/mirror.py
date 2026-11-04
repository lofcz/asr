"""Progressively mirror every evaluation result to the external SSD."""
import json,pathlib,subprocess,time
HERE=pathlib.Path(__file__).resolve().parent
while True:
    result=subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=12 -S /home/lofcz/.cache/asr-ssh/mlserver-persistent',
        '--exclude=cache/','--exclude=adapter-450h/','--exclude=__pycache__/',
        'mlserver:/data/work/asr-experiments/qda-v1-release-eval/',str(HERE)+'/'],capture_output=True,text=True)
    status={'updated_at':time.time(),'ok':result.returncode==0,'error':result.stderr[-1000:] if result.returncode else None}
    temp=HERE/'mirror-status.building';temp.write_text(json.dumps(status)+'\n');temp.replace(HERE/'mirror-status.json')
    p=HERE/'status.json'
    if result.returncode==0 and p.exists() and json.loads(p.read_text()).get('stage')=='completed':break
    time.sleep(30)
