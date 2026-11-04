"""Keep dashboard metrics current while the independent checkpoint copier runs."""
import json,os,pathlib,subprocess,time
from contracts import atomic_json
HERE=pathlib.Path(__file__).resolve().parent
os.environ['RSYNC_RSH']='ssh -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=/home/lofcz/.cache/asr-ssh/mlserver-persistent'
TELEMETRY='''import json,subprocess,shutil,time
fields='index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
rows=[]
for line in p.stdout.splitlines():
 v=[x.strip() for x in line.split(',')]
 def num(x):
  try:return float(x)
  except ValueError:return None
 rows.append(dict(index=int(v[0]),name=v[1],utilization=num(v[2]),memory_used_mib=num(v[3]),memory_total_mib=num(v[4]),temperature_c=num(v[5]),power_w=num(v[6])))
d=shutil.disk_usage('/data/work')
print(json.dumps(dict(checked_at=time.time(),gpus=rows,disk_free_gib=d.free/2**30,disk_total_gib=d.total/2**30)))
'''
last_telemetry=0
while True:
    try:
        subprocess.run(['rsync','-az','--exclude=cache*/','--exclude=checkpoint-*/','--exclude=wandb/',
            '--exclude=backup-receipts/','--exclude=backup-retention.json','--exclude=superseded/',
            '--include=*/','--include=*.json','--include=*.jsonl','--include=output.txt','--exclude=*',
            'mlserver:/data/work/asr-experiments/qda-r64-continuation-v6/',str(HERE)+'/'],check=True,timeout=45)
        atomic_json(HERE/'live-sync-status.json',{'mirrored_at':time.time(),'status':'ok','scope':'metrics and status; checkpoint copies are independent'})
    except Exception as e:
        print(type(e).__name__+': '+str(e),flush=True)
        atomic_json(HERE/'live-sync-status.json',{'status':'error','error_at':time.time(),'error':str(e)})
    if time.time()-last_telemetry>=30:
        last_telemetry=time.time()
        try:
            import shlex
            command=shlex.split(os.environ['RSYNC_RSH'])+['mlserver','/usr/bin/python3 -']
            result=subprocess.run(command,input=TELEMETRY,text=True,capture_output=True,check=True,timeout=20)
            atomic_json(HERE/'live-telemetry.json',json.loads(result.stdout))
        except Exception as e:print('Telemetry: '+str(e),flush=True)
    time.sleep(10)
