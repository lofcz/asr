"""Publish numeric calibration metrics only; durable local history is canonical."""
import json, os, pathlib, time
ROOT=pathlib.Path(__file__).resolve().parent/'result'
env=pathlib.Path('/home/mstagl/.config/asr-monitor/wandb.env')
for line in env.read_text().splitlines():
    key,sep,value=line.partition('=')
    if sep and key.strip().startswith('WANDB_'):os.environ[key.strip()]=value.strip().strip('"').strip("'")
import wandb
while not (ROOT/'config.json').exists():time.sleep(2)
config=json.loads((ROOT/'config.json').read_text())
run=wandb.init(project=os.environ.get('WANDB_PROJECT','vibevoice-qda'),entity=os.environ.get('WANDB_ENTITY') or None,
    id='qda-r64-lr-range-v1',name='Rank64 LR range test',group='qda-full-r64-v4',resume='allow',dir=str(ROOT),
    config=config,tags=['lr-finder','rank64','discarded-calibration'])
run.define_metric('calibration/*',step_metric='optimizer_step')
(ROOT/'wandb-status.json').write_text(json.dumps({'url':run.url,'id':run.id})+'\n')
cursor=0
while True:
    path=ROOT/'history.json'
    if path.exists():
        history=json.loads(path.read_text())
        for row in history['records']:
            if row['step']<=cursor:continue
            run.log({'optimizer_step':row['step'],**{'calibration/'+k:v for k,v in row.items() if isinstance(v,(int,float)) and k!='step'}})
            cursor=row['step']
        if not history['running']:
            run.summary['calibration_discarded']=history.get('calibration_discarded',False);break
    time.sleep(5)
run.finish()
