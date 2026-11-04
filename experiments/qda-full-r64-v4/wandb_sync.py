"""Metrics-only sidecar; never sends transcripts, audio, or credentials as artifacts."""
import argparse,json,os,pathlib,time
from contracts import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=pathlib.Path);args=p.parse_args();root=args.run
 while not (root/'config.json').exists() or not (root/'events.jsonl').exists():time.sleep(2)
 config=json.loads((root/'config.json').read_text())
 env=pathlib.Path('/home/mstagl/.config/asr-monitor/wandb.env')
 if env.exists():
  for line in env.read_text().splitlines():
   k,sep,v=line.partition('=')
   if sep and k.strip().startswith('WANDB_'):os.environ[k.strip()]=v.strip().strip('"').strip("'")
 import wandb
 mode='online' if os.environ.get('WANDB_API_KEY') else 'offline'
 run=wandb.init(project=os.environ.get('WANDB_PROJECT','vibevoice-qda'),entity=os.environ.get('WANDB_ENTITY') or None,id='qda-v4-'+root.name,name='Clean restart · '+root.name,group='qda-full-r64-v4',config=config,dir=str(root),mode=mode,resume='allow' if mode=='online' else None,tags=['cs-sk','rank'+str(config['rank']),'from-vanilla','audited-data'],notes='Full corpus rank64 from vanilla, measured LR decision, fixed alpha32. Held-out interviews with Soniox pseudo-references and both CS/SK fixed markers reported separately. No audio/text artifacts uploaded.')
 for prefix in ('train','validation','validation_full'):run.define_metric(prefix+'/*',step_metric='optimizer_step')
 atomic_json(root/'wandb-status.json',{'mode':mode,'url':run.url,'id':run.id})
 cursor_path=root/'wandb-cursor.json';cursor=json.loads(cursor_path.read_text())['offset'] if cursor_path.exists() and mode=='online' else 0
 while True:
  with (root/'events.jsonl').open() as f:
   f.seek(cursor)
   while True:
    line=f.readline()
    if not line:break
    if not line.endswith('\n'):break
    e=json.loads(line);kind=e['event'];values={'optimizer_step':e.get('step',0)}
    if kind=='step':
     values.update({'train/'+k:v for k,v in e.items() if isinstance(v,(int,float)) and k not in ('time','step')})
    elif kind=='validation':
     prefix='validation_full' if e.get('full_panel') else 'validation';values.update({prefix+'/'+k:v for k,v in e.items() if isinstance(v,(int,float)) and k not in ('time','step')})
    elif kind=='decoded_validation':
     values.update({k:v for k,v in e.items() if isinstance(v,(int,float)) and k not in ('time','step')})
    else:cursor=f.tell();continue
    run.log(values);cursor=f.tell();atomic_json(cursor_path,{'offset':cursor})
  terminal=root/'monitor-completed.json'
  if terminal.exists():
   summary=json.loads(terminal.read_text());run.summary.update(summary);break
  time.sleep(5)
 run.finish()
if __name__=='__main__':main()
