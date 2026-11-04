"""Prepare after the old run finishes; train only on an explicit both/adamw/polora command."""
import argparse,fcntl,json,os,pathlib,shutil,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
from contracts import atomic_json,sha
from prepare_experiment import require_idle,pin_baseline
from runtime_state import runtime_hashes,evaluation_hashes
PY='/data/work/asr-pilot/venv/bin/python';MONITOR='/data/work/asr-training/monitor-venv/bin/python'

def run(cmd,log):
 with open(log,'a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=HERE)
def parallel(commands,logs):
 handles=[open(p,'a') for p in logs];children=[]
 try:
  for cmd,f in zip(commands,handles):children.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=HERE))
  failures=[]
  for p in children:
   if p.wait():failures.append(p.pid)
  if failures:raise RuntimeError('Worker failed; inspect '+', '.join(map(str,logs)))
 finally:
  for p in children:
   if p.poll() is None:p.terminate();p.wait()
  for f in handles:f.close()

def prepare():
 require_idle();assert (HERE/'plan.json').exists(),'Data/CPU plan is not ready';pin_baseline()
 audit=json.loads((HERE/'data-audit.json').read_text());assert audit['status']=='passed' and audit['plan']['sha256']==sha(HERE/'plan.json'),'Data audit is missing or stale'
 if (HERE/'ready.json').exists():
  prior=json.loads((HERE/'ready.json').read_text())
  if prior['runtime_sha256']==runtime_hashes() and prior['plan_sha256']==sha(HERE/'plan.json') and prior['baseline_sha256']==sha(HERE/'baseline/adapter_model.safetensors'):
   print('Already prepared and verified. Training has not been launched by this command.',flush=True);return
 (HERE/'logs').mkdir(exist_ok=True)
 atomic_json(HERE/'preparation-status.json',{'status':'caching_frozen_features','updated_at':time.time(),'training_started':False})
 parallel([[PY,str(HERE/'prepare_experiment.py'),'features','--rank',str(i),'--world','2'] for i in range(2)],[HERE/'logs'/f'features-{i}.log' for i in range(2)])
 atomic_json(HERE/'preparation-status.json',{'status':'gpu_preflight','updated_at':time.time(),'training_started':False})
 parallel([[PY,str(HERE/'gpu_preflight.py'),'--optimizer',o,'--gpu',str(i)] for i,o in enumerate(('adamw','polora'))],[HERE/'logs'/f'preflight-{o}.log' for o in ('adamw','polora')])
 atomic_json(HERE/'ready.json',{'status':'ready','created_at':time.time(),'plan_sha256':sha(HERE/'plan.json'),'baseline_sha256':sha(HERE/'baseline/adapter_model.safetensors'),'runtime_sha256':runtime_hashes(),'training_started':False})
 atomic_json(HERE/'preparation-status.json',{'status':'ready','updated_at':time.time(),'training_started':False})
 print('READY: feature cache and both longest-window GPU preflights passed; training has not started.',flush=True)

def evaluate(adapter,name):
 out=HERE/'evaluations'/name;out.mkdir(parents=True,exist_ok=True);plan=json.loads((HERE/'plan.json').read_text())
 parallel([[PY,str(HERE/'evaluate_panel.py'),'--adapter',str(adapter),'--out',str(out),'--rank',str(i)] for i in range(2)],[HERE/'logs'/f'eval-{name}-{i}.log' for i in range(2)])
 rows=[json.loads((out/r['window_id']/'result.json').read_text()) for r in plan['validation']]
 if not all(r.get('complete') and r['adapter_sha256']==sha(adapter/'adapter_model.safetensors') for r in rows):raise RuntimeError('Evaluation incomplete; refusing partial ranking')
 def aggregate(rs):
  m=[r['metrics'] for r in rs];words=sum(x['number_normalized_reference_words'] for x in m);speech=sum(float(x['der_details']['scored_speaker_time']) for x in m)
  return {'number_normalized_wer':sum(x['number_normalized_errors'] for x in m)/words,'number_normalized_errors':sum(x['number_normalized_errors'] for x in m),'reference_words':words,'wer':sum(x['substitutions']+x['deletions']+x['insertions'] for x in m)/sum(x['reference_words'] for x in m),'der_250ms':sum(sum(float(x['der_details'][k]) for k in ('missed_speaker_time','falarm_speaker_time','speaker_error_time')) for x in m)/speech,'reference_quality':'Soniox pseudo-reference; not human gold','windows':len(rs),'sessions':len({r['record_id'] for r in rs})}
 long_rows=[json.loads((out/r['window_id']/'result.json').read_text()) for r in plan['long_validation']]
 if not all(r.get('complete') and r['adapter_sha256']==sha(adapter/'adapter_model.safetensors') for r in long_rows):raise RuntimeError('Long-session evaluation incomplete')
 summary={'long_form':{'aggregate':aggregate(long_rows),'sessions':[{'language':r['language'],'duration':r['duration'],'metrics':r['metrics']} for r in long_rows]},'aggregate':aggregate(rows),'languages':{l:aggregate([r for r in rows if r['language']==l]) for l in ('cs','sk')},'adapter_sha256':sha(adapter/'adapter_model.safetensors')}
 summary['evaluation_sha256']=evaluation_hashes()
 atomic_json(out/'summary.json',summary)
 # Existing approved gold panel remains a separate guardrail; baseline final results are reused when available.
 gold=out/'gold';gold.mkdir(exist_ok=True);ad=gold/'adapter';ad.mkdir(exist_ok=True)
 for f in ('adapter_model.safetensors','adapter_config.json'):
  if (ad/f).exists():assert sha(ad/f)==sha(adapter/f),'Gold adapter changed under an existing evaluation'
  else:shutil.copy2(adapter/f,ad/f)
 source_step=json.loads((HERE/'baseline/provenance.json').read_text())['step']
 old=pathlib.Path('/data/work/asr-evaluation/periodic/runs')/f'checkpoint-{source_step:06d}'
 reuse=name=='baseline' and (old/'completed.json').exists() and (old/'adapter/adapter_model.safetensors').exists() and sha(old/'adapter/adapter_model.safetensors')==summary['adapter_sha256']
 if reuse:shutil.copytree(old/'finetuned',gold/'finetuned',dirs_exist_ok=True)
 else:
  envs=[];cmds=[]
  for i,lang in enumerate(('cs','sk')):
   cmds.append(['env',f'CUDA_VISIBLE_DEVICES={i}',PY,'/data/work/asr-evaluation/periodic/code/evaluate.py','--variant','finetuned','--out',str(gold),'--language',lang,'--step',str(source_step),'--hours','0'])
  parallel(cmds,[HERE/'logs'/f'gold-{name}-{l}.log' for l in ('cs','sk')])
 metrics=[json.loads((gold/'finetuned'/l/'result.json').read_text()) for l in ('cs','sk')]
 if not all(r.get('termination')=='eos' and 'der_250ms' in r.get('metrics',{}) for r in metrics):raise RuntimeError('Gold evaluation failed')
 # Old evaluator includes number-normalized scoring in current deployed version.
 gm=[r['metrics'] for r in metrics];den=sum(m['number_normalized_reference_words'] for m in gm);speech=sum(float(m['der_details']['scored_speaker_time']) for m in gm)
 summary['gold']={'number_normalized_wer':sum(m['number_normalized_errors'] for m in gm)/den,'der_250ms':sum(sum(float(m['der_details'][k]) for k in ('missed_speaker_time','falarm_speaker_time','speaker_error_time')) for m in gm)/speech,'reference_words':den,'note':'Two user-approved generated dialogue fixtures; anecdotal only'}
 atomic_json(out/'summary.json',summary);return summary

def train(optimizer,stage,lr,name):
 out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():
  config=json.loads((out/'config.json').read_text());done=json.loads((out/'completed.json').read_text())
  assert config['optimizer']==optimizer and config['stage']==stage and config['lr']==lr
  assert config['plan_sha256']==sha(HERE/'plan.json') and config['runtime_sha256']==runtime_hashes()
  assert config['baseline_sha256']==sha(HERE/'baseline/adapter_model.safetensors') and done['config_sha256']==sha(out/'config.json')
  return out
 run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(HERE/'train_continuation.py'),'--optimizer',optimizer,'--stage',stage,'--lr',str(lr),'--name',name],HERE/'logs'/(name+'.log'))
 if not (out/'completed.json').exists():raise RuntimeError('Continuation stopped by quality guard: '+name)
 return out

def option(optimizer):
 choices={'adamw':[3e-5,1e-5],'polora':[.001,.0003]};selected=None
 for lr in choices[optimizer]:
  name=f'{optimizer}-cal-{lr:g}';out=train(optimizer,'calibration',lr,name)
  events=[json.loads(l) for l in (out/'events.jsonl').read_text().splitlines()];vals=[e for e in events if e['event']=='validation'];first,last=vals[0],vals[-1]
  passed=last['content_loss']<=first['content_loss']*1.10 and last['loss']<=first['loss']*1.10
  atomic_json(out/'calibration-decision.json',{'passed':passed,'lr':lr,'purpose':'Safety screen only, not optimum search','content_loss_ratio':last['content_loss']/first['content_loss']})
  if passed:selected=lr;break
 if selected is None:raise RuntimeError('Both bounded LR safety probes rejected: '+optimizer)
 atomic_json(HERE/f'{optimizer}-selected-lr.json',{'lr':selected,'rule':'First safe candidate; same baseline, fresh optimizer, 6 updates; no grid search'})
 name=optimizer+'-probe';out=HERE/'runs'/name;out.mkdir(parents=True,exist_ok=True)
 # Sidecar waits for config/events and streams metrics during training.
 log=(HERE/'logs'/(name+'-wandb.log')).open('a');monitor=subprocess.Popen([MONITOR,str(HERE/'wandb_sync.py'),str(out)],stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
 try:
  out=train(optimizer,'probe',selected,name)
  adapter=out/json.loads((out/'latest.json').read_text())['checkpoint'];result=evaluate(adapter,name);base=json.loads((HERE/'evaluations/baseline/summary.json').read_text())
  result['delta']={k:result['aggregate'][k]-base['aggregate'][k] for k in ('number_normalized_wer','der_250ms')}
  result['eligible_for_manual_extension']=result['delta']['number_normalized_wer']<=-.001 and result['aggregate']['der_250ms']<=base['aggregate']['der_250ms']+.0025 and result['gold']['number_normalized_wer']<=base['gold']['number_normalized_wer']+.001 and result['long_form']['aggregate']['number_normalized_wer']<=base['long_form']['aggregate']['number_normalized_wer']+.001 and result['long_form']['aggregate']['der_250ms']<=base['long_form']['aggregate']['der_250ms']+.0025
  atomic_json(HERE/(optimizer+'-result.json'),result);atomic_json(out/'monitor-completed.json',{'development/'+k:v for k,v in result['aggregate'].items() if isinstance(v,(int,float))}|{'gold/'+k:v for k,v in result['gold'].items() if isinstance(v,(int,float))}|{'eligible_for_manual_extension':result['eligible_for_manual_extension']})
  code=monitor.wait(timeout=120)
  if code:print('W&B sidecar exited with code '+str(code)+'; durable metrics remain in '+str(out/'events.jsonl'),flush=True)
 finally:
  if monitor.poll() is None:monitor.terminate();monitor.wait()
  log.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['status','prepare','both','adamw','polora']);p.add_argument('--wait',action='store_true');args=p.parse_args()
 if args.command=='status':
  print(json.dumps({n:json.loads((HERE/n).read_text()) if (HERE/n).exists() else None for n in ('data-ready.json','preparation-status.json','ready.json','adamw-result.json','polora-result.json')},indent=2));return
 lock=(HERE/'gpu-work.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if args.wait:
  previous=None
  while True:
   try:require_idle();assert (HERE/'plan.json').exists();break
   except (RuntimeError,AssertionError) as error:
    reason=str(error) or 'Data/CPU plan is not ready'
    if reason!=previous:
     atomic_json(HERE/'preparation-status.json',{'status':'waiting','reason':reason,'updated_at':time.time(),'training_started':False});print('Waiting: '+reason,flush=True);previous=reason
    time.sleep(30)
 if args.command=='prepare':prepare();return
 require_idle()
 if not (HERE/'ready.json').exists():prepare()
 ready=json.loads((HERE/'ready.json').read_text());assert ready['plan_sha256']==sha(HERE/'plan.json') and ready['baseline_sha256']==sha(HERE/'baseline/adapter_model.safetensors');assert ready['runtime_sha256']==runtime_hashes(),'Runtime changed; repeat prepare'
 baseline_summary=HERE/'evaluations/baseline/summary.json'
 if not baseline_summary.exists() or 'gold' not in json.loads(baseline_summary.read_text()) or json.loads(baseline_summary.read_text()).get('evaluation_sha256')!=evaluation_hashes():evaluate(HERE/'baseline','baseline')
 for optimizer in (('adamw','polora') if args.command=='both' else (args.command,)):option(optimizer)
 print('Completed requested bounded experiments. No extension was launched.',flush=True)
if __name__=='__main__':main()
