import json,pathlib,statistics,time,subprocess,shutil
ROOT=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
def read_events(root=ROOT):
 result=[]
 for line in (root/'events.jsonl').read_text().splitlines():
  try:result.append(json.loads(line))
  except json.JSONDecodeError:continue
 return result

def summarize(root=ROOT,events=None,now=None):
 now=time.time() if now is None else now;events=read_events(root) if events is None else events
 config=json.loads((root/'config.json').read_text());plan=json.loads((root/'plan.json').read_text());steps={};validation={};rollbacks=[]
 for e in events:
  if e['event']=='step_start' and steps and e['step']<=max(steps):
   rollbacks.append({'time':e['time'],'from':max(steps),'to':e['step']-1})
   steps={k:v for k,v in steps.items() if k<e['step']};validation={k:v for k,v in validation.items() if k<e['step']}
  if e['event']=='step':steps[e['step']]=e
  if e['event']=='validation':validation[e['step']]=e
 current=max(steps,default=0);history=[];rates={600:[],1800:[],3600:[]}
 migration=root/'optimization-migration.json';optimized_after=json.loads(migration.read_text())['checkpoint_step'] if migration.exists() else -1
 for step,e in sorted(steps.items()):
  ranks=[r for r in e['ranks'] if r.get('active',True)];n=sum(r['target_tokens'] for r in ranks);wall=max(r['wall_seconds'] for r in ranks);secs=sum(r['seconds'] for r in ranks)
  pair=plan[(step-1)*2:step*2];bucket=max(r['bucket'] for r in pair);duration=max(r['seconds'] for r in pair)
  if duration>=bucket*.6 and step>optimized_after:rates[bucket].append(wall/(duration/bucket))
  history.append({'token_balance':min(r['target_tokens'] for r in ranks)/max(r['target_tokens'] for r in ranks),'timing':[r.get('timing',{}) for r in ranks],'step':step,'time':e['time'],'loss':sum(r['loss']*r['target_tokens'] for r in ranks)/max(n,1),'grad_norm':max(r['grad_norm'] for r in ranks),'wall_seconds':wall,'audio_hours':secs/3600,'realtime_factor':secs/wall,'target_tokens_per_second':n/wall,'lr':e['lr'],'bucket_minutes':bucket//60,'gpu_peak_gib':max(r['peak_gpu_gib'] for r in ranks)})
 costs={b:statistics.median(v[-40:]) if len(v)>=5 else {600:25,1800:65,3600:165}[b] for b,v in rates.items()};eta=0;remaining={600:0,1800:0,3600:0}
 for i in range(current*2,len(plan),2):
  pair=plan[i:i+2];bucket=max(r['bucket'] for r in pair);remaining[bucket]+=1;eta+=max(r['seconds'] for r in pair)/bucket*costs[bucket]+3
 eta+=max(0,config['steps']-current)/25*35
 completed=(root/'completed.json').exists()
 if completed:eta=0
 latest=json.loads((root/'latest.json').read_text()) if (root/'latest.json').exists() else {}
 vals=[validation[k] for k in sorted(validation)]
 return {'optimized_after':optimized_after,'generated_at':now,'config':config,'step':current,'total_steps':config['steps'],'completed':completed,'hours_done':sum(r['seconds'] for r in plan[:current*2])/3600,'hours_total':sum(r['seconds'] for r in plan)/3600,'eta_seconds':eta,'eta_low_seconds':eta*.8,'eta_high_seconds':eta*1.25,'finish_at':now+eta,'eta_basis':{'seconds_per_full_pair':costs,'measured_buckets':[b for b,v in rates.items() if len(v)>=5],'remaining_pairs':remaining,'note':'Long stages use prior hardware benchmarks until five matching live steps; range is an estimate, not a confidence interval.'},'history':history,'validation':vals,'checkpoint':latest,'rollbacks':rollbacks,'event_age_seconds':now-events[-1]['time'] if events else None,'last_loss':history[-1]['loss'] if history else None,'validation_loss':vals[-1]['loss'] if vals else None}

def telemetry():
 try:
  output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5)
  gpus=[dict(zip(['index','utilization','memory_used_mib','memory_total_mib','temperature','power_w'],map(float,row.split(',')))) for row in output.strip().splitlines()]
 except Exception:gpus=[]
 try:status=subprocess.check_output(['systemctl','--user','is-active','qda496-training'],text=True,timeout=3).strip()
 except subprocess.CalledProcessError as e:status=e.output.strip()
 except Exception:status='unknown'
 disk=shutil.disk_usage('/data');return {'gpus':gpus,'service':status,'disk_free_gib':disk.free/2**30}

def event_metrics(e,index):
 d={'event/index':index,'train/update':e.get('step',0),'event/unix_time':e['time']}
 if e['event']=='step':
  rr=[r for r in e['ranks'] if r.get('active',True)];n=sum(r['target_tokens'] for r in rr);wall=max(r['wall_seconds'] for r in rr)
  d.update({'train/loss':sum(r['loss']*r['target_tokens'] for r in rr)/max(n,1),'train/grad_norm':max(r['grad_norm'] for r in rr),'train/learning_rate':e['lr'],'throughput/step_seconds':wall,'throughput/audio_realtime_factor':sum(r['seconds'] for r in rr)/wall,'throughput/target_tokens_per_second':n/wall})
  d['throughput/token_balance']=min(r['target_tokens'] for r in rr)/max(r['target_tokens'] for r in rr)
  for r in rr:
   d.update({f"gpu{r['rank']}/peak_allocated_gib":r['peak_gpu_gib'],f"gpu{r['rank']}/loss":r['loss']})
   for key,value in r.get('timing',{}).items():d[f"gpu{r['rank']}/{key}"]=value
 elif e['event']=='validation':d.update({'validation/pseudo_label_loss':e['loss'],'validation/target_tokens':e['target_tokens']})
 elif e['event']=='checkpoint':d['checkpoint/update']=e['step']
 return d
