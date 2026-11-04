import os,json,time,pathlib,threading,http.server,argparse,re
from metrics import ROOT,read_events,summarize,telemetry,event_metrics
BASE=pathlib.Path(__file__).resolve().parent

def web():
 snapshot={}
 def update():
  while True:
   try:
    data=summarize();data.update(telemetry())
    evaluation=pathlib.Path('/data/work/asr-evaluation/after-100h/comparison.json')
    if evaluation.exists():data['evaluation100h']=json.loads(evaluation.read_text())
    periodic=pathlib.Path('/data/work/asr-evaluation/periodic')
    if (periodic/'history.json').exists():
     data['periodic_evaluation']=json.loads((periodic/'history.json').read_text())
     data['periodic_evaluation']['policy']=json.loads((periodic/'policy.json').read_text())
     data['periodic_evaluation']['running']=(periodic/'request.json').exists()
     data['periodic_evaluation']['activation_pending']=not (periodic/'activated.json').exists()
    p=ROOT/'wandb-status.json';data['wandb']=json.loads(p.read_text()) if p.exists() else {'mode':'connecting'}
    snapshot.clear();snapshot.update(data)
   except Exception as e:snapshot['error']=str(e)
   time.sleep(10)
 threading.Thread(target=update,daemon=True).start()
 class Handler(http.server.BaseHTTPRequestHandler):
  def do_HEAD(self):self.do_GET()
  def serve_audio(self,file,kind):
   size=file.stat().st_size;start=0;end=size-1;value=self.headers.get('Range');status=200
   if value:
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
    if match and any(match.groups()):
     left,right=match.groups()
     if left:start=int(left);end=min(int(right),size-1) if right else size-1
     else:start=max(0,size-int(right))
    else:start=size
    if start>=size or end<start:
     self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
    status=206
   self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-store')
   if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
   self.end_headers()
   if self.command=='HEAD':return
   try:
    with file.open('rb') as f:
     f.seek(start);remaining=end-start+1
     while remaining:
      block=f.read(min(65536,remaining))
      if not block:break
      self.wfile.write(block);remaining-=len(block)
   except (BrokenPipeError,ConnectionResetError):pass
  def do_GET(self):
   path=self.path.split('?')[0]
   if path=='/api/status':body=json.dumps(snapshot).encode();kind='application/json'
   elif path=='/':body=(BASE/'dashboard.html').read_bytes();kind='text/html; charset=utf-8'
   elif path=='/api/comparisons':
    from comparisons import build
    try:body=json.dumps(build(),ensure_ascii=False).encode();kind='application/json'
    except Exception as e:self.send_error(503,'Comparison artifacts unavailable');return
   elif path in ('/evaluations','/evaluations/') or re.fullmatch(r'/evaluations/\d+h/?',path):
    if re.fullmatch(r'/evaluations/\d+h/?',path):
     milestone=int(re.search(r'(\d+)h',path).group(1));h=json.loads(pathlib.Path('/data/work/asr-evaluation/periodic/history.json').read_text())
     available={100}|{int(r['training_audio_hours']//50)*50 for r in h['runs']}
     if milestone not in available:self.send_error(404,'This evaluation has not completed yet');return
    body=(BASE/'comparisons.html').read_bytes();kind='text/html; charset=utf-8'
   elif path=='/evaluations/100h/comparison.json':
    is_json=path.endswith('.json');file=pathlib.Path('/data/work/asr-evaluation/after-100h')/('comparison.json' if is_json else 'report.html')
    if not file.is_file():self.send_error(404);return
    body=file.read_bytes();kind='application/json' if is_json else 'text/html; charset=utf-8'
   elif path=='/evaluations/history.json':
    file=pathlib.Path('/data/work/asr-evaluation/periodic/history.json')
    if not file.exists():self.send_error(404);return
    body=file.read_bytes();kind='application/json'
   elif path.startswith('/references/'):
    parts=path.split('/')
    kinds={'review.html':'text/html; charset=utf-8','audio.lossless.flac':'audio/flac','audio.playback.wav':'audio/wav','audit.json':'application/json','candidate.reference.json':'application/json','gold.reference.json':'application/json','gold-approval.json':'application/json'}
    if len(parts)!=4 or parts[2] not in ('cs','sk') or parts[3] not in kinds:self.send_error(404);return
    file=pathlib.Path('/data/work/asr-evaluation/production-reference-audit')/parts[2]/parts[3]
    if not file.is_file():self.send_error(404);return
    kind=kinds[parts[3]]
    if kind.startswith('audio/'):self.serve_audio(file,kind);return
    body=file.read_bytes()
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
   if self.command!='HEAD':self.wfile.write(body)
  def log_message(self,*args):pass
 http.server.ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()

def track():
 import wandb
 config=json.loads((ROOT/'config.json').read_text());mode='online' if os.environ.get('WANDB_API_KEY') else 'offline'
 run=wandb.init(project=os.environ.get('WANDB_PROJECT','vibevoice-qda'),entity=os.environ.get('WANDB_ENTITY') or None,id='qda496-qlora-r16-v1',name='QDA 496h · QLoRA r16 · 10→30→60m',config=config,dir=str(ROOT),mode=mode,resume='allow' if mode=='online' else None,tags=['cs-sk','pseudo-labels','dual-a5000','curriculum'],notes='Metrics sidecar for existing trainer; loss is Soniox teacher agreement, not gold WER/DER. Encoder and connectors frozen. No transcripts or audio uploaded.')
 run.define_metric('periodic/audio_hours');run.define_metric('periodic/*',step_metric='periodic/audio_hours');run.define_metric('evaluation/checkpoint_update');run.define_metric('evaluation/*',step_metric='evaluation/checkpoint_update');run.define_metric('event/index');run.define_metric('train/update');run.define_metric('*',step_metric='event/index');run.define_metric('train/*',step_metric='train/update');run.define_metric('validation/*',step_metric='train/update');run.define_metric('validation/pseudo_label_loss',summary='min')
 status={'mode':mode,'url':run.url if mode=='online' else None,'project':run.project,'entity':run.entity,'id':run.id}
 (ROOT/'wandb-status.json').write_text(json.dumps(status)+'\n')
 cursor=int(run.summary.get('monitor/last_event_index',-1))+1 if mode=='online' else 0
 while True:
  events=read_events()
  for i in range(cursor,len(events)):
   e=events[i]
   if e['event'] in ('step','validation','checkpoint'):run.log(event_metrics(e,i))
   run.summary['monitor/last_event_index']=i
  cursor=len(events);state=summarize(events=events);system=telemetry()
  d={'event/index':max(0,cursor-1),'progress/update':state['step'],'progress/audio_hours':state['hours_done'],'progress/audio_percent':100*state['hours_done']/state['hours_total'],'progress/eta_hours':state['eta_seconds']/3600,'progress/event_age_seconds':state['event_age_seconds'],'checkpoint/latest_update':state['checkpoint'].get('step',0),'system/disk_free_gib':system['disk_free_gib']}
  for gpu in system['gpus']:
   for key in ('utilization','memory_used_mib','temperature','power_w'):d[f"gpu{int(gpu['index'])}/{key}"]=gpu[key]
  evaluation=pathlib.Path('/data/work/asr-evaluation/after-100h/comparison.json')
  if evaluation.exists():
   result=json.loads(evaluation.read_text())
   if run.summary.get('evaluation/after100h_created_at')!=result['created_at']:
    evaluation_metrics={'evaluation/checkpoint_update':result['checkpoint_step'],'evaluation/training_audio_hours':result['training_audio_hours']}
    for sample in result['samples']:
     for variant in ('vanilla','finetuned'):
      for key in ('wer','cpwer','cer','der_250ms','predicted_speakers'):
       if key in sample[variant]['metrics']:evaluation_metrics[f"evaluation/{sample['language']}/{variant}/{key}"]=sample[variant]['metrics'][key]
    run.log(evaluation_metrics);run.summary['evaluation/after100h_created_at']=result['created_at']
  periodic=pathlib.Path('/data/work/asr-evaluation/periodic/history.json')
  if periodic.exists():
   panel=json.loads(periodic.read_text())
   for row in [panel['baseline']]+panel['runs']:
    marker='periodic_logged/'+row['id']+'/'+panel['normalization_version']
    if not run.summary.get(marker):
     values={'periodic/audio_hours':row['training_audio_hours'],'periodic/checkpoint_update':row['checkpoint_step']}
     for key,value in row['aggregate'].items():values[f"periodic/{row['variant']}/aggregate/{key}"]=value
     for sample in row['samples']:
      for key in ('wer','number_normalized_wer','der_250ms'):values[f"periodic/{row['variant']}/{sample['language']}/{key}"]=sample['metrics'][key]
     run.log(values);run.summary[marker]=True
  run.log(d);run.summary.update({'progress/current_update':state['step'],'progress/total_updates':state['total_steps'],'data/gold_wer_available':evaluation.exists(),'data/gold_der_available':evaluation.exists() and all('der_250ms' in sample[variant]['metrics'] for sample in result['samples'] for variant in ('vanilla','finetuned'))})
  if state['completed']:run.finish();break
  time.sleep(15)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['web','track']);a=p.parse_args();web() if a.mode=='web' else track()
