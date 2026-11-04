"""Systemd-owned evaluation handoff. Timeout/failure always releases training.

Trainer saves optimizer + RNG at the first update crossing each milestone,
atomically writes request.json and exits cleanly. Timer handles the request.
ExecStopPost calls recover even after timeout or SIGKILL of this service.
"""
import pathlib,json,time,subprocess,shutil,sys,os
from history import BASE,atomic,build
TRAIN=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
REQUEST=BASE/'request.json'
def recover():
 if not REQUEST.exists():return
 req=json.loads(REQUEST.read_text());folder=BASE/'runs'/req['id'];folder.mkdir(parents=True,exist_ok=True)
 if not (folder/'completed.json').exists():atomic(folder/'failed.json',{'id':req['id'],'training_audio_hours':req['hours'],'time':time.time(),'status':'failed_or_timed_out','log_directory':str(folder)})
 policy=json.loads((BASE/'policy.json').read_text());policy['next_hours']=(int(req['hours']//50)+1)*50
 if req.get('final'):policy['enabled']=False
 atomic(BASE/'policy.json',policy)
 shutil.copy2(REQUEST,folder/'request.json');REQUEST.unlink()
 # The checkpoint remains the trainer's latest, including optimizer and RNG.
 subprocess.run(['systemctl','--user','start','qda496-training.service'],check=True)
 build()
def run():
 if not REQUEST.exists():return
 req=json.loads(REQUEST.read_text());folder=BASE/'runs'/req['id'];folder.mkdir(parents=True,exist_ok=True)
 # Let torchrun leave its distributed process group before reclaiming GPUs.
 deadline=time.monotonic()+60
 while subprocess.run(['systemctl','--user','is-active','--quiet','qda496-training.service']).returncode==0:
  if time.monotonic()>deadline:raise RuntimeError('Trainer did not finish handoff; refusing concurrent GPU use')
  time.sleep(2)
 latest=json.loads((TRAIN/'latest.json').read_text());assert latest['step']==req['step']
 source=TRAIN/latest['checkpoint'];adapter=folder/'adapter';adapter.mkdir(exist_ok=True)
 for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(source/name,adapter/name)
 atomic(folder/'request.json',req)
 jobs=[];logs=[]
 try:
  for gpu,lang in enumerate(('cs','sk')):
   log=(folder/f'{lang}.log').open('a');logs.append(log)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'/data/work/asr-pilot/VibeVoice','HF_HOME':'/data/work/asr-pilot/hf','OMP_NUM_THREADS':'2'}
   jobs.append(subprocess.Popen(['/data/work/asr-pilot/venv/bin/python','-u',str(BASE/'code/evaluate.py'),'--variant','finetuned','--out',str(folder),'--language',lang,'--step',str(req['step']),'--hours',str(req['hours'])],stdout=log,stderr=subprocess.STDOUT,env=env))
  codes=[job.wait() for job in jobs]
  if any(codes):raise RuntimeError(f'Inference failed: {codes}')
  from history import entry
  result=entry(folder,'finetuned')
  # Validate frozen inputs against baseline before publishing completion.
  baseline=entry(pathlib.Path('/data/work/asr-evaluation/after-100h'),'vanilla')
  for a,b in zip(baseline['samples'],result['samples']):
   for k in ('audio_sha256','reference_sha256','feature_sha256','dtype','decoding','customized_context'):assert a[k]==b[k],k
  atomic(folder/'completed.json',{'time':time.time(),'checkpoint_step':req['step']})
 finally:
  for job in jobs:
   if job.poll() is None:job.terminate()
  for log in logs:log.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='recover':recover()
 else:run()
