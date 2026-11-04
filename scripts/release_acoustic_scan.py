"""Resumable full-timeline acoustic evidence on mlserver, one shard per GPU."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import argparse,pathlib,json,time,subprocess,hashlib,tempfile,logging
import numpy as np
import torch
import imageio_ffmpeg
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification
MODEL='MIT/ast-finetuned-audioset-10-10-0.4593';REVISION='f826b80d28226b62986cc218e5cec390b1096902'
GROUPS={'[Music]':['Music'],'[Human Sounds]':['Laughter','Cough','Sneeze','Breathing','Sigh','Gasp','Throat clearing','Hiccup','Crying, sobbing'],'[Noise]':['Noise','Static','White noise','Pink noise','Hiss','Distortion'],'[Environmental Sounds]':['Wind','Rain','Traffic noise, roadway noise','Vehicle','Bird','Dog','Door','Water'],'[Silence]':['Silence']}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--shards',type=int,default=2);p.add_argument('--batch',type=int,default=24);a=p.parse_args();root=pathlib.Path(a.root)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');torch.set_num_threads(2)
 rows=json.loads((root/'selection.json').read_bytes())[a.rank::a.shards];extractor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION,local_files_only=True);model=AutoModelForAudioClassification.from_pretrained(MODEL,revision=REVISION,use_safetensors=True,local_files_only=True).cuda().eval();lookup={v:int(k) for k,v in model.config.id2label.items()};ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();failed={}
 while True:
  finished=0;progress=False
  for r in rows:
   dest=root/'acoustic'/f"{r['record_id']}.json";source=root/r['path']
   if dest.exists():finished+=1;continue
   if not source.exists() or failed.get(r['record_id'],0)>time.time():continue
   try:
    if source.stat().st_size!=r['bytes'] or sha(source)!=r['sha256']:raise ValueError('Source identity mismatch')
    started=time.monotonic();decoded=subprocess.run([ffmpeg,'-nostdin','-v','error','-threads','1','-i',str(source),'-vn','-ar','16000','-ac','1','-f','f32le','-'],capture_output=True,check=True);wave=np.frombuffer(decoded.stdout,dtype='<f4');duration=len(wave)/16000
    if not len(wave):raise ValueError('Empty decode')
    windows=[(float(s),min(float(s+10),duration)) for s in range(0,int(np.ceil(duration)),10)];results=[];batch=a.batch;offset=0
    while offset<len(windows):
     current=windows[offset:offset+batch];samples=[wave[int(s*16000):int(e*16000)].copy() for s,e in current];inputs=extractor(samples,sampling_rate=16000,return_tensors='pt')
     try:
      with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):scores=model(**{k:v.cuda() for k,v in inputs.items()}).logits.sigmoid().float().cpu().numpy()
     except torch.cuda.OutOfMemoryError:
      if batch<=1:raise
      batch=max(1,batch//2);torch.cuda.empty_cache();continue
     for (s,e),v in zip(current,scores):
      top=np.argsort(v)[-8:][::-1];groups={tag:max((float(v[lookup[n]]) for n in names if n in lookup),default=0) for tag,names in GROUPS.items()};clip=wave[int(s*16000):int(e*16000)];rms=float(np.sqrt(np.mean(clip.astype(np.float64)**2)));peak=float(np.abs(clip).max())
      results.append({'start':s,'end':e,'group_scores':groups,'speech_score':float(v[lookup['Speech']]),'rms_dbfs':float(20*np.log10(max(rms,1e-12))),'peak_dbfs':float(20*np.log10(max(peak,1e-12))),'top_labels':[{'label':model.config.id2label[int(i)],'score':float(v[i])} for i in top]})
     offset+=len(current)
    save(dest,{'record_id':r['record_id'],'audio_sha256':r['sha256'],'model':MODEL,'revision':REVISION,'decoded_seconds':duration,'decoder_diagnostics':decoded.stderr.decode(errors='replace'),'window_seconds':10,'windows':results,'status':'machine_evidence_not_verified_annotations','elapsed_seconds':time.monotonic()-started})
    progress=True;finished+=1;logging.info('%s %.1f minutes %d windows %.1fs',r['id'],duration/60,len(windows),time.monotonic()-started)
   except Exception as e:
    failed[r['record_id']]=time.time()+1800;logging.exception('%s failed',r['id']);save(root/'acoustic/errors'/f"{r['record_id']}.json",{'error':str(e),'time':time.time()})
  save(root/'acoustic'/f'progress-rank{a.rank}.json',{'completed':finished,'total':len(rows),'updated_at':time.time()})
  if finished==len(rows):break
  time.sleep(5 if progress else 15)
if __name__=='__main__':main()
