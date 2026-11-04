"""Portable waveform+token cache. Contains no acoustic/semantic learned features."""
import concurrent.futures,hashlib,json,os,pathlib,time
import torch
from runtime import HERE,paths,prepare_item,atomic,sha,TOKENIZER_REVISION
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

def main():
 torch.set_num_threads(1);base,tok=paths();processor=VibeVoiceASRProcessor.from_pretrained(base,language_model_pretrained_name=tok)
 manifest=json.loads((HERE/'manifest.json').read_text());rows=manifest['train']+manifest['development'];folder=HERE/'raw-inputs';folder.mkdir(exist_ok=True)
 def prepare(row):
  dest=folder/(row['window_id']+'.pt');receipt=dest.with_suffix('.json')
  complete=dest.exists() and receipt.exists()
  item=prepare_item(row,processor,read_cache=complete)
  if not complete:
   temp=dest.with_suffix('.building');torch.save(item,temp)
   with temp.open('rb') as f:os.fsync(f.fileno())
   os.replace(temp,dest)
   atomic(receipt,{'sha256':sha(dest),'row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
    'tokenizer_revision':TOKENIZER_REVISION,'source_audio_sha256':row['audio_sha256'],'source_start':row['source_start'],'source_end':row['source_end'],
    'waveform_samples':item['speech'].numel(),'learned_features_cached':False,'target_tokens':item['target_tokens']})
  return {'window_id':row['window_id'],'sha256':sha(dest),'source_audio_sha256':row['audio_sha256']}
 done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
  futures={executor.submit(prepare,row):row for row in rows}
  for future in concurrent.futures.as_completed(futures):
   try:done.append(future.result())
   except Exception as exc:
    atomic(HERE/'input-failure.json',{'window_id':futures[future]['window_id'],'error':str(exc),'updated_at':time.time()})
    for pending in futures:pending.cancel()
    raise
   atomic(HERE/'input-progress.json',{'completed':len(done),'total':len(rows),'updated_at':time.time()})
 done.sort(key=lambda value:value['window_id'])
 atomic(HERE/'inputs-ready.json',{'status':'passed','manifest_sha256':sha(HERE/'manifest.json'),'inputs':done,'learned_features_cached':False,
  'bytes':sum(p.stat().st_size for p in folder.glob('*.pt')),'checked_at':time.time()})
 print(json.dumps({'prepared':len(done),'learned_features_cached':False}),flush=True)
if __name__=='__main__':main()
