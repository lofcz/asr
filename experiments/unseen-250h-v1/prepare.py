"""Freeze the requested checkpoint and identical deterministic speech inputs for both models."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import pathlib,sys,json,gc,shutil,time,subprocess
import numpy as np,torch
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/data/work/asr-experiments/qda-continuation-v2','/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice','/data/work/asr-datasets/qda-50h-v2/code']
from contracts import sha,atomic_json
from model_paths import model_paths,MODEL_REVISION
from qda_release_dataset import ReleaseDataset
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
TRAIN=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')

def guard():
 assert (TRAIN/'paused-by-user.json').exists(),'User pause marker is absent'
 for service in ('qda496-training.service','qda496-periodic-eval.service','qda50-v2-prepare.service','qda50-v2-experiments.service'):
  state=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True).stdout.strip()
  assert state in ('inactive','failed','unknown'),(service,state)

def main():
 guard();torch.set_num_threads(2);torch.cuda.set_device(0)
 panel=json.loads((HERE/'panel.json').read_text());root=pathlib.Path(panel['data_release'])
 assert sha(pathlib.Path('/data/work/asr-datasets/qda-496h-v1/training/sources.jsonl'))==panel['baseline_training_manifest_sha256']
 original=json.loads(pathlib.Path('/data/work/asr-datasets/qda-496h-v1/selection.json').read_text());oldhashes={r['sha256'] for r in original}
 plan_path=pathlib.Path('/data/work/asr-experiments/qda-continuation-v2/plan.json');assert sha(plan_path)==panel['continuation_plan_sha256'];plan=json.loads(plan_path.read_text());probe={r['record_id'] for pair in plan['probe_pairs'] for r in pair}
 assert len(panel['samples'])==10 and len({r['record_id'] for r in panel['samples']})==10
 for r in panel['samples']:assert r['audio_sha256'] not in oldhashes and r['record_id'] not in probe
 checkpoint=TRAIN/panel['checkpoint'];adapter=HERE/'adapter';adapter.mkdir(exist_ok=True)
 for name in ('adapter_model.safetensors','adapter_config.json'):
  if (adapter/name).exists():assert sha(adapter/name)==sha(checkpoint/name)
  else:shutil.copy2(checkpoint/name,adapter/name)
 atomic_json(adapter/'provenance.json',{'source':str(checkpoint),'hours':panel['checkpoint_audio_hours'],'weights_sha256':sha(adapter/'adapter_model.safetensors'),'config_sha256':sha(adapter/'adapter_config.json'),'model_revision':MODEL_REVISION})
 path,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
 cache=HERE/'inputs';cache.mkdir(exist_ok=True);datasets={};encoder=None
 for row in panel['samples']:
  sid=row['sample_id'];folder=cache/sid;folder.mkdir(exist_ok=True);receipt=folder/'receipt.json'
  if receipt.exists():
   saved=json.loads(receipt.read_text());assert saved['panel_sha256']==sha(HERE/'panel.json') and saved['prepare_sha256']==sha(HERE/'prepare.py') and sha(folder/'input.pt')==saved['input_sha256'];continue
  if encoder is None:
   encoder=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
   encoder.model.language_model=None;encoder.lm_head=None;encoder.requires_grad_(False);gc.collect();encoder.to('cuda')
  key=(row['split'],600)
  if key not in datasets:datasets[key]=ReleaseDataset(root,*key)
  ds=datasets[key];indices=[i for i,r in enumerate(ds.rows) if r['window_id']==row['window_id']];assert len(indices)==1
  sample,wave=ds.decode(indices[0]);assert sample['record_id']==row['record_id'] and abs(sample['audio_duration']-row['audio_duration'])<.01
  enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
  started=time.monotonic()
  with torch.random.fork_rng(devices=[0]),torch.inference_mode():
   torch.manual_seed(int(row['window_id'][:8],16));features=encoder.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).cuda()).squeeze(0).bfloat16().cpu()
  assert features.shape[0]==sum(enc['acoustic_input_mask']) and torch.isfinite(features).all()
  ref=[{'Start':round(s['start'],2),'End':round(s['end'],2),'Speaker':s['speaker'],'Content':s['text']} for s in sample['segments']]
  atomic_json(folder/'reference.json',ref)
  payload={'ids':torch.tensor(enc['input_ids'],dtype=torch.long),'mask':torch.tensor(enc['acoustic_input_mask'],dtype=torch.bool),'features':features}
  tmp=folder/'input.building';torch.save(payload,tmp)
  with tmp.open('rb') as f:os.fsync(f.fileno())
  tmp.replace(folder/'input.pt')
  # A local lossless crop makes later discrepancy listening reproducible without another decode.
  import soundfile as sf
  sf.write(folder/'audio.wav',wave,24000,subtype='FLOAT')
  atomic_json(receipt,{'sample_id':sid,'record_id':row['record_id'],'window_id':row['window_id'],'audio_source_sha256':row['audio_sha256'],'crop_sha256':sha(folder/'audio.wav'),'reference_sha256':sha(folder/'reference.json'),'input_sha256':sha(folder/'input.pt'),'feature_sha256':__import__('hashlib').sha256(features.view(torch.uint8).numpy().tobytes()).hexdigest(),'panel_sha256':sha(HERE/'panel.json'),'prepare_sha256':sha(HERE/'prepare.py'),'model_revision':MODEL_REVISION,'duration_seconds':sample['audio_duration'],'feature_seed':int(row['window_id'][:8],16),'customized_context':[]})
  print(json.dumps({'event':'cached','sample':sid,'seconds':time.monotonic()-started}),flush=True)
 atomic_json(HERE/'inputs-ready.json',{'samples':10,'panel_sha256':sha(HERE/'panel.json'),'model_revision':MODEL_REVISION,'audio_minutes':panel['audio_minutes'],'checkpoint_sha256':sha(adapter/'adapter_model.safetensors')})
if __name__=='__main__':main()
