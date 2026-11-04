"""CPU token audit/plan now; frozen audio feature cache only after baseline training ends."""
import argparse,collections,gc,hashlib,json,os,pathlib,random,shutil,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
from contracts import atomic_json,sha,target_with_spans,content_token_mask
from planning import arrange,duration_class
DATA=pathlib.Path('/data/work/asr-datasets/qda-50h-v2');BASE=pathlib.Path('/data/work/asr-training/qda-496h-v1-qlora-r16')
SEED=20261023

def require_idle():
 if not (BASE/'completed.json').exists():raise RuntimeError('Baseline training is not complete; refusing GPU work')
 for service in ('qda496-training.service','qda496-periodic-eval.service'):
  state=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True).stdout.strip()
  if state not in ('inactive','failed','unknown'):raise RuntimeError('GPU work still active: '+service)
 # Check external GPU consumers as well, before allocating any model.
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if p.stdout.strip():raise RuntimeError('GPU compute processes are still running')

def runtime():
 sys.path[:0]=[str(DATA/'code'),'/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice']
 from qda_release_dataset import ReleaseDataset
 from model_paths import model_paths,MODEL_REVISION
 from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
 return ReleaseDataset,model_paths,MODEL_REVISION,VibeVoiceASRProcessor

def make_inputs():
 import torch
 ReleaseDataset,model_paths,revision,Processor=runtime();path,tok=model_paths();proc=Processor.from_pretrained(path,language_model_pretrained_name=tok)
 verified=json.loads((DATA/'verification.json').read_text());assert verified['status']=='verified'
 assert sha(DATA/'training/catalog.sqlite3')==verified['training_catalog_sha256']
 assert sha(DATA/'training/sources.jsonl')==verified['training_manifest_sha256']
 cache=HERE/'cache';cache.mkdir(exist_ok=True)
 cache_contract={'schema':1,'dataset_manifest_sha256':verified['training_manifest_sha256'],'model_revision':revision,'target_code_sha256':sha(HERE/'contracts.py'),'context':[],'max_tokens':65536,'feature_dtype':'bfloat16','feature_seed':'first 8 hex digits of window_id','cache_code_sha256':sha(HERE/'prepare_experiment.py')}
 if (cache/'contract.json').exists():assert json.loads((cache/'contract.json').read_text())==cache_contract,'Cache contract changed'
 else:atomic_json(cache/'contract.json',cache_contract)
 datasets={(s,b):ReleaseDataset(DATA,s,b) for s in ('train','validation') for b in (600,1800,3600)}
 sources=[json.loads(l) for l in (DATA/'training/sources.jsonl').read_text().splitlines()];metadata={r['record_id']:r for r in sources}
 groups={}
 for key,ds in datasets.items():
  g=collections.defaultdict(list)
  for i,r in enumerate(ds.rows):g[r['record_id']].append(i)
  groups[key]=g
 allrows=[];longrows=[]
 for source in sources:
  split=source['split'];rid=source['record_id']
  if split=='train':
   v=int(hashlib.sha256(('view-'+rid).encode()).hexdigest()[:8],16)%100
   buckets=[600] if v<35 else [1800,600] if v<75 else [3600,1800,600]
  else:buckets=[600,3600]
  accepted=None
  for bucket in buckets:
   ds=datasets[split,bucket];indices=groups[split,bucket][rid];built=[]
   for index in indices:
    sample=ds.label(index);wid=sample['window_id'];dest=cache/(wid+'.input.pt');receipt=cache/(wid+'.input.json')
    if receipt.exists():
     row=json.loads(receipt.read_text());assert row['label_sha256']==source['label_sha256'] and sha(dest)==row['input_sha256']
    else:
     sample,wave=ds.decode(index);enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
     body,spans=target_with_spans(sample);template=proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=False,add_generation_prompt=False)
     target=proc.tokenizer(template,add_special_tokens=False,return_offsets_mapping=True)
     assert target['input_ids']==proc.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=True,add_generation_prompt=False)
     start=template.index(body);cm=content_token_mask(target['offset_mapping'],[(a+start,b+start) for a,b in spans])
     ids=enc['input_ids']+target['input_ids'];mask=enc['acoustic_input_mask']+[0]*len(target['input_ids']);labels=[-100]*len(enc['input_ids'])+target['input_ids']
     row={'window_id':wid,'record_id':rid,'split':split,'language':source['language'],'bucket':bucket,'index':index,'duration':sample['audio_duration'],'source_start':sample['source_start'],'source_end':sample['source_end'],'audio_path':source['audio_path'],'audio_sha256':source['audio_sha256'],'label_sha256':source['label_sha256'],'total_tokens':len(ids),'target_tokens':len(target['input_ids']),'content_tokens':sum(cm),'speech_features':sum(mask)}
     payload={'ids':torch.tensor(ids,dtype=torch.int64),'mask':torch.tensor(mask,dtype=torch.bool),'labels':torch.tensor(labels,dtype=torch.int64),'content_mask':torch.tensor([False]*len(enc['input_ids'])+cm,dtype=torch.bool)}
     tmp=dest.with_suffix('.building');torch.save(payload,tmp)
     with tmp.open('rb') as f:os.fsync(f.fileno())
     tmp.replace(dest);row['input_sha256']=sha(dest);atomic_json(receipt,row)
     del wave,enc,payload
    row={**row,'bucket':bucket,'index':index}
    built.append(row)
   if built and max(r['total_tokens'] for r in built)<=65536 and abs(sum(r['duration'] for r in built)-source['audio_duration'])<.05:
    if split=='validation':
     if bucket==600:accepted=built
     else:longrows+=built
    else:accepted=built;break
  if accepted is None:raise RuntimeError('Cannot represent complete source within token budget: '+rid)
  allrows+=accepted
  print(json.dumps({'event':'tokenized','record':rid,'windows':len(accepted),'max_tokens':max(r['total_tokens'] for r in accepted)}),flush=True)
 train=[r for r in allrows if r['split']=='train'];all_validation=[r for r in allrows if r['split']=='validation'];validation=[]
 for language in ('cs','sk'):
  source_ids=sorted({r['record_id'] for r in all_validation if r['language']==language})
  for rid in source_ids[:4]:
   candidates=sorted((r for r in all_validation if r['record_id']==rid),key=lambda r:r['source_start']);validation.append(candidates[len(candidates)//2])
 # Actual duration balance prevents nominal whole-session views of short sources from dominating.
 arranged,probe=arrange(train,SEED);seconds=sum(r['duration'] for pair in probe for r in pair)
 # Calibration uses a short deterministic prefix; diagnostics only, not an optimal-LR claim.
 calibration=sorted(probe,key=lambda pair:max(r['total_tokens'] for r in pair))[:6]
 val_sorted=sorted(validation,key=lambda r:(r['language'],r['record_id'],r['source_start']))
 val_small=[]
 for language in ('cs','sk'):
  choices=[r for r in val_sorted if r['language']==language]
  if not choices:raise RuntimeError('Missing held-out language: '+language)
  val_small+=choices[:2]
 long_validation=[max((r for r in longrows if r['language']==lang),key=lambda r:r['duration']) for lang in ('cs','sk')]
 selected_ids={r['window_id'] for p in probe for r in p}|{r['window_id'] for r in validation+long_validation}
 plan={'version':1,'seed':SEED,'dataset':str(DATA),'dataset_sha256':verified['training_manifest_sha256'],'cache_contract_sha256':sha(cache/'contract.json'),'probe_pairs':probe,'calibration_pairs':calibration,'all_training_pairs':arranged,'validation':validation,'long_validation':long_validation,'validation_small':val_small,'probe_audio_hours':seconds/3600,'probe_target_tokens':sum(r['target_tokens'] for p in probe for r in p),'cached_windows_initial':sorted(selected_ids),'training_language_hours':{l:sum(r['duration'] for p in probe for r in p if r['language']==l)/3600 for l in ('cs','sk')},'probe_length_hours':{str(b):sum(r['duration'] for p in probe for r in p if r['bucket']==b)/3600 for b in (600,1800,3600)},'quality':'machine-validated Soniox pseudo-labels; development labels not human gold'}
 plan['planning_sha256']=sha(HERE/'planning.py');plan['probe_duration_hours']={d:sum(r['duration'] for pair in probe for r in pair if duration_class(r)==d)/3600 for d in ('short','medium','long')}
 dest=HERE/'plan.json'
 if dest.exists():assert json.loads(dest.read_text())==plan,'Immutable experiment plan changed'
 else:atomic_json(dest,plan)
 print(json.dumps({k:plan[k] for k in ('probe_audio_hours','probe_target_tokens','training_language_hours','probe_length_hours')}),flush=True)

def pin_baseline():
 require_idle();latest=json.loads((BASE/'latest.json').read_text());completed=json.loads((BASE/'completed.json').read_text());assert latest['step']==completed['steps']
 src=BASE/latest['checkpoint'];dst=HERE/'baseline';dst.mkdir(exist_ok=True)
 for name in ('adapter_model.safetensors','adapter_config.json'):
  if (dst/name).exists():assert sha(src/name)==sha(dst/name),'Baseline already pinned to different checkpoint'
  else:shutil.copy2(src/name,dst/name)
 atomic_json(dst/'provenance.json',{'source':str(src),'step':latest['step'],'adapter_sha256':sha(dst/'adapter_model.safetensors'),'config_sha256':sha(dst/'adapter_config.json'),'model_revision':json.loads((BASE/'config.json').read_text())['model_revision']})

def make_features(rank,world):
 import torch,numpy as np
 if world==1:require_idle()
 else:
  if not (BASE/'completed.json').exists():raise RuntimeError('Baseline incomplete')
  for service in ('qda496-training.service','qda496-periodic-eval.service'):
   status=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,text=True).stdout.strip()
   if status not in ('inactive','failed','unknown'):raise RuntimeError('Baseline services still active')
 ReleaseDataset,model_paths,revision,Processor=runtime();plan=json.loads((HERE/'plan.json').read_text());cache=HERE/'cache'
 path,tok=model_paths();proc=Processor.from_pretrained(path,language_model_pretrained_name=tok)
 from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
 torch.cuda.set_device(rank);torch.set_num_threads(2)
 # Initialize on CPU, discard unused language model, then allocate only frozen encoder on GPU.
 encoder=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval();encoder.model.language_model=None;encoder.lm_head=None;encoder.requires_grad_(False);gc.collect();encoder.to('cuda')
 rows={r['window_id']:r for pair in plan['all_training_pairs'] for r in pair};rows.update({r['window_id']:r for r in plan['validation']+plan['long_validation']})
 datasets={};contract=plan['cache_contract_sha256']
 for wid in plan['cached_windows_initial'][rank::world]:
  r=rows[wid];dest=cache/(wid+'.features.npy');receipt=cache/(wid+'.features.json')
  if receipt.exists():
   old=json.loads(receipt.read_text());assert old['contract_sha256']==contract and old['input_sha256']==r['input_sha256'] and sha(dest)==old['sha256'];continue
  key=(r['split'],r['bucket'])
  if key not in datasets:datasets[key]=ReleaseDataset(DATA,*key)
  sample,wave=datasets[key].decode(r['index']);assert sample['window_id']==wid
  enc=proc._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=None)
  with torch.random.fork_rng(devices=[rank]),torch.inference_mode():
   torch.manual_seed(int(wid[:8],16));f=encoder.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).to('cuda')).squeeze(0).to(torch.bfloat16).cpu()
  assert f.shape[0]==r['speech_features'] and torch.isfinite(f).all()
  temp=dest.with_suffix('.building')
  with temp.open('wb') as out:np.save(out,f.view(torch.uint16).numpy(),allow_pickle=False);out.flush();os.fsync(out.fileno())
  temp.replace(dest);atomic_json(receipt,{'window_id':wid,'shape':list(f.shape),'dtype':'bfloat16-as-uint16','sha256':sha(dest),'contract_sha256':contract,'input_sha256':r['input_sha256'],'seed':int(wid[:8],16)})
  print(json.dumps({'event':'features','rank':rank,'window_id':wid}),flush=True)
 del encoder;gc.collect();torch.cuda.empty_cache()

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['inputs','pin-baseline','features']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);args=p.parse_args()
 if args.command=='inputs':make_inputs()
 elif args.command=='pin-baseline':pin_baseline()
 else:
  # Parent launch_ready.py owns the admission lock for both workers.
  if not (BASE/'completed.json').exists():raise RuntimeError('Baseline incomplete')
  make_features(args.rank,args.world)
if __name__=='__main__':main()
