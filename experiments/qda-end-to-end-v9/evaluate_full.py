"""Decode with updated encoders/connectors; old v8 feature caches are forbidden."""
import argparse,collections,gc,json,pathlib,sys,time
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from runtime import HERE,atomic,sha,load_parent,prepare_item,MODEL_REVISION,TOKENIZER_REVISION
from audio_path import encode_speech_trainable
from metrics import metrics,validate_hypothesis


def main():
 p=argparse.ArgumentParser();p.add_argument('--export',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True)
 p.add_argument('--gpu',type=int,default=0);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
 p.add_argument('--wall-seconds-per-clip',type=float,default=600.);args=p.parse_args()
 torch.cuda.set_device(args.gpu);torch.set_num_threads(2);torch.manual_seed(20261025)
 model,processor,counts=load_parent();identity={'parent':json.loads((HERE/'manifest.json').read_text())['parent_adapter_sha256'],'full_weights':None,
  'manifest_sha256':sha(HERE/'manifest.json'),'model_revision':MODEL_REVISION,'tokenizer_revision':TOKENIZER_REVISION,
  'runtime_sha256':{n:sha(HERE/n) for n in ['runtime.py','audio_path.py','evaluate_full.py','metrics.py','number_wer.py']}}
 if args.export:
  path=args.export/'pytorch_model.bin';identity['full_weights']=sha(path)
  state=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
  # DeepSpeed exports the Objective wrapper's state, whose first prefix is model.
  if not all(k.startswith('model.') for k in state):raise ValueError('Unexpected full-weight export namespace')
  model.load_state_dict({k[len('model.'):]:v for k,v in state.items()},strict=True);del state
 model.eval().requires_grad_(False).cuda();gc.collect()
 manifest=json.loads((HERE/'manifest.json').read_text());summary=[]
 for row in manifest['development'][args.rank::args.world]:
  folder=args.out/row['window_id'];folder.mkdir(parents=True,exist_ok=True)
  stamp=folder/'result.json'
  if stamp.exists():
   old=json.loads(stamp.read_text())
   if old['identity']==identity and old.get('complete'):summary.append(old);continue
  item=prepare_item(row,processor);n=int(torch.where(item['labels']!=-100)[0][0]);ids=item['ids'][:n].unsqueeze(0).cuda();mask=item['mask'][:n].unsqueeze(0).cuda()
  torch.manual_seed(int(row['window_id'][:8],16));generated=[];stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');reason='token_limit';begin=time.monotonic()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   # Recompute from this checkpoint's encoders every time, including baseline.
   features=encode_speech_trainable(model,item['speech'].unsqueeze(0).cuda(),checkpoint_activations=False)
   embed=model.get_input_embeddings()(ids).clone();embed[mask]=features.reshape(-1,features.shape[-1])
   header=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
   embed=torch.cat([embed,model.get_input_embeddings()(torch.tensor([header],device='cuda'))],dim=1)
   state=model.model(inputs_embeds=embed,use_cache=True,return_dict=True);past=state.past_key_values
   for i in range(max(1024,int(row['duration']*30))):
    token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
    if token==stop:reason='eos';break
    generated.append(token)
    if i%64==0 and time.monotonic()-begin>args.wall_seconds_per_clip:reason='wall_budget';break
    state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True);past=state.past_key_values
  text=processor.tokenizer.decode(generated,skip_special_tokens=True);(folder/'output.txt').write_text(text)
  result={'window_id':row['window_id'],'record_id':row['record_id'],'language':row['language'],'identity':identity,
   'raw_audio_features_recomputed':True,'termination':reason,'wall_seconds':time.monotonic()-begin,'complete':False}
  try:
   output=json.loads(text)
   if reason!='eos' or not output:raise ValueError('Incomplete generation')
   result['timestamp_issues']=[i for i,s in enumerate(output) if not 0<=s['Start']<=s['End']<=row['duration']+.02]
   # DER uses the existing scorer and clip UEM, with raw timestamp defects retained.
   validate_hypothesis(output,row['duration'])
   result['metrics']=json.loads(json.dumps(metrics(row['target'],output,folder,row['duration'],row['language']),default=str))
   result['complete']=True
  except Exception as exc:result['complete']=False;result['error']=str(exc)
  atomic(stamp,result);summary.append(result);print(json.dumps(result),flush=True)
  del embed,features,state,past,ids,mask;gc.collect();torch.cuda.empty_cache()
 atomic(args.out/f'completed-rank-{args.rank}.json',{'results':summary,'all_complete':all(x['complete'] for x in summary)})
if __name__=='__main__':main()
