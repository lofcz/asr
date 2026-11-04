"""Deterministic decoded WER/DER on new held-out Soniox references, cached once per adapter."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import argparse,gc,json,pathlib,sys,time
import numpy as np,torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from peft import PeftModel
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice','/data/work/asr-evaluation/periodic/code','/data/work/asr-datasets/qda-50h-v2/code']
from model_paths import model_paths,MODEL_REVISION
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from qda_release_dataset import ReleaseDataset
from evaluate import metrics
from contracts import sha,atomic_json
from runtime_state import evaluation_hashes

def main():
 p=argparse.ArgumentParser();p.add_argument('--adapter',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=2);p.add_argument('--wall-budget',type=int,default=1800);args=p.parse_args()
 plan=json.loads((HERE/'plan.json').read_text());adapter_hash=sha(args.adapter/'adapter_model.safetensors');code_hashes=evaluation_hashes();torch.cuda.set_device(args.rank);torch.set_num_threads(2);torch.manual_seed(plan['seed'])
 path,tok=model_paths();processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
 model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
 for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
 model=PeftModel.from_pretrained(model,args.adapter).merge_and_unload().to('cuda').eval();gc.collect();torch.cuda.empty_cache()
 datasets={b:ReleaseDataset(plan['dataset'],'validation',b) for b in (600,3600)}
 panel=list({r['window_id']:r for r in plan['validation']+plan['long_validation']}.values())
 for row in panel[args.rank::args.world]:
  wid=row['window_id'];folder=args.out/wid;folder.mkdir(parents=True,exist_ok=True);resultpath=folder/'result.json'
  if resultpath.exists():
   saved=json.loads(resultpath.read_text())
   if saved.get('complete') and saved['adapter_sha256']==adapter_hash and saved['input_sha256']==row['input_sha256'] and saved.get('evaluation_sha256')==code_hashes:continue
  start=time.monotonic();cache=HERE/'cache';ip=cache/(wid+'.input.pt');fp=cache/(wid+'.features.npy');fm=json.loads((cache/(wid+'.features.json')).read_text())
  assert sha(ip)==row['input_sha256'] and sha(fp)==fm['sha256'] and fm['contract_sha256']==plan['cache_contract_sha256'] and fm['input_sha256']==row['input_sha256']
  inp=torch.load(ip,map_location='cpu',weights_only=True);n=int(torch.where(inp['labels']!=-100)[0][0]);ids=inp['ids'][:n].unsqueeze(0).cuda();mask=inp['mask'][:n].unsqueeze(0).cuda();features=torch.from_numpy(np.load(fp,allow_pickle=False).copy()).view(torch.bfloat16).cuda()
  sample=datasets[row['bucket']].label(row['index']);assert sample['window_id']==wid
  ref=[{'Start':round(s['start'],2),'End':round(s['end'],2),'Speaker':s['speaker'],'Content':s['text']} for s in sample['segments']];atomic_json(folder/'reference.json',ref)
  stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');generated=[];termination='token_limit';max_tokens=min(65536-n-8,max(4096,int(row['duration']*20)))
  with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   embed=model.get_input_embeddings()(ids).clone();embed[mask]=features
   header=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False);embed=torch.cat([embed,model.get_input_embeddings()(torch.tensor([header],device='cuda'))],dim=1)
   state=model.model(inputs_embeds=embed,use_cache=True,return_dict=True);past=state.past_key_values
   for i in range(max_tokens):
    token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
    if token==stop:termination='eos';break
    generated.append(token)
    if i%64==0 and time.monotonic()-start>args.wall_budget:termination='wall_budget';break
    if len(generated)>=256 and len(generated)%64==0 and generated[-64:]==generated[-128:-64]==generated[-192:-128]==generated[-256:-192]:termination='repetition_abort';break
    state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True);past=state.past_key_values
    if (i+1)%512==0:(folder/'output.partial.txt').write_text(processor.tokenizer.decode(generated,skip_special_tokens=True))
  text=processor.tokenizer.decode(generated,skip_special_tokens=True);(folder/'output.txt').write_text(text)
  result={'window_id':wid,'record_id':row['record_id'],'language':row['language'],'duration':row['duration'],'adapter_sha256':adapter_hash,'input_sha256':row['input_sha256'],'features_sha256':fm['sha256'],'reference_sha256':sha(folder/'reference.json'),'termination':termination,'wall_seconds':time.monotonic()-start,'generated_tokens':len(generated),'reference_quality':'unreviewed Soniox pseudo-reference; not gold','model_revision':MODEL_REVISION,'dtype':'bfloat16','decoding':'greedy','complete':False}
  result['evaluation_sha256']=code_hashes
  try:
   hyp=json.loads(text);assert termination=='eos' and isinstance(hyp,list) and hyp
   assert all({'Start','End','Speaker','Content'}<=s.keys() and isinstance(s['Content'],str) and 0<=float(s['Start'])<=float(s['End'])<=row['duration']+1 for s in hyp)
   result['metrics']=json.loads(json.dumps(metrics(ref,hyp,folder,row['duration'],row['language']),default=str));assert 'der_250ms' in result['metrics'];result['complete']=True;atomic_json(folder/'hypothesis.json',hyp)
  except Exception as e:result['error']=str(e)
  atomic_json(resultpath,result);print(json.dumps({'window':wid,'complete':result['complete'],'wall_seconds':result['wall_seconds']}),flush=True)
  del embed,state,past,ids,mask,features;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
