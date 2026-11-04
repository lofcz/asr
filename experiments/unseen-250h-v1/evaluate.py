"""Identical BF16 greedy decoding on frozen cached inputs; only adapter weights differ."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,pathlib,sys,json,gc,time,importlib.util,hashlib
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/data/work/asr-experiments/qda-continuation-v2','/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice','/data/work/asr-evaluation/periodic/code']
from contracts import sha,atomic_json
from model_paths import model_paths,MODEL_REVISION
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from peft import PeftModel
spec=importlib.util.spec_from_file_location('established_metrics','/data/work/asr-evaluation/periodic/code/evaluate.py');metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['vanilla','finetuned'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--wall-budget',type=int,default=1200);a=p.parse_args()
 torch.set_num_threads(2);torch.cuda.set_device(a.gpu);torch.manual_seed(20261023)
 panel=json.loads((HERE/'panel.json').read_text());adapter_sha=sha(HERE/'adapter/adapter_model.safetensors') if a.variant=='finetuned' else None
 contract={'panel_sha256':sha(HERE/'panel.json'),'evaluator_sha256':sha(HERE/'evaluate.py'),'metrics_sha256':sha(pathlib.Path('/data/work/asr-evaluation/periodic/code/evaluate.py')),'normalizer_sha256':sha(pathlib.Path('/data/work/asr-evaluation/periodic/code/number_wer.py')),'model_revision':MODEL_REVISION,'adapter_sha256':adapter_sha,'dtype':'bfloat16','decoding':'greedy','customized_context':[],'max_generated_tokens':20000,'wall_budget':a.wall_budget}
 out=HERE/a.variant;out.mkdir(exist_ok=True)
 if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract,'Evaluation contract changed'
 else:atomic_json(out/'contract.json',contract)
 path,tok=model_paths();proc=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
 model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
 for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
 if a.variant=='finetuned':model=PeftModel.from_pretrained(model,HERE/'adapter').merge_and_unload()
 gc.collect();model.to('cuda').eval();torch.cuda.empty_cache()
 for row in panel['samples']:
  sid=row['sample_id'];folder=out/sid;folder.mkdir(exist_ok=True);done=folder/'result.json';source=HERE/'inputs'/sid;receipt=json.loads((source/'receipt.json').read_text())
  if done.exists():
   old=json.loads(done.read_text());assert old['contract_sha256']==sha(out/'contract.json')
   if old.get('complete'):continue
  assert receipt['panel_sha256']==contract['panel_sha256'] and sha(source/'input.pt')==receipt['input_sha256'] and sha(source/'reference.json')==receipt['reference_sha256']
  inp=torch.load(source/'input.pt',weights_only=True,map_location='cpu');ids=inp['ids'].unsqueeze(0).cuda();mask=inp['mask'].unsqueeze(0).cuda();features=inp['features'].cuda();ref=json.loads((source/'reference.json').read_text())
  started=time.monotonic();generated=[];stop=proc.tokenizer.convert_tokens_to_ids('<|im_end|>');termination='token_limit';torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   embed=model.get_input_embeddings()(ids).clone();embed[mask]=features
   header=proc.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False);embed=torch.cat([embed,model.get_input_embeddings()(torch.tensor([header],device='cuda'))],dim=1)
   state=model.model(inputs_embeds=embed,use_cache=True,return_dict=True);past=state.past_key_values
   for i in range(contract['max_generated_tokens']):
    token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
    if token==stop:termination='eos';break
    generated.append(token)
    if i%64==0 and time.monotonic()-started>a.wall_budget:termination='wall_budget';break
    if len(generated)>=256 and len(generated)%64==0 and generated[-64:]==generated[-128:-64]==generated[-192:-128]==generated[-256:-192]:termination='repetition_abort';break
    state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True);past=state.past_key_values
    if (i+1)%512==0:
     (folder/'output.partial.txt').write_text(proc.tokenizer.decode(generated,skip_special_tokens=True))
     atomic_json(out/'progress.json',{'variant':a.variant,'sample':sid,'tokens':len(generated),'wall_seconds':time.monotonic()-started,'updated_at':time.time()})
  text=proc.tokenizer.decode(generated,skip_special_tokens=True);(folder/'output.txt').write_text(text)
  result={'sample_id':sid,'variant':a.variant,'language':row['language'],'record_id':row['record_id'],'duration_seconds':row['audio_duration'],'contract_sha256':sha(out/'contract.json'),'input_sha256':receipt['input_sha256'],'feature_sha256':receipt['feature_sha256'],'reference_sha256':receipt['reference_sha256'],'termination':termination,'tokens':len(generated),'wall_seconds':time.monotonic()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'complete':False}
  try:
   hyp=json.loads(text);assert termination=='eos' and isinstance(hyp,list) and hyp
   assert all({'Start','End','Speaker','Content'}<=s.keys() and isinstance(s['Content'],str) and 0<=float(s['Start'])<=float(s['End'])<=row['audio_duration']+1 for s in hyp),'Invalid output fields/times'
   result['metrics']=json.loads(json.dumps(metric_module.metrics(ref,hyp,folder,row['audio_duration'],row['language']),default=str));assert 'der_250ms' in result['metrics']
   result['complete']=True;atomic_json(folder/'hypothesis.json',hyp)
  except Exception as e:result['error']=repr(e)
  atomic_json(done,result);print(json.dumps(result),flush=True)
  del embed,state,past,ids,mask,features,inp;gc.collect();torch.cuda.empty_cache()
 atomic_json(out/'finished.json',{'finished_at':time.time(),'samples':len(panel['samples'])})
if __name__=='__main__':main()
