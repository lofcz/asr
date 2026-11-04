import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import pathlib,sys,json,time,hashlib,argparse,re,dataclasses,unicodedata,gc
sys.path.insert(0,'/data/work/asr-pilot/long-context-pilot')
import torch,soundfile as sf
from torch.nn.attention import sdpa_kernel,SDPBackend
from peft import PeftModel
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from model_paths import model_paths,MODEL_REVISION
from rapidfuzz.distance import Levenshtein
import meeteval
from number_wer import score as number_score
ROOT=pathlib.Path('/data/work/asr-evaluation/production-reference-audit-v2');OUT=pathlib.Path('/data/work/asr-evaluation/after-100h')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFC',s).casefold()))
def save(p,x):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n');os.replace(temp,p)
def metrics(ref,hyp,folder,duration,language):
 speech=[s for s in hyp if not re.fullmatch(r'\s*\[[^\[\]]+\]\s*',s['Content'])]
 a=norm(''.join(s['Content'] for s in ref));b=norm(' '.join(s['Content'] for s in speech));aa=a.split();bb=b.split();ops=Levenshtein.editops(aa,bb)
 result={'reference_words':len(aa),'hypothesis_words':len(bb),'wer':len(ops)/len(aa),'cer':Levenshtein.distance(a,b)/len(a),'substitutions':sum(o.tag=='replace' for o in ops),'deletions':sum(o.tag=='delete' for o in ops),'insertions':sum(o.tag=='insert' for o in ops),'predicted_speakers':len({str(s['Speaker']) for s in speech}),'reference_speakers':len({str(s['Speaker']) for s in ref}),'normalization':'Unicode NFC, lowercase, Unicode word tokens; punctuation ignored, diacritics retained; numeral wording NOT normalized. Soniox chunks concatenate losslessly before word splitting.'}
 result.update(number_score(''.join(s['Content'] for s in ref),' '.join(s['Content'] for s in speech),language))
 def group(rows,reference):
  speakers={}
  for s in rows:speakers.setdefault(str(s['Speaker']),[]).append(s['Content'])
  return [{'session_id':'sample','speaker':k,'start_time':0,'end_time':duration,'words':norm(('' if reference else ' ').join(v))} for k,v in speakers.items()]
 result['cpwer']=float(meeteval.wer.cpwer(group(ref,True),group(speech,False))['sample'].error_rate)
 def timed(rows):return [{'session_id':'sample','speaker':str(s['Speaker']),'start_time':s['Start'],'end_time':s['End'],'words':norm(s['Content'])} for s in rows]
 uem=folder/'evaluation.uem';uem.write_text(f'sample 1 0 {duration}\n')
 try:
  der=meeteval.der.md_eval_22(timed(ref),timed(speech),collar=.25,uem=uem)['sample'];result['der_250ms']=float(der.error_rate);result['der_details']=dataclasses.asdict(der)
 except Exception as e:result['der_error']=str(e)
 save(folder/'word-errors.json',[{'operation':o.tag,'reference_index':o.src_pos,'hypothesis_index':o.dest_pos,'reference_word':aa[o.src_pos] if o.tag!='insert' else None,'hypothesis_word':bb[o.dest_pos] if o.tag!='delete' else None} for o in ops])
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['vanilla','finetuned'],required=True);p.add_argument('--out',type=pathlib.Path,default=OUT);p.add_argument('--language',choices=['cs','sk']);p.add_argument('--step',type=int,default=325);p.add_argument('--hours',type=float,default=102.06957241319445);a=p.parse_args();out=a.out;torch.set_num_threads(2);torch.manual_seed(20261022)
 path,tok=model_paths();processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
 model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
 encoder=model
 # Cache exactly one deterministic frozen feature draw per sample before adapter application.
 inputs={};manifest=json.loads((ROOT/'gold-manifest.json').read_text())
 for row in manifest['samples']:
  if a.language and row['language']!=a.language:continue
  lang=row['language'];audio=ROOT/row['audio_path'];ref=ROOT/row['reference_path'];assert sha(audio)==row['audio_sha256'] and sha(ref)==row['reference_sha256'];wave,sr=sf.read(audio,dtype='float32');assert sr==24000 and wave.ndim==1
  enc=processor._process_single_audio(wave,sampling_rate=sr,add_generation_prompt=False,use_streaming=True,context_info=None)
  with torch.random.fork_rng(devices=[0]),torch.inference_mode():
   torch.manual_seed(int(row['audio_sha256'][:8],16));features=encoder.encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).cuda()).squeeze(0).to(torch.bfloat16).cpu()
  ids=enc['input_ids'];mask=enc['acoustic_input_mask'];assert features.shape[0]==sum(mask)
  feature_hash=hashlib.sha256(features.view(torch.uint8).numpy().tobytes()).hexdigest();inputs[lang]=(row,ids,mask,features,feature_hash)
 for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
 gc.collect();torch.cuda.empty_cache()
 if a.variant=='finetuned':model=PeftModel.from_pretrained(model,str(out/'adapter')).merge_and_unload()
 model.eval()
 for lang,(row,ids,mask,features,feature_hash) in inputs.items():
  folder=out/a.variant/lang;folder.mkdir(parents=True,exist_ok=True)
  result={'variant':a.variant,'language':lang,'model_revision':MODEL_REVISION,'checkpoint_step':a.step if a.variant=='finetuned' else None,'training_audio_hours':a.hours if a.variant=='finetuned' else 0,'audio_sha256':row['audio_sha256'],'reference_sha256':row['reference_sha256'],'feature_sha256':feature_hash,'dtype':'bfloat16','decoding':'greedy','customized_context':[],'reference_status':'user-approved Soniox gold; generated dialogue fixture','duration_seconds':row['duration_seconds']}
  ref=json.loads((ROOT/row['reference_path']).read_text());save(folder/'reference.json',ref)
  ids=torch.tensor([ids],device='cuda');mask=torch.tensor([mask],device='cuda',dtype=torch.bool);features=features.cuda();generated=[];past=None;stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');termination='token_limit';started=time.monotonic();torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   embed=model.get_input_embeddings()(ids).clone();embed[mask]=features
   header=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False);embed=torch.cat([embed,model.get_input_embeddings()(torch.tensor([header],device='cuda'))],dim=1)
   state=model.model(inputs_embeds=embed,use_cache=True,return_dict=True);past=state.past_key_values
   for i in range(12000):
    token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
    if token==stop:termination='eos';break
    generated.append(token)
    if len(generated)>=256 and len(generated)%64==0 and generated[-64:]==generated[-128:-64]==generated[-192:-128]==generated[-256:-192]:termination='repetition_abort';break
    state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True);past=state.past_key_values
    if (i+1)%512==0:
     (folder/'output.partial.txt').write_text(processor.tokenizer.decode(generated,skip_special_tokens=True));print(json.dumps({'variant':a.variant,'language':lang,'tokens':len(generated),'elapsed_seconds':time.monotonic()-started}),flush=True)
  text=processor.tokenizer.decode(generated,skip_special_tokens=True);(folder/'output.txt').write_text(text);result.update(termination=termination,generated_tokens=len(generated),wall_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
  try:
   hyp=json.loads(text);assert isinstance(hyp,list) and hyp
   assert all({'Start','End','Content'}<=s.keys() for s in hyp)
   assert all(0<=float(s['Start'])<=float(s['End'])<=row['duration_seconds']+1 for s in hyp)
   result.update(valid_json=True,valid_timestamps=True,metrics=metrics(ref,hyp,folder,row['duration_seconds'],lang));save(folder/'hypothesis.json',hyp)
  except Exception as e:result.update(scoring_error=str(e),valid_output=False)
  save(folder/'result.json',result);print(json.dumps(result,default=str),flush=True)
  del embed,state,past,ids,mask,features;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
