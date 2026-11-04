"""Full-weight BF16 continuation from the evaluated parent adapter."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import gc,hashlib,json,pathlib,sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[os.environ.get('VIBEVOICE_REPO','/data/work/asr-pilot/VibeVoice'),'/data/work/asr-pilot/long-context-pilot']
from audio_path import encode_speech_trainable,trainable_groups
MODEL_REVISION='d0c9efdb8d614685062c04425d91e01b6f37d944'
TOKENIZER_REVISION='d149729398750b98c0af14eb82c78cfe92750796'

def sha(path):
 with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic(path,value):
 import tempfile
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def paths():
 from huggingface_hub import snapshot_download
 base=snapshot_download('microsoft/VibeVoice-ASR',revision=MODEL_REVISION,cache_dir=str(pathlib.Path(os.environ['HF_HOME'])/'hub'))
 tok=snapshot_download('Qwen/Qwen2.5-7B',revision=TOKENIZER_REVISION,cache_dir=str(pathlib.Path(os.environ['HF_HOME'])/'hub'),allow_patterns=['*token*','*.json'])
 return base,tok

def load_parent(parent=None):
 from peft import PeftModel
 from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
 from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
 base,tok=paths();parent=pathlib.Path(parent or HERE/'parent')
 manifest=json.loads((HERE/'manifest.json').read_text())
 if sha(parent/'adapter_model.safetensors')!=manifest['parent_adapter_sha256']:raise ValueError('Wrong parent adapter')
 model=VibeVoiceASRForConditionalGeneration.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
 model=PeftModel.from_pretrained(model,str(parent)).merge_and_unload()
 # Acoustic synthesis decoder is not part of ASR and has no loss path.
 model.model.acoustic_tokenizer.decoder=None
 model.requires_grad_(True)
 model.config.use_cache=False;model.model.language_model.config.use_cache=False
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 groups=trainable_groups(model)
 if any('lora_' in name for name,_ in model.named_parameters()):raise ValueError('Parent adapter was not merged')
 if any(p.__class__.__name__=='Params4bit' for p in model.parameters()):raise ValueError('Full-weight continuation must not be quantized')
 processor=VibeVoiceASRProcessor.from_pretrained(base,language_model_pretrained_name=tok)
 return model,processor,{name:sum(p.numel() for _,p in values) for name,values in groups.items()}

def audio_path(row):
 roots=json.loads((HERE/'audio-paths.json').read_text()) if (HERE/'audio-paths.json').exists() else {}
 return roots.get(row['audio_sha256'],row['audio_file'])

def pcm_crop(pcm, start, duration, sample_rate=24000):
 first=round(start*sample_rate);count=round(duration*sample_rate)
 if first<0 or count<=0:raise ValueError('Invalid PCM interval')
 wave=pcm[first:first+count].copy()
 # Resampling a source whose duration was measured at a different sample rate
 # may round its final sample differently. Retain the real samples, never pad.
 if not 0<=count-len(wave)<=3 or not np.isfinite(wave).all():
  raise ValueError(f'Incorrect decoded crop: expected {count}, got {len(wave)} samples')
 return wave

def prepare_item(row,processor,*,read_cache=True):
 import subprocess,imageio_ffmpeg
 cache=HERE/'raw-inputs'/(row['window_id']+'.pt');receipt=cache.with_suffix('.json')
 row_digest=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 if read_cache and (cache.exists() or receipt.exists()):
  saved=json.loads(receipt.read_text())
  if saved['row_sha256']!=row_digest or saved['sha256']!=sha(cache) or saved['tokenizer_revision']!=TOKENIZER_REVISION:
   raise ValueError('Raw waveform/input cache differs from source, target or tokenizer')
  item=torch.load(cache,map_location='cpu',weights_only=True)
  if set(item)!={'ids','mask','labels','speech','target_tokens','window_id'}:
   raise ValueError('Raw input cache contains unexpected tensors, possibly frozen features')
  return item
 path=audio_path(row)
 if sha(path)!=row['audio_sha256']:raise ValueError('Audio/source pairing changed')
 # Decode the source timeline once per item. Short diagnostic files only; exact
 # PCM indexing avoids the original AAC/container timestamp failure.
 result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',path,
  '-vn','-ac','1','-ar','24000','-f','f32le','-'],capture_output=True,check=True)
 if result.stderr:raise ValueError('Audio decode error')
 pcm=np.frombuffer(result.stdout,dtype='<f4');wave=pcm_crop(pcm,row['source_start'],row['duration'])
 encoded=processor._process_single_audio(wave,sampling_rate=24000,add_generation_prompt=False,use_streaming=True,context_info=row.get('customized_context') or None)
 body=json.dumps(row['target'],ensure_ascii=False,separators=(',',':'))
 template=processor.tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=False,add_generation_prompt=False)
 tail=processor.tokenizer.encode(template,add_special_tokens=False)
 prefix=encoded['input_ids'];n=len(prefix)
 ids=torch.tensor(prefix+tail,dtype=torch.long)
 mask=torch.tensor(list(encoded['acoustic_input_mask'])+[False]*len(tail),dtype=torch.bool)
 labels=ids.clone();labels[:n]=-100
 # Only waveform and token tensors, never trainable audio representations.
 return {'ids':ids,'mask':mask,'labels':labels,'speech':torch.from_numpy(encoded['speech'].copy()),
         'target_tokens':len(tail),'window_id':row['window_id']}

class Objective(torch.nn.Module):
 def __init__(self,model,tile_seconds=30.):super().__init__();self.model=model;self.tile_seconds=tile_seconds
 def forward(self,item):
  device=self.model.get_input_embeddings().weight.device
  ids=item['ids'].unsqueeze(0).to(device);mask=item['mask'].unsqueeze(0).to(device);labels=item['labels'].unsqueeze(0).to(device)
  features=encode_speech_trainable(self.model,item['speech'].unsqueeze(0).to(device),self.tile_seconds,True)
  if features.shape[1]!=int(mask.sum()):raise ValueError('Audio feature/token-mask mismatch')
  embeddings=self.model.get_input_embeddings()(ids).clone();embeddings[mask]=features.reshape(-1,features.shape[-1])
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   hidden=self.model.model(inputs_embeds=embeddings,use_cache=False,return_dict=True).last_hidden_state
  y=labels[:,1:].reshape(-1);x=hidden[:,:-1].reshape(-1,hidden.shape[-1]);valid=y!=-100;x=x[valid];y=y[valid]
  def loss_chunk(a,b):return F.cross_entropy(self.model.lm_head(a).float(),b,reduction='sum')
  total=x.new_zeros((),dtype=torch.float32)
  for start in range(0,len(y),128):
   a,b=x[start:start+128],y[start:start+128]
   total+=checkpoint(loss_chunk,a,b,use_reentrant=False) if self.training and torch.is_grad_enabled() else loss_chunk(a,b)
  return total/len(y)
