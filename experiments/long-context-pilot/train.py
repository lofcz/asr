import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,contextlib,json,pathlib,time,traceback,types,random,gc
import numpy as np,torch,psutil
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from model_paths import model_paths,MODEL_REVISION
BASE=pathlib.Path(__file__).resolve().parent

class Cache:
    def __init__(self):
        cache_root=pathlib.Path(os.environ.get('ASR_CACHE_DIR',str(BASE/'cache')))
        assert not (cache_root/'.building').exists()
        self.metadata=json.loads((cache_root/'metadata.json').read_text())
        self.rows=[json.loads(l) for l in (cache_root/'index.jsonl').read_text().splitlines()]
        self.arrays={k:np.load(cache_root/f'{k}.npy',mmap_mode='r') for k in ('tokens','labels','mask','features')}
    def item(self,r,device='cuda'):
        o,n=r['token_offset'],r['token_length'];f,m=r['feature_offset'],r['feature_length']
        def take(k,dtype):return torch.as_tensor(np.array(self.arrays[k][o:o+n],copy=True),device=device,dtype=dtype).unsqueeze(0)
        return take('tokens',torch.long),take('mask',torch.bool),torch.as_tensor(np.array(self.arrays['features'][f:f+m],copy=True),device=device,dtype=torch.bfloat16),take('labels',torch.long)

class BoundaryOffload:
    def __init__(self,enabled,pin_memory=True):self.enabled=enabled;self.pin_memory=pin_memory;self.total_bytes=0
    def pack(self,tensor):
        if tensor.ndim==3 and tensor.shape[-1]==3584 and tensor.shape[1]>512 and tensor.device.type=='cuda':
            cpu=torch.empty_like(tensor,device='cpu',pin_memory=self.pin_memory);cpu.copy_(tensor.detach());self.total_bytes+=tensor.numel()*tensor.element_size();return ('host',cpu,tensor.device)
        return ('keep',tensor.detach())
    def unpack(self,packed):return packed[1].to(packed[2],non_blocking=True) if packed[0]=='host' else packed[1]
    def context(self):return torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack) if self.enabled else contextlib.nullcontext()

def chunk_mlps(model,size):
    if not size:return
    for layer in model.get_base_model().model.language_model.layers:
        mlp=layer.mlp;original=mlp.forward
        def forward(self,x,original=original):
            if x.shape[1]<=size:return original(x)
            chunks=[]
            for i in range(0,x.shape[1],size):
                part=x[:,i:i+size]
                chunks.append(checkpoint(original,part,use_reentrant=False) if self.training and torch.is_grad_enabled() else original(part))
            return torch.cat(chunks,dim=1)
        mlp.forward=types.MethodType(forward,mlp)

def load_model(mode,rank,mlp_chunk):
    path,_=model_paths();kw={'torch_dtype':torch.bfloat16,'attn_implementation':'sdpa'}
    if mode=='qlora':kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['lm_head','acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector']),device_map={'':torch.cuda.current_device()})
    model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,**kw)
    for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
    if mode=='bf16':model.to('cuda')
    else:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        for p in model.parameters():
            if p.is_floating_point() and p.__class__.__name__!='Params4bit':p.data=p.data.to(torch.bfloat16)
    model.config.use_cache=False;model.model.language_model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});chunk_mlps(model,mlp_chunk)
    return model

def loss_for(model,item,offload):
    ids,mask,features,labels=item;base=model.get_base_model()
    with offload.context(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        embeddings=base.get_input_embeddings()(ids).clone();embeddings[mask]=features
        hidden=base.model(inputs_embeds=embeddings,use_cache=False,return_dict=True).last_hidden_state
    targets=labels[:,1:].reshape(-1);states=hidden[:,:-1].reshape(-1,hidden.shape[-1]);valid=targets!=-100;states=states[valid];targets=targets[valid]
    def chunk(x,y):return F.cross_entropy(base.lm_head(x.to(base.lm_head.weight.dtype)).float(),y,reduction='sum')
    total=states.new_zeros((),dtype=torch.float32)
    for i in range(0,len(targets),128):
        x,y=states[i:i+128],targets[i:i+128]
        total=total+(checkpoint(chunk,x,y,use_reentrant=False) if model.training else chunk(x,y))
    return total/len(targets)

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--mode',choices=['bf16','qlora'],default='bf16');p.add_argument('--bucket',type=int,default=600);p.add_argument('--steps',type=int,default=2);p.add_argument('--rank',type=int,default=16);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--offload',action='store_true');p.add_argument('--mlp-chunk',type=int,default=0);p.add_argument('--evaluate',action='store_true');p.add_argument('--save',action='store_true');args=p.parse_args()
    out=BASE/'runs'/args.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261022);random.seed(20261022)
    result={'config':vars(args),'status':'started','model_revision':MODEL_REVISION};events=[]
    def log(d):
        print(json.dumps(d),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
    try:
        cache=Cache();rows=[r for r in cache.rows if r['split']=='train' and r['bucket']==args.bucket and r.get('context_mode','none')=='none'];assert rows;random.shuffle(rows)
        result['manifest_sha256']=cache.metadata['manifest_sha256']
        log({'event':'loading','lengths':[{'id':r['id'],'tokens':r['token_length'],'seconds':r['audio_duration']} for r in rows]})
        model=load_model(args.mode,args.rank,args.mlp_chunk);trainable=[p for p in model.parameters() if p.requires_grad];offload=BoundaryOffload(args.offload)
        def evaluate():
            model.eval();r=next(r for r in cache.rows if r['split']=='validation')
            with torch.no_grad():v=loss_for(model,cache.item(r),offload).item()
            model.train();return v
        if args.evaluate:result['validation_loss_before']=evaluate();log({'event':'validation_before','loss':result['validation_loss_before']})
        model.train();optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            r=rows[step%len(rows)];log({'event':'step_start','step':step+1,'tokens':r['token_length'],'seconds':r['audio_duration']});torch.cuda.synchronize();t0=time.monotonic()
            loss=loss_for(model,cache.item(r),offload);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.0);assert torch.isfinite(norm) and norm>0
            optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();elapsed=time.monotonic()-t0
            e={'event':'step','step':step+1,'loss':loss.item(),'grad_norm':norm.item(),'wall_seconds':elapsed,'audio_seconds':r['audio_duration'],'tokens':r['token_length'],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'host_rss_gib':psutil.Process().memory_info().rss/2**30,'offloaded_gib_total':offload.total_bytes/2**30};events.append(e);log(e)
            if args.save:model.save_pretrained(out/f'checkpoint-{step+1}',safe_serialization=True)
        if args.evaluate:result['validation_loss_after']=evaluate();log({'event':'validation_after','loss':result['validation_loss_after']})
        measured=events[1:] if len(events)>1 else events;rtf=sum(e['audio_seconds'] for e in measured)/sum(e['wall_seconds'] for e in measured)
        result.update(status='completed',steps=events,audio_realtime_factor=rtf,hours_per_500h_epoch_one_gpu=500/rtf,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except Exception as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),steps=events);log({'event':'failed','error_type':type(e).__name__,'error':str(e)});traceback.print_exc()
    (out/'result.json').write_text(json.dumps(result,indent=2));log({'event':'complete','status':result['status']})
    if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
