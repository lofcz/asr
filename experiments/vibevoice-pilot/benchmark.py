import os,argparse,json,pathlib,time,random,gc,traceback,contextlib
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from model_paths import model_paths, MODEL_REVISION
BASE=pathlib.Path(__file__).resolve().parent

class CachedLoss(torch.nn.Module):
    def __init__(self,model,head_chunk=128):
        super().__init__();self.model=model;self.head_chunk=head_chunk
    def forward(self,ids,mask,features,labels):
        base=self.model.get_base_model()
        embeddings=base.get_input_embeddings()(ids).clone()
        embeddings[mask]=features.to(embeddings.dtype)
        hidden=base.model(inputs_embeds=embeddings,use_cache=False,return_dict=True).last_hidden_state
        y=labels[:,1:].reshape(-1);h=hidden[:,:-1].reshape(-1,hidden.shape[-1]);valid=y!=-100;h=h[valid];y=y[valid]
        def loss_chunk(x,target):return F.cross_entropy(base.lm_head(x.to(base.lm_head.weight.dtype)).float(),target,reduction='sum')
        total=h.new_zeros((),dtype=torch.float32)
        for i in range(0,len(y),self.head_chunk):
            x=h[i:i+self.head_chunk];target=y[i:i+self.head_chunk]
            total=total+(checkpoint(loss_chunk,x,target,use_reentrant=False) if self.training else loss_chunk(x,target))
        return total/len(y)

class Cache:
    def __init__(self):
        assert not (BASE/'cache/.building').exists(), 'Feature cache is incomplete'
        self.rows=[json.loads(l) for l in (BASE/'cache/index.jsonl').read_text().splitlines()]
        self.arrays={n:np.load(BASE/'cache'/f'{n}.npy',mmap_mode='r') for n in ('tokens','mask','labels','features')}
    def item(self,r,device):
        o,n=r['token_offset'],r['token_length'];f,m=r['feature_offset'],r['feature_length']
        tensors=[torch.from_numpy(np.array(self.arrays[k][o:o+n],copy=True)).to(device=device,dtype=d).unsqueeze(0) for k,d in [('tokens',torch.long),('mask',torch.bool)]]
        features=torch.from_numpy(np.array(self.arrays['features'][f:f+m],copy=True)).to(device=device,dtype=torch.bfloat16)
        labels=torch.from_numpy(np.array(self.arrays['labels'][o:o+n],copy=True)).to(device=device,dtype=torch.long).unsqueeze(0)
        return *tensors,features,labels

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-audio',action='store_true');p.add_argument('--mode',choices=['bf16','qlora'],default='bf16');p.add_argument('--bucket',type=int,default=30);p.add_argument('--steps',type=int,default=8);p.add_argument('--accum',type=int,default=4);p.add_argument('--name',required=True);p.add_argument('--eval-count',type=int,default=4);p.add_argument('--rank',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4);args=p.parse_args()
    local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:torch.distributed.init_process_group('nccl',device_id=device)
    torch.set_num_threads(2);torch.manual_seed(20261022);random.seed(20261022)
    out=BASE/'runs'/args.name;out.mkdir(parents=True,exist_ok=True)
    result={'config':vars(args),'world_size':world,'rank':rank,'status':'starting'}
    def record(x):
        with (out/f'events-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
        if rank==0:print(json.dumps(x),flush=True)
    try:
        model_path,tokenizer_path=model_paths()
        kwargs={'torch_dtype':torch.bfloat16,'attn_implementation':'sdpa'}
        if args.mode=='qlora':kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['lm_head','acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector']),device_map={'':local})
        model=VibeVoiceASRForConditionalGeneration.from_pretrained(model_path,**kwargs)
        if not args.raw_audio:
            for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
        if args.mode=='bf16':model.to(device)
        else:
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
            # PEFT casts nonquantized weights to FP32; BF16 embedding/head suffice here.
            model.get_input_embeddings().to(torch.bfloat16);model.lm_head.to(torch.bfloat16)
        model.config.use_cache=False;model.model.language_model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
        model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        trainable=[p for p in model.parameters() if p.requires_grad]
        result.update(trainable_parameters=sum(p.numel() for p in trainable),resident_parameters=sum(p.numel() for p in model.parameters()),initial_gpu_gib=torch.cuda.memory_allocated()/2**30)
        wrapper=CachedLoss(model).to(device);cache=Cache()
        result['cache_manifest_sha256']=json.loads((BASE/'cache/metadata.json').read_text())['manifest_sha256']
        if args.raw_audio:
            import soundfile as sf
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
            processor=VibeVoiceASRProcessor.from_pretrained(model_path,language_model_pretrained_name=tokenizer_path)
        def get_item(r):
            ids,mask,features,labels=cache.item(r,device)
            if args.raw_audio:
                audio,sr=sf.read(BASE/'dataset'/r['audio_path'],dtype='float32')
                enc=processor._process_single_audio(audio,sampling_rate=sr,add_generation_prompt=False,use_streaming=True)
                with torch.no_grad():features=model.get_base_model().encode_speech(torch.from_numpy(enc['speech']).unsqueeze(0).to(device)).squeeze(0)
            return ids,mask,features,labels
        rows=[r for r in cache.rows if r['split']=='train' and r['bucket']==args.bucket];random.shuffle(rows)
        val_all=[r for r in cache.rows if r['split']=='validation' and r['bucket']==args.bucket]
        by_lang=[[r for r in val_all if r['language']==lang] for lang in sorted({r['language'] for r in val_all})]
        val=[r for group in zip(*by_lang) for r in group][:args.eval_count]
        def evaluate():
            wrapper.eval();total=0;count=0
            with torch.no_grad():
                for r in val:
                    loss=wrapper(*get_item(r));total+=loss.item()*r['target_tokens'];count+=r['target_tokens']
            wrapper.train();return total/count
        result['validation_loss_before']=evaluate();record({'event':'loaded',**result})
        wrapped=DDP(wrapper,device_ids=[local],find_unused_parameters=False) if world>1 else wrapper
        optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01)
        wrapper.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();metrics=[];micro=0
        for step in range(args.steps):
            torch.cuda.synchronize();t0=time.monotonic();audio=0;ntokens=0;loss_sum=0
            for j in range(args.accum):
                r=rows[(micro*world+rank)%len(rows)];micro+=1
                ctx=wrapped.no_sync() if world>1 and j<args.accum-1 else contextlib.nullcontext()
                with ctx:
                    loss=wrapped(*get_item(r));assert torch.isfinite(loss),'Nonfinite loss';(loss/args.accum).backward()
                audio+=r['audio_duration'];ntokens+=r['token_length'];loss_sum+=loss.detach().item()/args.accum
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.0);assert torch.isfinite(norm),'Nonfinite gradient';assert norm>0,'Zero gradient'
            optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();elapsed=time.monotonic()-t0
            stats=torch.tensor([audio,ntokens,elapsed],device=device,dtype=torch.float64)
            if world>1:
                torch.distributed.all_reduce(stats[:2],op=torch.distributed.ReduceOp.SUM);torch.distributed.all_reduce(stats[2:],op=torch.distributed.ReduceOp.MAX)
            entry={'event':'step','step':step+1,'loss':loss_sum,'grad_norm':norm.item(),'audio_seconds':stats[0].item(),'tokens':int(stats[1].item()),'wall_seconds':stats[2].item(),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
            metrics.append(entry);record(entry)
        result['validation_loss_after']=evaluate();warm=metrics[1:] if len(metrics)>1 else metrics
        result.update(status='completed',peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,measured_steps=len(warm),audio_seconds=sum(m['audio_seconds'] for m in warm),wall_seconds=sum(m['wall_seconds'] for m in warm))
        result['audio_realtime_factor']=result['audio_seconds']/result['wall_seconds'];result['hours_per_500h_epoch']=500/result['audio_realtime_factor']
        if rank==0:model.save_pretrained(out/'adapter',safe_serialization=True)
        record({'event':'complete',**result})
    except Exception as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());record({'event':'failed','error':str(e)});traceback.print_exc()
    finally:
        (out/f'result-rank{rank}.json').write_text(json.dumps(result,indent=2))
        if world>1:torch.distributed.destroy_process_group()
    if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
