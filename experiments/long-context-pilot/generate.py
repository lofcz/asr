import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse,json,pathlib,time,re,dataclasses,decimal,traceback
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from peft import PeftModel
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from train import Cache
from model_paths import model_paths
BASE=pathlib.Path(__file__).resolve().parent

def normalize(t):return ' '.join(re.findall(r'\w+',t.lower()))
def json_default(x):
    if isinstance(x,decimal.Decimal):return float(x)
    if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
    return str(x)

def score(r,segments,out):
    import meeteval
    from rapidfuzz.distance import Levenshtein
    ref=[dict(session_id=r['id'],speaker=str(s['speaker']),start_time=s['start'],end_time=s['end'],words=normalize(s['text'])) for s in r['segments']]
    hyp=[dict(session_id=r['id'],speaker=str(s['Speaker']),start_time=float(s['Start']),end_time=float(s['End']),words=normalize(s['Content'])) for s in segments]
    (out/'reference.seglst.json').write_text(json.dumps(ref,ensure_ascii=False));(out/'hypothesis.seglst.json').write_text(json.dumps(hyp,ensure_ascii=False))
    a=' '.join(s['words'] for s in ref).split();b=' '.join(s['words'] for s in hyp).split()
    metrics={'wer':Levenshtein.distance(a,b)/len(a),'reference_words':len(a),'hypothesis_words':len(b)}
    for name,fn in [('cpwer',lambda:meeteval.wer.cpwer(ref,hyp)),('tcpwer_5s',lambda:meeteval.wer.tcpwer(ref,hyp,collar=5))]:
        obj=fn()[r['id']];metrics[name]=float(obj.error_rate);metrics[name+'_details']=dataclasses.asdict(obj)
    uem_path=out/'evaluation.uem';uem_path.write_text(f"{r['id']} 1 0 {r['audio_duration']}\n")
    try:
        obj=meeteval.der.md_eval_22(ref,hyp,collar=.25,uem=uem_path)[r['id']];metrics['der_250ms']=float(obj.error_rate);metrics['der_details']=dataclasses.asdict(obj)
    except Exception as e:metrics['der_error']=str(e)
    return metrics

def assess_output(r,text,out):
    result={'valid_json':False}
    try:
        segments=json.loads(text)
        assert isinstance(segments,list) and segments, 'Expected nonempty segment array'
        result['valid_json']=True
        for s in segments:
            assert {'Start','End','Content'}<=s.keys(), 'Missing Start/End/Content'
            assert isinstance(s['Content'],str), 'Content is not text'
        events=[s for s in segments if re.fullmatch(r'\s*\[[^\[\]]+\]\s*',s['Content'])]
        speech=[s for s in segments if s not in events]
        result.update(segment_count=len(segments),bracketed_event_segments=len(events),
                      all_segments_have_speaker=all('Speaker' in s for s in segments),
                      valid_speech_schema=all('Speaker' in s for s in speech),
                      valid_timestamps=all(0<=float(s['Start'])<=float(s['End'])<=r['audio_duration']+1 for s in segments),
                      last_end_seconds=max(float(s['End']) for s in segments),
                      predicted_speakers=len({str(s.get('Speaker','__missing__')) for s in speech}))
        assert result['valid_speech_schema'], 'Missing speaker on spoken text'
        assert result['valid_timestamps'], 'Invalid timestamp bounds'
        result['metrics']=score(r,speech,out)
        result['metrics']['event_policy']='Bracket-only event labels excluded; Soniox reference contains speech only.'
    except Exception as e:
        result['parse_or_scoring_error']=str(e) or type(e).__name__
        traceback.print_exc()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--adapter');p.add_argument('--context-mode',choices=['none','provided','distractor'],default='none');p.add_argument('--split',default='validation');p.add_argument('--bucket',type=int,default=600);p.add_argument('--max-new-tokens',type=int,default=50000);p.add_argument('--prefill-chunk',type=int,default=32768);args=p.parse_args()
    out=BASE/'generation'/args.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261022)
    result={'config':vars(args),'status':'started','reference_type':'Soniox pseudo-reference; not manually verified ground truth'}
    def save(): (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=json_default))
    try:
        cache=Cache();r=next(r for r in cache.rows if r['split']==args.split and r['bucket']==args.bucket and r.get('context_mode','none')==args.context_mode);path,tokpath=model_paths();result.update(sample_id=r['id'],audio_seconds=r['audio_duration'],reference_speakers=r['speaker_count'],is_heldout=r['split']=='validation')
        model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        for n in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,n,None)
        model.to('cuda')
        if args.adapter:model=PeftModel.from_pretrained(model,args.adapter).merge_and_unload()
        model.eval();processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tokpath)
        ids,mask,features,labels=cache.item(r);n=r['prompt_length'];ids=ids[:,:n];mask=mask[:,:n]
        generated=[];past=None;stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');termination='token_limit'
        torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
        with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            embed=model.get_input_embeddings()(ids).clone();embed[mask]=features
            header=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
            embed=torch.cat([embed,model.get_input_embeddings()(torch.tensor([header],device='cuda'))],dim=1)
            # Explicit absolute positions make chunked prefill equivalent to whole-prompt causal attention.
            for start in range(0,embed.shape[1],args.prefill_chunk):
                part=embed[:,start:start+args.prefill_chunk]
                state=model.model(inputs_embeds=part,past_key_values=past,cache_position=torch.arange(start,start+part.shape[1],device='cuda'),use_cache=True,return_dict=True);past=state.past_key_values
            print(json.dumps({'event':'prefill_complete','prompt_tokens':embed.shape[1],'wall_seconds':time.monotonic()-t0}),flush=True)
            for i in range(args.max_new_tokens):
                token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
                if token==stop:termination='eos';break
                generated.append(token)
                if len(generated)>=256 and len(generated)%64==0:
                    if generated[-64:]==generated[-128:-64]==generated[-192:-128]==generated[-256:-192]:termination='repetition_abort';break
                state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True);past=state.past_key_values
                if (i+1)%1024==0:
                    text=processor.tokenizer.decode(generated,skip_special_tokens=True);(out/'output.partial.txt').write_text(text)
                    ends=re.findall(r'"End"\s*:\s*([\d.]+)',text);progress={'event':'generation_progress','tokens':len(generated),'elapsed_seconds':round(time.monotonic()-t0,2),'latest_end_seconds':float(ends[-1]) if ends else None};print(json.dumps(progress),flush=True);(out/'progress.json').write_text(json.dumps(progress))
        torch.cuda.synchronize();text=processor.tokenizer.decode(generated,skip_special_tokens=True);(out/'output.txt').write_text(text)
        result.update(status='generated',termination=termination,generated_tokens=len(generated),wall_seconds=time.monotonic()-t0,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,valid_json=False)
        result.update(assess_output(r,text,out))
        result['status']='completed';save();print(json.dumps(result,default=json_default),flush=True)
    except Exception as e:
        result.update(status='failed',error=str(e),traceback=traceback.format_exc());save();traceback.print_exc();raise
if __name__=='__main__':main()
