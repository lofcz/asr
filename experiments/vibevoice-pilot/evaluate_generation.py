import argparse,json,pathlib,time,re,os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import torch
from peft import PeftModel
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from benchmark import Cache
from model_paths import model_paths, MODEL_REVISION
BASE=pathlib.Path(__file__).resolve().parent

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def words(text):return re.findall(r'\w+',text.lower())
def main():
    p=argparse.ArgumentParser();p.add_argument('--bucket',type=int,default=30);p.add_argument('--adapter');p.add_argument('--name',required=True);args=p.parse_args();torch.set_num_threads(2)
    model_path,tokenizer_path=model_paths()
    model=VibeVoiceASRForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
    model.to('cuda')
    if args.adapter:model=PeftModel.from_pretrained(model,args.adapter).merge_and_unload()
    model.eval();processor=VibeVoiceASRProcessor.from_pretrained(model_path,language_model_pretrained_name=tokenizer_path);cache=Cache();rows=[]
    for lang in ('cs','sk'):rows.append(next(r for r in cache.rows if r['split']=='validation' and r['bucket']==args.bucket and r['language']==lang))
    results=[]
    for r in rows:
        ids,mask,features,labels=cache.item(r,'cuda');n=r['prompt_length'];ids=ids[:,:n];mask=mask[:,:n]
        header=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
        with torch.inference_mode():
            embed=model.get_input_embeddings()(ids).clone();embed[mask]=features;tail=model.get_input_embeddings()(torch.tensor([header],device='cuda'));embed=torch.cat([embed,tail],dim=1)
            torch.cuda.synchronize();t0=time.monotonic()
            state=model.model(inputs_embeds=embed,use_cache=True,return_dict=True)
            generated=[];stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
            for _ in range(1024):
                token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
                if token==stop:break
                generated.append(token)
                state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=state.past_key_values,use_cache=True,return_dict=True)
            output=[generated]
            torch.cuda.synchronize();elapsed=time.monotonic()-t0
        text=processor.tokenizer.decode(output[0],skip_special_tokens=True);parsed=None;error=None
        try:
            parsed=json.loads(text);assert isinstance(parsed,list)
            hypothesis=' '.join(s['Content'] for s in parsed)
        except Exception as e:error=str(e);hypothesis=text
        reference=' '.join(s['text'] for s in r['segments']);ref=words(reference);hyp=words(hypothesis)
        result={'id':r['id'],'language':r['language'],'audio_seconds':r['audio_duration'],'generated_tokens':len(output[0]),'wall_seconds':elapsed,'valid_json':error is None,'parse_error':error,'reference':reference,'hypothesis':hypothesis,'raw_output':text,'reference_words':len(ref),'word_errors':distance(ref,hyp),'pseudo_reference_wer':distance(ref,hyp)/len(ref),'reference_speakers':r['speaker_count'],'predicted_speakers':len({str(s.get('Speaker')) for s in parsed}) if error is None else None}
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('reference','hypothesis','raw_output')},ensure_ascii=False),flush=True)
    (BASE/'runs'/f'generation-{args.name}.json').write_text(json.dumps({'adapter':args.adapter,'note':'Tiny machine-labeled held-out smoke test, not human WER or diarization quality evaluation','samples':results},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
