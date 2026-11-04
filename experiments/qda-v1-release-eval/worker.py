"""Release evaluation: identical BF16 merged inference and fixed per-clip features."""
import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
import argparse,gc,json,pathlib,shutil,time
import numpy as np,torch
from common import HERE,BASE,sha,atomic_json,code_hashes
from repetition import repetition_period
from model_paths import model_paths,MODEL_REVISION
from scoring import metrics,validate_hypothesis
from final_report import output_issues
from peft import PeftModel
from torch.nn.attention import sdpa_kernel,SDPBackend
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--phase',choices=['main','long'],default='main')
    p.add_argument('--rerun-incomplete',action='store_true',help='Archive and rerun an incomplete prior result')
    p.add_argument('--unbounded-long',action='store_true',help='For long sessions, do not impose a wall-clock cutoff')
    args=p.parse_args()
    selection=json.loads((HERE/'selection.json').read_text());spec=selection['models'][args.model];hashes=code_hashes()
    if spec['adapter']:
        assert sha(pathlib.Path(spec['adapter'])/'adapter_model.safetensors')==spec['adapter_sha256']
        assert sha(pathlib.Path(spec['adapter'])/'adapter_config.json')==spec['adapter_config_sha256']
    rows=selection['rows'] if args.phase=='main' else selection['long_rows']
    rows=[json.loads((HERE/'ready'/(r['window_id']+'.json')).read_text()) for r in rows][args.rank::2]
    out=HERE/'evaluations'/args.model;out.mkdir(parents=True,exist_ok=True)
    pending=[]
    for row in rows:
        folder=out/row['window_id'];folder.mkdir(exist_ok=True);dest=folder/'result.json'
        if dest.exists():
            done=json.loads(dest.read_text())
            assert done['adapter_sha256']==spec['adapter_sha256'] and done['input_sha256']==row['input_sha256']
            if args.rerun_incomplete and args.phase=='long' and done.get('complete'):
                # A previously completed long decode is still valid: the
                # changed worker only removes a wall-clock cutoff. Preserve
                # its raw output and metrics rather than decoding it again.
                continue
            if args.rerun_incomplete and args.phase=='long' and not done.get('complete'):
                archive=folder/('attempt-'+str(int(time.time())))
                archive.mkdir()
                shutil.copy2(dest,archive/'result.json')
                for old_name in ('output.txt','output.partial.txt','hypothesis.json','reference.json'):
                    old_path=folder/old_name
                    if old_path.exists():shutil.copy2(old_path,archive/old_name)
            else:
                assert done['release_evaluation_sha256']==hashes
                continue
        old_name={'300h':'r64-full-300h','450h':'r64-full-450h','final':'r64-full-final','vanilla':'vanilla'}[args.model]
        old=BASE/('evaluations' if args.phase=='main' else 'long-evaluations')/old_name/row['window_id']
        prior=json.loads((old/'result.json').read_text()) if (old/'result.json').exists() else {}
        if prior.get('complete') and prior.get('termination')=='eos' and prior.get('adapter_sha256')==spec['adapter_sha256'] and prior.get('input_sha256')==row['input_sha256']:
            reference=json.loads((old/'reference.json').read_text());hyp=json.loads((old/'output.txt').read_text())
            assert reference==row['target'] and prior['model_revision']==MODEL_REVISION and prior['dtype']=='bfloat16' and prior['decoding']=='greedy'
            fm=json.loads((pathlib.Path(row['cache_root'])/(row['window_id']+'.features.json')).read_text())
            assert prior['features_sha256']==fm['sha256']
            shutil.copyfile(old/'output.txt',folder/'output.txt');atomic_json(folder/'reference.json',reference);atomic_json(folder/'hypothesis.json',hyp)
            measured=metrics(reference,hyp,folder,row['duration'],row['language'])
            assert measured['number_normalized_errors']==prior['metrics']['number_normalized_errors'] and measured['der_250ms']==prior['metrics']['der_250ms']
            result={**prior,'metrics':json.loads(json.dumps(measured,default=str)),'cohort':row['cohort'],'show':row['show'],
                    'release_evaluation_sha256':hashes,'reused_decode':True,'imported_from':str(old),'imported_result_sha256':sha(old/'result.json')}
            atomic_json(dest,result)
        else:pending.append(row)
    if not pending:return
    torch.cuda.set_device(args.rank);torch.set_num_threads(2);torch.manual_seed(20261023)
    path,tok=model_paths();processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
    model=VibeVoiceASRForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    for name in ('acoustic_tokenizer','semantic_tokenizer','acoustic_connector','semantic_connector'):setattr(model.model,name,None)
    if spec['adapter']:model=PeftModel.from_pretrained(model,spec['adapter']).merge_and_unload()
    model=model.to('cuda').eval();gc.collect();torch.cuda.empty_cache()
    for row in pending:
        wid=row['window_id'];folder=out/wid;cache=pathlib.Path(row['cache_root']);start=time.monotonic()
        fm=json.loads((cache/(wid+'.features.json')).read_text())
        assert sha(cache/(wid+'.input.pt'))==row['input_sha256']==fm['input_sha256']
        assert sha(cache/(wid+'.features.npz'))==fm['sha256'] and fm['contract_sha256']==row['feature_contract_sha256']
        result={'window_id':wid,'record_id':row['record_id'],'language':row['language'],'show':row['show'],'cohort':row['cohort'],
                'duration':row['duration'],'adapter_sha256':spec['adapter_sha256'],'input_sha256':row['input_sha256'],
                'features_sha256':fm['sha256'],'release_evaluation_sha256':hashes,'model_revision':MODEL_REVISION,
                'dtype':'bfloat16','decoding':'greedy','complete':False,'reused_decode':False}
        generated=[];termination='exception';tensors={}
        atomic_json(folder/'reference.json',row['target'])
        atomic_json(HERE/f'worker-{args.rank}.json',{'model':args.model,'phase':args.phase,'window_id':wid,'stage':'decoding','started_at':time.time()})
        try:
            inp=torch.load(cache/(wid+'.input.pt'),map_location='cpu',weights_only=True);n=int(torch.where(inp['labels']!=-100)[0][0])
            tensors['ids']=inp['ids'][:n].unsqueeze(0).cuda();mask=inp['mask'][:n].unsqueeze(0).cuda()
            with np.load(cache/(wid+'.features.npz'),allow_pickle=False) as packed:tensors['features']=torch.from_numpy(packed['features'].copy()).view(torch.bfloat16).cuda()
            stop=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');termination='token_limit'
            max_tokens=min(65536-n-8,max(4096,int(row['duration']*20)))
            wall_budget=None if args.phase=='long' and args.unbounded_long else (1200 if args.phase=='long' else 900)
            with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                embed=model.get_input_embeddings()(tensors['ids']).clone();embed[mask]=tensors['features']
                header=processor.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
                tensors['embed']=torch.cat([embed,model.get_input_embeddings()(torch.tensor([header],device='cuda'))],dim=1);del embed
                state=model.model(inputs_embeds=tensors['embed'],use_cache=True,return_dict=True);past=state.past_key_values
                for i in range(max_tokens):
                    token=model.lm_head(state.last_hidden_state[:,-1:]).argmax(-1).item()
                    if token==stop:termination='eos';break
                    generated.append(token)
                    if i%64==0:
                        period=repetition_period(generated)
                        if period:termination='repetition_abort';result['repetition_period_tokens']=period;break
                        if wall_budget is not None and time.monotonic()-start>wall_budget:
                            termination='wall_budget';break
                    state=model.model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True);past=state.past_key_values
                    if (i+1)%512==0:
                        (folder/'output.partial.txt').write_text(processor.tokenizer.decode(generated,skip_special_tokens=True))
                        atomic_json(HERE/f'worker-{args.rank}.json',{'model':args.model,'phase':args.phase,'window_id':wid,'stage':'decoding','generated_tokens':len(generated),'elapsed_seconds':time.monotonic()-start,'updated_at':time.time()})
            text=processor.tokenizer.decode(generated,skip_special_tokens=True);(folder/'output.txt').write_text(text)
            if termination!='eos':raise ValueError('Incomplete generation: '+termination)
            hypothesis=json.loads(text);issues=output_issues(hypothesis,row['duration'])
            validate_hypothesis(hypothesis,max(row['duration'],max(s['End'] for s in hypothesis)))
            result.update(output_issues=issues,strict_output_valid=not issues,der_uem_seconds=row['duration'],all_predicted_words_scored=True)
            result['metrics']=json.loads(json.dumps(metrics(row['target'],hypothesis,folder,row['duration'],row['language']),default=str))
            atomic_json(folder/'hypothesis.json',hypothesis);result['complete']=True
        except (ValueError,AssertionError,RuntimeError,KeyError,TypeError) as exc:
            result['error']=str(exc);(folder/'output.txt').write_text(processor.tokenizer.decode(generated,skip_special_tokens=True))
        result.update(termination=termination,wall_seconds=time.monotonic()-start,generated_tokens=len(generated),reference_sha256=sha(folder/'reference.json'))
        atomic_json(folder/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','release_evaluation_sha256','output_issues')}),flush=True)
        tensors.clear()
        if 'state' in locals():del state
        if 'past' in locals():del past
        if 'mask' in locals():del mask
        gc.collect();torch.cuda.empty_cache()
    atomic_json(HERE/f'worker-{args.rank}.json',{'model':args.model,'phase':args.phase,'stage':'completed','updated_at':time.time()})
if __name__=='__main__':main()
